import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import zarr as za

from freqtrade.configuration import TimeRange
from freqtrade.constants import (
    DEFAULT_DATAFRAME_COLUMNS,
    DEFAULT_TRADES_COLUMNS,
    ListPairsWithTimeframes,
)
from freqtrade.data.converter import clean_ohlcv_dataframe

from .idatahandler import IDataHandler, TradeList


logger = logging.getLogger(__name__)


def ts(val, adj=False):
    return pd.Timestamp(val * 1e9, tz="UTC") if adj else pd.Timestamp(val, tz="UTC")


class ZarrDataHandler(IDataHandler):

    _columns = DEFAULT_DATAFRAME_COLUMNS
    _storedir = "zarr"
    _store: Optional[za.DirectoryStore] = None
    _group: Optional[za.Group] = None
    _compressor = za.Blosc(cname="zstd", clevel=2)
    _force_overwrite = False

    @property
    def store(self):
        if not self._store:
            self._store = za.DirectoryStore(self._datadir / self._storedir)
        return self._store

    @property
    def group(self):
        if not self._group:
            self._group = za.open_group(store=self.store)
        return self._group

    @classmethod
    def get_group(cls, datadir: Path, mode="r"):
        store = za.DirectoryStore(datadir)
        return za.open_group(store=store, mode=mode)

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :return: List of Tuples of (pair, timeframe)
        """
        group = cls.get_group(datadir)
        pairs_tf = []
        for pair in group.keys():
            p = pair.replace("_", "/")
            for tf in group[pair]:
                pairs_tf.append((p, tf))
        return pairs_tf

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        for the specified timeframe
        :param datadir: Directory to search for ohlcv files
        :param timeframe: Timeframe to search pairs for
        :return: List of Pairs
        """

        return [
            pair.replace("_", "/") for pair in cls.get_group(datadir / "ohlcv").keys()
        ]

    def _save(
        self, key: str, data: pd.DataFrame, kind: str, td: Optional[float] = 0,
    ):
        """ save data into group checking bounds for contiguity """
        if kind == "ohlcv":
            col = "date"
            columns = self._columns
        elif kind == "trades":
            col = "timestamp"
            columns = DEFAULT_TRADES_COLUMNS
        if key in self.group and self.group[key].shape[0] > 0:
            arr = self.group[key]
            assert isinstance(arr, za.Array)
            saved_first_date = arr[0, col]
            saved_last_date = arr[-1, col]
            data_first_date = data[col].iloc[0]
            data_last_date = data[col].iloc[-1]
            if kind == "ohlcv":
                data_first_date = data_first_date.value
                data_last_date = data_last_date.value
            # don't append if there are missing rows
            self._check_contiguity(
                data_first_date, data_last_date, saved_first_date, saved_last_date, td
            )
            # if appending data
            if data_first_date >= saved_first_date:
                if self._force_overwrite:
                    # when overwriting get the index where data starts overwriting storage
                    # we count the number of candles using the difference
                    offset = int((data_first_date - saved_first_date) // td)
                else:
                    # when not overwriting get the index where data has new values
                    data_offset = pd.to_numeric(data[col]).searchsorted(
                        saved_last_date, side="right"
                    )
                    data = data.iloc[data_offset:]
                    offset = len(arr)
            # inserting requires overwrite
            else:  # fetch the saved data and combine with new one
                # fetch saved data starting after the last date of the new data
                # which has to be >= saved_first_date because we checked for contig
                if td:
                    saved_offset = int(max(0, data_last_date - saved_first_date) // td)
                else:
                    saved_offset = arr[:, col].searchsorted(
                        data_first_date, side="right"
                    )
                saved_data = self._arr_to_df(arr[saved_offset:], kind)
                data = data[columns]
                self._upcast_time(data, col)
                self._upcast_time(saved_data, col)
                data = pd.concat(
                    [data, saved_data], axis=0, ignore_index=True, copy=False
                )
                offset = 0
            if len(data):
                arr.resize(offset + len(data))
                # arr.set_orthogonal_selection(slice(offset, None), self._to_records(data, kind))
                arr[offset:] = self._to_records(data, kind)
            else:
                logger.debug("no new data was found for %s", key)
        else:
            if key in self.group:
                del self.group[key]
            offset = 0
            # Zarr (0.2.4) supports records, allowing to query by fields
            # (although fields aren't required at creation since they are inferred
            # from the dtype)
            logger.debug("creating zarr dataset for key %s, len: %s", key, len(data))
            self.group.create_dataset(
                key, data=self._to_records(data, kind), compressor=self._compressor,
            )

    @staticmethod
    def _check_contiguity(
        data_first_date, data_last_date, saved_first_date, saved_last_date, td
    ):
        """ TODO: We enforce contiguity...however because of compression sparsity wouldn't
        cause much overhead...
        """
        if data_first_date > saved_last_date + td:
            raise ValueError(
                "Data stored ends at {} while new data starts at {}, "
                "contiguity can't be satisfied".format(
                    ts(saved_last_date), ts(data_first_date),
                )
            )
        elif (
            data_first_date < saved_first_date
            and data_last_date + td < saved_first_date
        ):
            raise ValueError(
                "Data stored starts at {} while new data ends at {}, "
                "contiguity can't be satisfied".format(
                    ts(saved_first_date), ts(data_last_date),
                )
            )

    def ohlcv_store(
        self, pair: str, timeframe: str, data: pd.DataFrame, cleaned=False
    ) -> None:
        """
        Store data.
        :param pair: Pair - used to generate filename
        :timeframe: Timeframe - used to generate filename
        :data: Dataframe containing OHLCV data
        :return: None
        """
        td = pd.Timedelta(timeframe).value
        key = self._pair_ohlcv_key(pair, timeframe)
        self._save(key, data, "ohlcv", td)
        # clear cache
        self._del_key_cleaned(key)

    @classmethod
    def _arr_to_df(cls, arr: za.Array, kind: str):
        if kind == "ohlcv":
            columns = cls._columns
            col = "date"
        else:
            columns = DEFAULT_TRADES_COLUMNS
            col = "timestamp"
        if len(arr):
            df = pd.DataFrame.from_records(arr[columns])
            cls._upcast_time(df, col)
            return df
        else:
            return pd.DataFrame(columns=columns)

    @classmethod
    def _downcast_time(cls, data: pd.DataFrame, col: str):
        if data[col].dtype not in ("int32", "float32", "int64", "float64"):
            data[col] = data[col].astype("datetime64[ns]")

    @classmethod
    def _upcast_time(cls, data: pd.DataFrame, col: str):
        if data[col].dtype in ("int32", "float32", "int64", "float64"):
            data[col] = data[col].astype("datetime64[ns, UTC]")

    @classmethod
    def _to_records(cls, data: pd.DataFrame, kind: str):
        if kind == "ohlcv":
            col = "date"
            columns = cls._columns
            col_types = float
        elif kind == "trades":
            col = "timestamp"
            columns = DEFAULT_TRADES_COLUMNS
            col_types = {"*": float, col: float, "side": str, "type": str, "id": int}
        with pd.option_context("mode.chained_assignment", None):
            # trades data is ms...and we are storing as ns
            # it is wrong but the conversion happens after...
            cls._downcast_time(data, col)
        return data[columns].to_records(index=False, column_dtypes=col_types)

    def _del_key(self, key):
        dlt = False
        if key in self.group:
            del self.group[key]
            dlt = True
        self._del_key_cleaned(key)
        return dlt

    def _del_key_cleaned(self, key):
        key_c = f"{key}_cleaned"
        if key_c in self.group:
            del self.group[key_c]

    def _ohlcv_load(
        self,
        pair: str,
        timeframe: str,
        timerange: Optional[TimeRange] = None,
        cleaned=False,
    ) -> pd.DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        key = self._pair_ohlcv_key(pair, timeframe)

        if key not in self.group:
            return pd.DataFrame(columns=self._columns)
        if cleaned:
            key_c = f"{key}_cleaned"
            if key_c not in self.group:
                logger.debug("cleaning data as requested for pair %s", pair)
                # load raw data WITHOUT timerange
                raw = self._ohlcv_load(pair, timeframe, cleaned=False)
                cleaned_data = clean_ohlcv_dataframe(
                    raw, timeframe, pair=pair, fill_missing=True, drop_incomplete=False
                )
                self._save(key_c, cleaned_data, "ohlcv", pd.Timedelta(timeframe).value)
                logger.debug("saved cleaned data to %s", key_c)
            arr = self.group[key_c]
            logger.debug(
                "loaded cleaned data for %s, start: %s, end: %s",
                key_c,
                ts(arr[0, "date"]),
                ts(arr[-1, "date"]),
            )
        else:
            arr = self.group[key]

        # only trim by timerange if data is cleaned
        # otherwise can't guess ranges correctly
        if timerange and cleaned:
            start, stop = arr.get_coordinate_selection(([0, -1]), fields=["date"])
            # because of struct arrays, unpack the tuples
            start, stop = start[0], stop[0]
            td = pd.Timedelta(timeframe)
            if timerange.starttype == "date":
                date_offset = max(0, timerange.startts * 1e9 - start)
                arr_start = int(date_offset // td.value)
                logger.debug("timerange query start: %s", ts(timerange.startts, True))
            else:
                arr_start = 0
            if timerange.stoptype == "date":
                arr_period = arr_start + int(
                    (timerange.stopts - timerange.startts) * 1e9 // td.value
                )
                logger.debug("timerange query stop: %s", ts(timerange.stopts, True))
            else:
                arr_period = None
        else:
            arr_start = arr_period = None
        # pairdata = arr.get_orthogonal_selection((slice(arr_start, arr_period)))
        logger.debug(
            "loading pairdata with range: %s-%s , cleaned: %s",
            arr_start,
            arr_period,
            cleaned,
        )
        pairdata = arr[arr_start:arr_period]

        if list(pairdata.dtype.names) != self._columns:
            raise ValueError("Wrong dataframe format")

        df = pd.DataFrame.from_records(pairdata)
        # cast date from numeric to datetime, with UTC timezone
        self._upcast_time(df, "date")
        return df

    def ohlcv_purge(self, pair: str, timeframe: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :param timeframe: Timeframe (e.g. "5m")
        :return: True when deleted, false if file did not exist.
        """
        key = self._pair_ohlcv_key(pair, timeframe)
        return self._del_key(key)

    def ohlcv_append(self, pair: str, timeframe: str, data: pd.DataFrame) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        """
        raise NotImplementedError()

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs for which trade data is available in this
        :param datadir: Directory to search for ohlcv files
        :return: List of Pairs
        """
        return [
            pair.replace("_", "/") for pair in cls.get_group(datadir / "trades").keys()
        ]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        key = self._pair_trades_key(pair)
        data = pd.DataFrame(data, columns=DEFAULT_TRADES_COLUMNS)
        self._save(key, data, "trades")

    def trades_append(self, pair: str, data: TradeList):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(
        self, pair: str, timerange: Optional[TimeRange] = None
    ) -> TradeList:
        """
        Load a pair from h5 file.
        :param pair: Load trades for this pair
        :param timerange: Timerange to load trades for - currently not implemented
        :return: List of trades
        """
        key = self._pair_trades_key(pair)

        if key not in self.group:
            return []
        arr = self.group[key]
        if timerange:
            timestamp = arr[:, "timestamp"]
            if timerange.starttype == "date":
                arr_start = timestamp.searchsorted(timerange.startts)
            else:
                arr_start = 0
            if timerange.stoptype == "date":
                arr_period = timestamp.searchsorted(timerange.stopts, side="right")
            else:
                arr_period = None
        else:
            arr_start = 0
            arr_period = None
        # pairdata = arr.get_orthogonal_selection((slice(arr_start, arr_period)))
        pairdata = arr[arr_start:arr_period]

        if list(pairdata.dtype.names) != DEFAULT_TRADES_COLUMNS:
            raise ValueError("Wrong dataframe format")

        self._upcast_time(pairdata, "timestamp")
        return pairdata.tolist()

    def trades_purge(self, pair: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :return: True when deleted, false if file did not exist.
        """
        key = self._pair_trades_key(pair)
        return self._del_key(key)

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str) -> str:
        return f"{pair}/ohlcv/tf_{timeframe}"

    @classmethod
    def _pair_trades_key(cls, pair: str, cleaned: bool) -> str:
        return f"{pair}/trades"
