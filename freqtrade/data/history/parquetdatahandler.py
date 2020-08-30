import logging
import re
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Dict

import numpy as np
from pandas import DataFrame, read_json, to_datetime, read_parquet

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import (DEFAULT_DATAFRAME_COLUMNS, ListPairsWithTimeframes)
from freqtrade.data.converter import trades_dict_to_list

from .idatahandler import IDataHandler, TradeList

logger = logging.getLogger(__name__)


class ParquetDataHandler(IDataHandler):

    _columns = DEFAULT_DATAFRAME_COLUMNS
    _compression = "gzip"

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :return: List of Tuples of (pair, timeframe)
        """
        _tmp = [re.search(r'^(\d+\S+)\/([a-zA-Z_]+)', p.name)
                for p in datadir.glob(f"*.{cls._get_file_extension()}")]
        return [(match[1].replace('_', '/'), match[2]) for match in _tmp
                if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        for the specified timeframe
        :param datadir: Directory to search for ohlcv files
        :param timeframe: Timeframe to search pairs for
        :return: List of Pairs
        """

        _tmp: List[str] = list(map(str, datadir.glob(f"{timeframe}/*")))
        # Check if regex found something and only return these results
        return [match[0].replace("_", "/") for match in _tmp if match]

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        Store data in json format "values".
            format looks as follows:
            [[<date>,<open>,<high>,<low>,<close>]]
        :param pair: Pair - used to generate filename
        :timeframe: Timeframe - used to generate filename
        :data: Dataframe containing OHLCV data
        :return: None
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if len(data) > 0:
            data["month"] = data["date"].dt.year.astype(str) + data["date"].dt.month.astype(str)
            kwargs: Dict = {}
            data.to_parquet(str(filename), compression=self._compression, partition_cols=["month"], **kwargs)

    def _ohlcv_load(
        self, pair: str, timeframe: str, timerange: Optional[TimeRange] = None,
    ) -> DataFrame:
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
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if not filename.exists():
            return DataFrame(columns=self._columns)
        # only pyarrow engine support directories
        pairdata = read_parquet(filename, engine='pyarrow', columns=self._columns)
        # pairdata.drop(columns="month", inplace=True)
        pairdata = pairdata.astype(
            dtype={
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float",
            }
        )
        pairdata["date"] = to_datetime(
            pairdata["date"], unit="ms", utc=True, infer_datetime_format=True
        )
        return pairdata

    def ohlcv_purge(self, pair: str, timeframe: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :param timeframe: Timeframe (e.g. "5m")
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_data_filename(self._datadir, pair, timeframe)
        if filename.exists():
            rmtree(filename)
            return True
        return False

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame) -> None:
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
        _tmp = list(map(str, datadir.glob(f"trades/*")))
        # Check if regex found something and only return these results to avoid exceptions.
        return [match[0].replace("_", "/") for match in _tmp if match]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        from datetime import datetime
        import pandas as pd
        filename = self._pair_trades_filename(self._datadir, pair)
        df = DataFrame(data)
        df.columns = df.columns.astype(str)
        dt = pd.to_datetime(df['0'], unit='ms').dt
        df["day"] = dt.year.astype(str) + dt.month.astype(str) + dt.day.astype(str)
        df.to_parquet(str(filename), compression=self._compression, partition_cols=["day"])

    def trades_append(self, pair: str, data: TradeList):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param timerange: Timerange to load trades for - currently not implemented
        :return: List of trades
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        tradesdata = read_parquet(filename).drop(columns=["day"]).values.tolist() if filename.exists() else []
        return tradesdata

    def trades_purge(self, pair: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        if filename.exists():
            # filename.unlink()
            rmtree(filename)
            return True
        return False

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str) -> Path:
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath("{}/{}/".format(timeframe, pair_s))
        return filename

    @classmethod
    def _get_file_extension(cls):
        return "parquet"

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str) -> Path:
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath(f"trades/{pair_s}/")
        return filename
