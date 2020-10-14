from datetime import datetime
from functools import partial
from typing import Protocol
import logging

from freqtrade.data.dataprovider import DataProvider

logger = logging.getLogger(__name__)


class GetPairDf(Protocol):
    def __call__(pair: str, timeframe: str, last_date=None, *args, **kwargs):
        ...


class PairsCache:
    pairs_data = {}
    _get_pairs_data = None
    dp: DataProvider

    def get_data(
        self,
        pair: str,
        timeframe: str,
        last_date: datetime,
        fn: GetPairDf,
        *args,
        **kwargs,
    ):
        sector = (pair, timeframe)
        key = last_date.timestamp()
        c_key, c_data = self.pairs_data.get(sector, (0, None))
        if sector not in self.pairs_data:
            self.pairs_data[sector] = (
                key,
                fn(pair=pair, timeframe=timeframe, *args, **kwargs),
            )
            logger.debug(
                "storing cache new: %s, sec: %s", key, sector,
            )
            ret = self.pairs_data[sector][1]
        elif key > c_key:
            logger.debug(
                "deleting cache sec:  %s, old: %s , new: %s", sector, c_key, key,
            )
            del self.pairs_data[sector]
            self.pairs_data[sector] = (
                key,
                fn(pair=pair, timeframe=timeframe, *args, **kwargs),
            )
            ret = self.pairs_data[sector][1]
        elif key < c_key:
            ret = c_data.loc[:key]
            logger.debug(
                "returning trimmed data from cache, queried key: %s, len: %s",
                key,
                len(ret),
            )
        else:
            ret = c_data
        return ret

    @property
    def get_pairs_data(self):
        if not self._get_pairs_data:
            self._get_pairs_data = partial(
                self.get_data, fn=self.dp.get_pair_dataframe,
            )
        return self._get_pairs_data
