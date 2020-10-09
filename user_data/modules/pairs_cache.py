from datetime import datetime
from functools import partial
from typing import Protocol

from freqtrade.data.dataprovider import DataProvider


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
        if sector not in self.pairs_data:
            self.pairs_data[sector] = (
                key,
                fn(pair=pair, timeframe=timeframe, *args, **kwargs),
            )
            # logger.debug(f"storing cache {key}, {sector}, {self.pairs_data[sector][0]}")
        elif key != self.pairs_data[sector][0]:
            del self.pairs_data[sector]
            self.pairs_data[sector] = (
                key,
                fn(pair=pair, timeframe=timeframe, *args, **kwargs),
            )
            # logger.debug(f"deleting cache {key}, {sector}, {self.pairs_data[sector][0]}")
        return self.pairs_data[sector][1].copy()

    @property
    def get_pairs_data(self):
        if not self._get_pairs_data:
            self._get_pairs_data = partial(
                self.get_data, fn=self.dp.get_pair_dataframe,
            )
        return self._get_pairs_data
