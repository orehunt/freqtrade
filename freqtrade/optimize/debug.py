from freqtrade.exceptions import OperationalException
from freqtrade.optimize.backtest_constants import *  # noqa ignore=F405
from freqtrade.optimize.backtest_utils import replace_values
from freqtrade.strategy.interface import SellType
from typing import Tuple, Dict, Any
import os
import pandas as pd
import numpy as np


class BacktestDebug:
    bts_loc: Dict
    backtesting: object
    events: pd.DataFrame
    ref = {}

    def _validate_events(self, events_buy: pd.DataFrame, events_sell: pd.DataFrame):
        try:
            assert len(events_buy) == len(events_sell)
        except AssertionError:
            # find locations where a sell is after two or more buys
            flcols = self._cols(self.bts_loc)
            if isinstance(events_buy, pd.DataFrame):
                events_buy, events_sell = events_buy.values, events_sell.values
            events_buy = pd.DataFrame(
                events_buy,
                columns=self.bts_loc,
                index=events_buy[:, self.bts_loc["ohlc_ofs"]],
            )
            events_sell = pd.DataFrame(
                events_sell,
                columns=self.bts_loc,
                index=events_sell[:, self.bts_loc["ohlc_ofs"]],
            )
            leb = len(events_buy)
            les = len(events_sell)
            nxt = True
            for n, i in enumerate(events_buy.index.values[1 : min(leb, les)], 1):
                nxt = (events_sell.iloc[n].name >= i) & (
                    events_sell.iloc[n - 1].name < i
                ) & (events_buy.iloc[n].name <= events_sell.iloc[n].name)
                if not nxt:
                    print(events_buy.iloc[n])
                    print(events_buy.iloc[n - 1 : min(leb, n + 1)])
                    print(events_sell.iloc[n - 1 : min(les, n + 1)], end="\n")
                    print("buy:", len(events_buy), "sell:", len(events_sell))
                    raise OperationalException(
                        f"Buy and sell events not matching at {n}"
                    )
            if nxt:
                events_buy, events_sell = (
                    events_buy[flcols],
                    events_sell[flcols],
                )
                print("\nHEAD\n")
                print(events_buy.iloc[:10], "\n", events_sell.iloc[:10])
                print("\nTAIL\n")
                print(events_buy.iloc[-10:], "\n", events_sell.iloc[-10:])
                print("buy:", len(events_buy), "sell:", len(events_sell))
                raise OperationalException("Buy and sell events not matching")

    @staticmethod
    def start_pyinst(interval=0.001, skip=0):
        from pyinstrument import Profiler

        global counter
        if skip:
            if "counter" not in globals():
                counter = 0
            if counter >= skip:
                skip = 0
                counter = -1
            else:
                counter += 1
        if not skip and "profiler" not in globals():
            global profiler
            profiler = Profiler(interval=interval)
            profiler.start()

    @staticmethod
    def stop_pyinst(ex=True, keep=False):
        global profiler
        if "profiler" not in globals():
            return
        is_skipping = "counter" in globals() and counter != -1
        if not is_skipping:
            profiler.stop()
        if not is_skipping or (is_skipping and counter == -1):
            print(profiler.output_text(unicode=True, color=True))
            if not keep:
                exit()
            else:
                del profiler

    @staticmethod
    def print_pyinst():
        global profiler
        print(profiler.output_text(unicode=True, color=True))
        exit()

    def _debug_opts(self):
        # import os
        # import psutil
        # pid = psutil.Process(os.getpid())
        pd.set_option("display.max_rows", 1000)
        self.cols = [
            "buy",
            "sell",
            "ohlc",
            "pair",
            "bought_or_sold",
            "next_sold_ofs",
            "trigger_bought_ofs",
            "trigger_ofs",
            "trigger_ofs_b",
            "stoploss_triggered",
            "stoploss_rate",
            "low",
            "roi_profit",
            "roi_triggered",
            "roi_triggered_b",
            "roi_profit_b",
            "date_trigger",
            "trigger_ofs_max",
            "last_trigger",
            "last_trigger_diff",
            "ohlc_ofs",
        ]
        self.counter = 0

    def _cols(self, src: Any):
        columns = src.columns.values if isinstance(src, pd.DataFrame) else src
        flt_cols = []
        for col in self.cols:
            if col in columns:
                flt_cols.append(col)
        return flt_cols

    def _asdf(self, arr, df):
        return pd.DataFrame(arr, columns=df.columns, index=arr[:, df.columns.get_loc(df.index.name or df.columns[0]) or 0])

    def _load_results(self) -> pd.DataFrame:
        import pickle

        with open("/tmp/backtest.pkl", "rb+") as fp:
            return pickle.load(fp)

    def _dump_results(self, results: pd.DataFrame):
        import pickle

        with open("/tmp/backtest.pkl", "rb+") as fp:
            pickle.dump(results, fp)

    def _cmp_indexes(
        self,
        where: Tuple[str, str],
        results: pd.DataFrame,
        saved_results: pd.DataFrame,
        ex=False,
        print_data=False,
        filter_fsell=True,
        print_inc=True,
            cols=None,
    ):
        """ find all the non matching indexes between results, differentiate between not present (to include)
        and invalid (to exclude) """
        to_inc, to_exc = [], []
        key_0 = where[0]
        key_1 = where[1]
        key_pair_0 = f"pair_{key_0}"
        key_pair_1 = f"pair_{key_1}"
        events = self.backtesting.events if "events" in dir(self.backtesting) else self.events
        if not isinstance(results, pd.DataFrame):
            results = pd.DataFrame(
                results,
                columns=cols,
                index=results[:, cols["ohlc_ofs"]],
            )

        if len(results) == 0 and len(saved_results) == 0:
            return
        # we don't consider missing force sells as wrong
        if filter_fsell:
            if events is not None:
                end_candles = events[
                    events["next_sold_ofs"].isin(self.backtesting.pairs_ofs_end)
                ]["ohlc"].values
            else:
                end_candles = []
            results = results.loc[
                (results["sell_reason"].values != SellType.FORCE_SELL)
                & ~(results[key_0].isin(end_candles))
            ]
            saved_results = saved_results.loc[
                (saved_results["sell_reason"].values != SellType.FORCE_SELL)
                & ~(saved_results[key_1].isin(end_candles).values)
            ]

        # stock results are sorted, so align indexes
        if results["pair"].dtype == float:
            results["pair"] = replace_values(
                self.backtesting.pairs_idx,
                self.backtesting.pairs_name,
                results["pair"].values,
            )
            events["pair"] = replace_values(
                self.backtesting.pairs_idx,
                self.backtesting.pairs_name,
                events["pair"].values,
            )
        results = results.sort_values(by=["pair", key_0])
        saved_results = saved_results.sort_values(by=["pair", key_1])
        # have to match indexes to the correct pairs, so make sets of (index, pair) tuples
        where_0 = list(
            zip(
                results[key_0]
                .fillna(method="pad")
                .astype(saved_results[key_1].dtype)
                .values,
                results["pair"].values,
            )
        )
        where_1 = list(zip(saved_results[key_1].values, saved_results["pair"].values,))
        results[key_pair_0], saved_results[key_pair_1] = where_0, where_1
        where_0_set = set(where_0)
        where_1_set = set(where_1)

        for i in results[key_pair_0].values:
            if i not in where_1_set:
                to_exc.append(i)
        for i in saved_results[key_pair_1]:
            if i not in where_0_set:
                to_inc.append(i)
        if print_data:
            print(to_inc, to_exc)
            print(saved_results.set_index(key_pair_1).loc[to_inc[:10]])
            print(results.set_index(key_pair_0).loc[to_exc[:10]])
        if to_inc:
            print(
                "TO INCLUDE (total): ",
                to_inc[0] if len(to_inc) > 0 else None,
                f"({len(to_inc)})",
            )
        if to_exc:
            print(
                "TO EXCLUDE (total): ",
                to_exc[0] if len(to_exc) > 0 else None,
                f"({len(to_exc)})",
            )
        # print the first event that is wrong and the range of the
        # boughts_to_sold df (saved in events) that includes it
        if to_inc and print_inc:
            first = to_inc[0]
        elif to_exc:
            first = to_exc[0]
        else:
            first = None

        if first is not None:
            idx = (
                (events["ohlc"].values == int(first[0]))
                & (events["pair"].values == first[1])
            ).argmax()
            print(
                events.iloc[max(0, idx - 50) : min(idx + 100, len(events))][
                    self._cols(events)
                ]
            )
            s_idx = (
                (saved_results["pair"].values == first[1])
                & (saved_results[key_1].values == int(first[0]))
            ).argmax()
            print(
                saved_results.iloc[
                    max(0, s_idx - 10) : min(s_idx + 10, len(saved_results))
                ][saved_results.columns.difference(["pair_open_index"])]
            )
            print("idx:", idx, ", count:", self.counter)
            if ex:
                exit()

    def _print_events(self, events_buy, events_sell, bts_df, bts_loc, pos):
        cols = self._cols(bts_df)
        if not isinstance(events_buy, pd.DataFrame):
            events_buy = pd.DataFrame(
                events_buy, columns=bts_loc, index=events_buy[:, bts_loc["ohlc_ofs"]]
            )
            events_sell = pd.DataFrame(
                events_sell,
                columns=bts_loc,
                index=events_sell[:, bts_loc["ohlc_ofs"]],
            )
        s_res = dbg._load_results().sort_values(by=["pair", "open_index"])
        print(events_buy.iloc[pos][cols])
        print(events_sell.iloc[pos][cols])
        print(bts_df.iloc[pos][cols])
        print(s_res.iloc[pos])
        print(len(events_buy), len(events_sell))
        exit()

    def _check_counter(self, at=0) -> bool:
        self.counter += 1
        return self.counter < at

    def _wrap_backtest(
        self, processed: Dict[str, pd.DataFrame], **kwargs,
    ) -> pd.DataFrame:
        """ debugging """
        # results to debug
        cls = self.backtesting
        results = None
        # results to compare against
        saved_results = None
        # if some epoch far down the (reproducible) iteration needs debugging set it here
        check_at = int(os.getenv("FQT_CHECKAT", "0"))
        if check_at and self._check_counter(check_at):
            return cls.empty_results
        # if testing only one epoch use "store" once then set it to "load"
        cache = os.getenv("FQT_DEBUG", "")
        if cache == "load":
            results = cls.vectorized_backtest(processed)
            saved_results = self._load_results()
        elif cache == "store":
            self._dump_results(cls.backtest_vanilla(processed, **kwargs,))
            exit()
        elif cache == "1":
            results = cls.vectorized_backtest(processed)
        elif cache == "0":
            results = cls.backtest_vanilla(processed, **kwargs)
        else:
            results = cls.vectorized_backtest(processed)
            saved_results = cls.backtest_vanilla(processed, **kwargs,)
        if cache not in  ("0", "1"):
            idx_name = os.getenv("FQT_CMP_IDX", "open_index")
            self._cmp_indexes((idx_name, idx_name), results, saved_results)
        if os.getenv("FQT_SHOW_RES", ""):
            print(
                results.iloc[:10],
                "\n",
                saved_results.sort_values(by=["pair", "open_index"]).iloc[:10],
            )
        # return saved_results
        return results


dbg = BacktestDebug() if os.getenv("FQT_DEBUG", False) else None
