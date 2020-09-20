import os.path
from os import getenv
import logging
import json
import pickle
import glob
import warnings
from datetime import datetime
from functools import reduce

import numpy as np

import talib
from freqtrade.state import RunMode

from user_data.modules.sequencer import replay
from pandas import DataFrame, Series, Timedelta, date_range
from typing import Dict

# from modin.pandas import DataFrame, Series, Timedelta, date_range
# from statsmodels.tsa import stattools
from technical.indicators import ichimoku
import user_data.modules.toolkit as tk

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
andf = lambda x, y: x & y
orf = lambda x, y: x | y
mm_type = {  ## min max type
    "cci": 1,
    "cmo": 1,
    "bop": 1,
    "macd": 1,
    "macds": 1,
    "macdh": 1,
    "adx": 1,
    "pdi": 1,
    "mdi": 1,
    "mom": 1,
    "vol": 1,
    "ad": 1,
    "ppo": 1,
    "ard": 1,
    "aru": 1,
    "rsi": 1,
    "stok": 1,
    "stod": 1,
    "trg": 1,
    "htpe": 1,
    "htph": 1,
    "htin": 1,
    "htqu": 1,
    "htsi": 1,
    "htls": 1,
    "sar": 0,
    "avgp": 0,
    "medp": 0,
    "typp": 0,
    "wclp": 0,
    "tn": 0,
    "kj": 0,
    "sa": 0,
    "sb": 0,
    "sh": 0,
    "lo": 0,
    "ck": 0,
    "emas": 0,
    "emam": 0,
    "h": 0,
    "l": 0,
    "o": 0,
    "c": 0,
    "tl": 0,
    "mp": 0,
    "mpp": 0,
    "ub": 0,
    "mb": 0,
    "lb": 0,
}

## tuning for window parameters based on timeframe interval
periods_dict = {
    "1m": {1: 1, 2: 3, 3: 5, 4: 10, 6: 15, 8: 20, 12: 30, 16: 45, 24: 60, 48: 120},
    "5m": {1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 9, 12: 12, 16: 18, 24: 24, 48: 48},
    "15m": {1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 8, 12: 12, 16: 16, 24: 24, 48: 48},
    "30m": {1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 8, 12: 12, 16: 16, 24: 24, 48: 48},
    "1h": {1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 8, 12: 12, 16: 16, 24: 24, 48: 48},
}

userdir = getenv("FQT_USERDIR") or "user_data"
decide_config_dir = userdir + "/strategies/decide"
dcache = userdir + "/backtest_data/"
dcache_hyperopt = "/tmp/.freqtrade/hyperopt_cache/"
dcache_live = "/tmp/.freqtrade/"


def trim(d: DataFrame, p: dict, s: str) -> Series:
    """removes the empty rows at the end of the dataframe"""
    return d[s].iloc[: -p[48]]


def minmax(d: DataFrame, p: dict, params=None) -> DataFrame:
    """compute max and min stddev over a rolling p12 period"""
    try:
        std_c = (talib.STDDEV(trim(d, p, "c"), p[12] * 10) / d["c"]).rolling(p[12])
        d["_max"] = 1 + std_c.max().values / 10
        d["_min"] = 1 - std_c.min().values / 10
        for m in mm_type:
            if mm_type[m] is 1:
                stdd = (talib.STDDEV(trim(d, p, m), p[12] * 10) / d[m]).rolling(p[12])
                d[m + "_max"] = 1 + stdd.max().values / 10
                d[m + "_min"] = 1 - stdd.min().values / 10
    except Exception as e:
        print("Error min_max: ", m if "m" in vars() else 0, e)
    return d


def periods(time: str) -> dict:
    """convert timeframe ratio based on 1h"""
    if time not in periods_dict:
        return periods_dict["1h"]
    else:
        return periods_dict[time]


def populate_indicators(cls, d: DataFrame, m: dict) -> DataFrame:
    tf = cls.config["timeframe"]
    tk._p = periods(tf)
    tk.qc = {}  # reset query cache because of previous pairs
    p = tk._p
    ta = talib.abstract
    # d = env(d)
    ## load dataframe from pickle
    cd = cache(cls, d, "r", m, p)
    if type(cd) == DataFrame:
        d = cd
    else:
        ## compute some indicators that are not friendly with NaN entries
        ## add rows for negative shift (sa/sb)
        idx_start = d.index.max() + 1
        d_fut = DataFrame(columns=d.columns, index=range(idx_start, idx_start + p[48]))
        d_fut["date"] = date_range(
            start=d["date"].iloc[-1] + Timedelta(tf), freq=Timedelta(tf), periods=p[48]
        )
        d = d.append(d_fut)

    # d = cudf.DataFrame.from_pandas
    tamacd = ta.MACD(d, timeperiod=p[12])
    d["macd"] = tamacd["macd"]
    d["macds"] = tamacd["macdsignal"]
    d["macdh"] = tamacd["macdhist"]

    d["mp"] = ta.MIDPOINT(d, timeperiod=p[12])
    d["mpp"] = ta.MIDPRICE(d, timeperiod=p[12])
    d["tl"] = ta.HT_TRENDLINE(d)

    bb = ta.BBANDS(d, matype=talib.MA_Type.MAMA, timeperiod=p[6])
    d["ub"] = bb["upperband"]
    d["mb"] = bb["middleband"]
    d["lb"] = bb["lowerband"]

    d["adx"] = ta.ADX(d, timeperiod=p[48])
    d["pdi"] = ta.PLUS_DI(d, timeperiod=p[48])
    d["mdi"] = ta.MINUS_DI(d, timeperiod=p[48])
    d["sar"] = ta.SAR(d, acceleration=0.1, maximum=1.0)
    d["mom"] = ta.MOM(d, timeperiod=p[24])

    t_tn = p[16]
    t_kj = p[48]
    t_sb = p[48] * 2
    t_dsp = p[24]
    ichi = ichimoku(d, t_tn, t_kj, t_sb, t_dsp)
    d["tn"] = ichi["tenkan_sen"]
    d["kj"] = ichi["kijun_sen"]
    d["sa"] = ichi["senkou_span_a"]
    d["sb"] = ichi["senkou_span_b"]

    # some pairs data doesn't have decimals and are type guessed to ints...
    d["volume"] = d["volume"].astype(float)
    d["ad"] = ta.AD(d)
    d["vol"] = d["volume"]
    d["h"] = d["high"]
    d["l"] = d["low"]
    d["o"] = d["open"]
    d["c"] = d["close"]

    d["sh"] = ta.SMA(d, timeperiod=p[3])
    d["lo"] = ta.SMA(d, timeperiod=p[6])

    d["emas"] = ta.EMA(d, timeperiod=p[8])
    d["emam"] = ta.EMA(d, timeperiod=p[16])

    d["bop"] = ta.BOP(d)
    d["cci"] = ta.CCI(d, timeperiod=p[4])
    d["cmo"] = ta.CMO(d, timeperiod=p[4])
    d["ppo"] = ta.PPO(d, fastperiod=p[12], slowperiod=p[24], matype=talib.MA_Type.KAMA)
    aroon = ta.AROON(d, timeperiod=p[16])
    d["ard"] = aroon["aroondown"]
    d["aru"] = aroon["aroonup"]
    d["rsi"] = ta.RSI(d, timeperiod=p[12])
    stochf = ta.STOCHF(d, fastk_period=p[6], fastd_period=p[3], fastd_matype=talib.MA_Type.MAMA)
    d["stok"] = stochf["fastk"]
    d["stod"] = stochf["fastd"]

    d["trg"] = ta.TRANGE(d)

    d["avgp"] = ta.AVGPRICE(d)
    d["medp"] = ta.MEDPRICE(d)
    d["typp"] = ta.TYPPRICE(d)
    d["wclp"] = ta.WCLPRICE(d)

    d["htpe"] = ta.HT_DCPERIOD(d)
    d["htph"] = ta.HT_DCPHASE(d)
    htpc = ta.HT_PHASOR(d)
    d["htin"] = htpc["inphase"]
    d["htqu"] = htpc["quadrature"]
    htsi = ta.HT_SINE(d)
    d["htsi"] = htsi["sine"]
    d["htls"] = htsi["leadsine"]
    d["htgr"] = ta.HT_TRENDMODE(d)

    # d['cdli'] = ta.CDL3INSIDE(d)
    # d['cdls'] = ta.CDL3LINESTRIKE(d)
    # d['pr'] = ta.CDL3OUTSIDE(d)

    d = minmax(d, p)

    # d['t1'] = talib.BETA(trim(d['h']), trim(d['l']))
    # d['t2'] = talib.BETA(trim(d['o']), trim(d['c']))
    # d['t1'] = talib.STDDEV(trim(d['sar']), len(trim(d)))
    # d['asd'] = talib.LINEARREG(trim(d['c']), timeperiod=10)

    # if 'buy_guards' not in d or cls.config['runmode'].name == 'plot':
    #     d['buy_guards'] = buy_guards(cls, d)
    #     d['buy_triggers'] = buy_triggers(cls, d)
    #     d['sell_guards'] = sell_guards(cls, d)
    #     d['sell_triggers'] = sell_triggers(cls, d)

    # d['tg'] = group(d, {}, cls.config)

    ## switch guards and triggers to async
    # d = compute_rules(cls, d, m)
    return d


def compute_guards_triggers(m: Dict, d: DataFrame, action: str) -> DataFrame:
    if action not in ("buy", "sell"):
        raise Exception(f"{action} is not a valid trade action")
    if action == "buy":
        if "buy_guards" not in d:
            d["buy_guards"] = buy_guards(m, d)
        if "buy_triggers" not in d:
            d["buy_triggers"] = buy_triggers(m, d)
    elif action == "sell":
        if "sell_guards" not in d:
            d["sell_guards"] = sell_guards(m, d)
        if "sell_triggers" not in d:
            d["sell_triggers"] = sell_triggers(m, d)
    return d


def eval_params(m: Dict, d: DataFrame, sigtype: str, op: str = "&"):
    tf = m["timeframe"]
    tk._p = periods(tf)  # needed by eval
    ret = []
    ## load into list of configurations
    configs = []
    mode = "test" if m["runmode"].name in ("HYPEROPT", "BACKTEST", "PLOT") else ""
    for p in glob.glob(f"{decide_config_dir}/{mode}/{sigtype}*{tf}*.json"):
        # print(f'evaluating file {p}...')
        try:
            with open(p, "r") as fp:
                configs.append(json.load(fp))
        except Exception as e:
            print(f"error: {e}")
            return np.full(len(d), False)

    sgn, sgt = sigtype.split("_")
    if sgt == "trigger":
        glue = "|"
    elif sgt == "guard":
        glue = "&"
    evald = []
    for c in configs:
        sent, _, _, _, n_evals, p_evals = replay(d, prev_params=c, op="+", sgn=sgn)
        evald.append((sent, n_evals, p_evals))
    return evald


def buy_triggers(m: Dict, d: DataFrame) -> np.array:
    # self = d.scope['self']
    ret = []
    # list of evaled triggers (arr, maxv)
    eps = eval_params(m, d, "buy_trigger", "+")
    for e in eps:
        # ret.append(e[0] == e[1])
        ret.append(e[0] > e[1] * e[2])
    # triggers are ORed default to false
    if len(ret) < 1:
        return np.full(len(d), False)
    else:
        return reduce(andf, ret)


def buy_guards(self, d: DataFrame) -> np.array:

    c = []

    c.append(np.full(len(d), True))

    ## debugging
    # last_true_condition(d, c, 'buy_guard_false')
    # count_conditions(d, c, 'buy_guards_count')

    ret = reduce(andf, c)
    return ret


def sell_guards(self, d: DataFrame) -> np.array:
    c = []
    c.append(np.full(len(d), True))
    ## debugging
    # self.last_true_condition(d, c, 'sell_guard_false')
    # self.count_conditions(d, c, 'sell_guards_count')
    ret = reduce(andf, c)
    return ret


def sell_triggers(m: Dict, d: DataFrame) -> np.array:
    ret = []
    # list of evaled triggers (arr, maxv)
    eps = eval_params(m, d, "sell_trigger", "+")
    for e in eps:
        # ret.append(e[0] == e[1])
        ret.append(e[0] > e[1] * 0.992)
    # triggers are ORed default to false
    if len(ret) < 1:
        return np.full(len(d), False)
    else:
        return reduce(andf, ret)


def cache(self, d, action, m: dict(), p: dict()):
    ## don't cache when running live
    runmode = self.config["runmode"].name
    live = runmode in ("LIVE", "DRY_RUN")
    if live:
        path = dcache_live
    elif runmode == "HYPEROPT":
        path = dcache_hyperopt
    else:
        path = dcache
    if not os.path.exists(path):
        os.mkdir(path)

    if "pickle_cache" not in m:
        m["pickle_cache"] = path + m["pair"].replace("/", "-") + ".pickle"
    if action == "r":
        if os.path.exists(m["pickle_cache"]):
            try:
                with open(m["pickle_cache"], "rb") as f:
                    current_date = d["date"].iloc[-1]
                    d = pickle.load(f)
                    cached_date = d["date"].iloc[-p[48] - 1]
                    if live:
                        # pickle is outdated
                        if cached_date != current_date or np.isnan(d["close"].iloc[-p[48] - 1]):
                            logger.info(
                                "Skipping cached pickle: "
                                f"cached_date - {cached_date}, "
                                f"current_date - {current_date}"
                            )
                            return False
                return d
            except Exception as e:
                logger.error(f"Caching error: {e}")
                return False
        else:
            return False
    elif action == "w":
        try:
            with open(m["pickle_cache"], "wb") as f:
                pickle.dump(d, f)
            return True
        except Exception as e:
            logger.error(f"Caching error: {e}")
            return False


def last_true_condition(self, d: DataFrame, conditions: [], name: str) -> np.array:
    """find the first False conditions in the conditions list per each df row"""
    d[name] = 0
    counter = d[name]
    for n, c in enumerate(conditions):
        newcounter = counter.astype(int) + c.astype(int)
        for i, c in counter.iteritems():
            if counter[i] >= 0 and newcounter[i] > c:
                counter[i] = newcounter[i]
            else:
                if counter[i] >= 0:
                    counter[i] = -n
    return counter


def count_conditions(self, d: DataFrame, conditions: [], name: str) -> np.array:
    """count the total number of True conditions per each df row"""
    d[name] = reduce(lambda x, y: x.astype(int) + y.astype(int), conditions)
    ret = d[name].apply(lambda x: x == len(conditions))
    return ret


def is_buy_timeout(self, d: DataFrame) -> bool:
    """True if the a predefined amount of time passed from the last candle update"""
    if self.config["runmode"] in [RunMode.LIVE, RunMode.DRY_RUN]:
        frame_date = d.loc[d.index.max(), "date"]
        time_buffer = Timedelta(self.config["timeframe"]) / 2
        now = datetime.now(frame_date.tzinfo)
        timeout = (frame_date + time_buffer).replace(day=now.day, month=now.month, year=now.year)
        return timeout <= now
    else:
        return False


# def compute_rules(self, d: DataFrame, m: dict) -> DataFrame:
#     if 'buy_guards' not in d:
#         loop = asyncio.get_event_loop()
#         tasks = buy_guards(self, d), buy_triggers(self, d), sell_guards(self, d), sell_triggers(self, d)
#         bg, bt, sg, st = loop.run_until_complete(asyncio.gather(*tasks))
#         d['buy_guards'] = bg
#         d['buy_triggers'] = bt
#         d['sell_guards'] = sg
#         d['sell_triggers'] = st
#         cache(self, d, 'w', m, p)
#     else:
#         d['tg'] = group(d, {})
#     return d

# def normalize_df(self, d: DataFrame):
#     """ drop empty values otherwise can't dump to json,
#     only after generating all the indicators
#     """
#     d.drop(d.index[-d.p[48]:], inplace=True)
#     return
