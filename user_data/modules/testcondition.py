# testcondition
import os
import pickle, json
import re
from functools import reduce
from typing import List, Tuple

import numpy as np
from pandas import DataFrame

from user_data.modules import sequencer as ts
from user_data.modules import toolkit as tk
from user_data.modules.toolkit import q

andf = lambda x, y: x & y
orf = lambda x, y: x | y
plusf = lambda x, y: x + y

PICKLE_PATH = "/tmp/.freqtrade/testcondition/"
RESULTS_PATH = "user_data/hyperopt_results"
PARAMS_PATH = f"{RESULTS_PATH}/params.json"
POS_PATH = f"{PICKLE_PATH}/pos.json"


def cache(action: str, m: dict, ret=False):
    """store a predefined evaluated set of conditions"""
    pair_pickle_path = PICKLE_PATH + m["pair"].replace("/", "-") + ".pickle"
    if m["runmode"].name in ["BACKTEST", "PLOT"]:
        try:
            os.remove(pair_pickle_path)
        except OSError:
            pass
        return False
    if action == "r":
        try:
            with open(pair_pickle_path, "rb") as f:
                ret = pickle.load(f)
                return ret
        except Exception as e:
            # print(e)
            try:
                os.mkdir(PICKLE_PATH)
            except Exception:
                pass
            return False
    elif action == "w":
        try:
            with open(pair_pickle_path, "wb") as f:
                pickle.dump(ret, f)
            return True
        except Exception as e:
            print(e)
            return False


def core_pick(conds: List[np.ndarray], d: DataFrame, params: dict, rel: bool) -> np.ndarray:
    """map numbered parameters to a list of conditions and returns the concat"""
    picked = []
    for p in params:
        if params[p] is True:
            n = re.search("[0-9]+$", p)
            if n is not None:
                picked.append(conds[int(n.group().lstrip("0")) - 1])
    if len(picked) > 0:
        if rel == "AND":
            ret = reduce(lambda x, y: x & y, picked)
        else:
            ret = reduce(lambda x, y: x | y, picked)
    else:
        ret = np.full(len(d), False)
    return ret


def parametrize(d: DataFrame, params: dict, g, sent=[], n=-1, o=-1) -> Tuple[np.array, np.array]:
    """logic based parameter evaluation based on parameters names"""
    gx = np.array(len(d))

    if "ppo01" in params and params["ppo01"] != "":
        g.append(q(params["ppo01"]))
    if "cmo01" in params and params["cmo01"] != "":
        g.append(q(params["cmo01"]))
    if "adx01" in params and params["adx01"] != "":
        g.append(q(params["adx01"]))
    if "pdi01" in params and params["pdi01"] != "":
        g.append(q(params["pdi01"]))
    if "mdi01" in params and params["mdi01"] != "":
        g.append(q(params["mdi01"]))
    if "cci" in params:
        cond = q("cci<" + str(params["cci01"])) & q("cci>" + str(params["cci02"]))
        g.append(cond)
    if "core01" in params:
        conds: List[np.ndarray]
        conds = [q("mp<-tn"), q("mp<tn"), q("mp<+tn")]
        gx = core_pick(conds, d, params, "AND")
    # inds = [
    # 'ad'
    # params['hloc01'],
    # params['hloc02'],
    # params['hloc03'],
    # params['hloc04'],
    # ]
    ## defined in spaces.txt and indicator_space()
    # prefixes = [
    # 'smp',
    # 'shift',
    # 'ratio',
    # 'cci',
    # ]
    # gx = (
    # )
    # gx = ts.addcond(d, params=params)

    return g, gx


def above(ret: np.ndarray, limit: int, total: int) -> np.ndarray:
    """check if an array of ints is higher than a percentage limit"""
    return ret >= (limit * total)


def test_cond(d: DataFrame, params: dict, m: dict, sgn: str) -> np.array:
    tk._p = m["periods"]
    tk.qc = {}
    group_parameters = []
    core_cond = os.getenv("FQT_CORE_COND")
    no_parameter_x = True
    skip_replay = os.getenv("FQT_SKIP_REPLAY")
    if not no_parameter_x:
        parameter_x = np.full(len(d), True)

    # if len(params) > 0:
    #     group_parameters, parameter_x = parametrize(d, params, g)

    if core_cond:
        core = []
        ret = reduce(plusf, core)
        n_evals = len(core)
    else:
        ret = cache("r", m)

    if ret is False:
        if os.path.exists(POS_PATH):
            with open(POS_PATH, "r") as fp:
                given = json.load(fp)
        else:
            given = {}
        sent, ands, n, o, n_evals, p_evals = ts.replay(d, sgn=sgn, given=given, op="+")
        if not no_parameter_x:
            prefix = []
            prefix = reduce(andf, prefix)
            sent = prefix & sent
        ret = [sent, ands, n, o, n_evals, p_evals]
        cache("w", m, ret)
    elif not core_cond:
        sent, ands, n, o, n_evals, p_evals = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]

    if m["runmode"].name == "HYPEROPT":
        # if any(s in ['buy', 'sell'] for s in m['spaces']): # if n counter is negative its a wrap
        if "spaces" not in m and not os.getenv("FQT_SKIP_COND"):
            ret = ts.addcond(d, sgn, sent, ands, n, o, params, m["periods"], op="+")
        elif not core_cond:
            ret = ret[0]
        if not no_parameter_x:
            if type(ret).__name__ != "tuple":
                if (
                    hasattr(parameter_x, "size")
                    and parameter_x.size > 0
                    or hasattr(parameter_x, "__len__")
                    and len(parameter_x) != 0
                ):
                    ret = ret & parameter_x
            else:
                ret = parameter_x
    else:
        ret = sent

    if len(group_parameters) > 0:
        rdc_parameters = reduce(andf, group_parameters)
        ret = ret & rdc_parameters

    if type(ret) != np.ndarray:
        print("empty test condition, defaulting to false")
        ret = np.full(len(d), False)

    ## negate?
    # ret = ~(ret)
    # above?
    if not n_evals:
        n_evals = len(ts.tconds[n][1])
    ret = above(ret, params["evals0"] if "evals0" in params else n_evals, n_evals)
    return ret
