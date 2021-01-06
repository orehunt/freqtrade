import os
import pickle
import numpy as np
import logging
import json
import re
from glob import glob
from pathlib import Path

from time import sleep
from typing import Any, List, NamedTuple
from pandas import DataFrame
from typing import NamedTuple
from functools import reduce
from enum import Enum
from pickle import dump as pdump, load as pload

from user_data.modules.helper import read_json_file

PREDS = "user_data/hyperopt_results/preds/"  # make sure it is on tmpfs
MEAN_DIR = "/tmp/.freqtrade/mean"
STATE_DIR = "/tmp/.freqtrade/state"
logger = logging.getLogger(__name__)

mi, ma = {}, {}
labels = {}
if not os.path.exists(MEAN_DIR):
    os.makedirs(MEAN_DIR)
else:
    list(map(os.remove, glob(f"{MEAN_DIR}/*")))
for l in ("buy_gr", "buy_lo", "sell_gr", "sell_lo"):
    labels[l] = {}
    for m in ("me", "qt"):
        labels[l][m] = []
if not os.path.exists(STATE_DIR):
    os.makedirs(STATE_DIR)
pid = os.getpid()


class IndMode(Enum):
    weights = 1
    limits = 2
    sig = 3
    limlist = 4
    rotate = 5
    multi_weights = 6
    pair = 7


class State(NamedTuple):
    mode: IndMode
    defv: None
    cond: str
    inds: list
    k: int
    se: int
    sgn: List[str]


#
def liweme(cond: str, ind_path="user_data/strategies/ind", glob={}) -> Any:
    """ returns limits and weights of pair """
    if glob:
        lwm = {}
        lwm["limits"] = []
        for f in Path(ind_path).glob(glob["limits"]):
            lwm["limits"].append(read_json_file(f))
        lwm["weights"] = []
        for f in Path(ind_path).glob(glob["weights"]):
            w = read_json_file(f)
            # get the limit reference from the last weight
            m = re.match("(.*)\.(.*)\.(.*)", list(w.keys())[-1])
            n_l = m[2] if m else 0
            lwm["weights"].append((int(n_l), w))
        lwm["multi"] = []
        for f in Path(ind_path).glob(glob["multi"]):
            lwm["multi"].append(read_json_file(f))
        return lwm
    else:
        limits = read_json_file(f"{ind_path}/limits/{cond}.json")
        weights = read_json_file(f"{ind_path}/weights/{cond}.json")
    return {"limits": limits, "weights": weights}


#
def choose_func(mode: IndMode) -> callable:
    """ Set the function that will compute the signals depending on mode """
    if mode is IndMode.limits:
        return eval_signals_limits
    elif mode is IndMode.weights:
        return eval_signals_mode
    elif mode is IndMode.multi_weights:
        return eval_signals_multi_mode
    elif mode is IndMode.pair:
        return eval_signals_pair_mode


def min_max_predicate(n: int, preds: list, arr: np.ndarray) -> dict:
    if len(arr) > 0:
        mi = np.nanmin(arr)
        ma = np.nanmax(arr)
    else:
        return preds
    if len(preds) < n + 1:
        space = []
        if len(arr) < 2 or mi == ma:
            tp = "catg"
        elif type(arr[-1]) in (int, np.int64):
            tp = "int"
        elif type(arr[-1]) in (float, np.float64):
            tp = "real"
        else:
            tp = "catg-f"
            space = [v for v in arr]
        preds.append({"min": None, "max": None, "type": tp, "space": space})
    try:
        preds[n]["min"] = min(preds[n]["min"], mi)
        preds[n]["max"] = max(preds[n]["min"], ma)
    except TypeError:
        preds[n]["min"] = mi
        preds[n]["max"] = ma
    return preds


#
def sgn_from_env() -> List[str]:
    try:
        return os.getenv("FQT_SGN").split(",")
    except AttributeError:
        return []


def dump_state(state: State, name="state"):
    with open(f"{STATE_DIR}/{name}.pickle", "wb") as sp:
        pdump(state, sp)


#
def load_state(name="state") -> State:
    for tries in range(3):
        try:
            with open(f"{STATE_DIR}/{name}.pickle", "rb") as sp:
                return pload(sp)
        except EOFError:
            sleep(0.5)


#
def aggregate_predicates(cond: str):
    """
    reads all the pickles in a directories that represent the evaluated predicates
    (indicators or conditions), for each pair, get the min and max values and aggregates
    the absolute min and max of all of them in a single list
    """
    preds = []
    if os.getenv("FQT_PREDS"):
        preds_dir = f"{PREDS}/{cond}"
        if os.path.exists(preds_dir):
            for rt, dr, fl in os.walk(preds_dir):
                for fp in fl:
                    pair = os.path.basename(fp).replace("-", "/").replace(".pickle", "")
                    with open(f"{rt}/{fp}", "rb") as pr:
                        pair_preds = pickle.load(pr)
                        for n, arr in enumerate(pair_preds):
                            preds = min_max_predicate(n, preds, arr)
    return preds


def write_predicates(
    preds_names: list,
    mode: IndMode,
    d: DataFrame,
    m: dict,
    cond: str,
    populate_predicates: callable,
) -> dict:
    """
    convert the indicators into a list or compute the predicates list
    save the results in a per pair pickle file
    """
    buy, sell = [], []
    pair = m["pair"]
    if not preds_names and mode != IndMode.sig:
        preds = populate_predicates(d, m, cond)
    else:
        preds = [d[p].values for p in preds_names]
    # used to generate spaces
    preds_dir = f"{PREDS}/{cond}"
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
    with open(f"{preds_dir}/{pair.replace('/', '-')}.pickle", "wb") as pr:
        pickle.dump(preds, pr)


def read_predicates(pair: str, cond: str):
    """ read the predicates of a pair """
    with open(f"{PREDS}/{cond}/{pair.replace('/', '-')}.pickle", "rb") as pr:
        return pickle.load(pr)


def eval_signals_limits(
    state: State,
    sgn: str,
    params: dict,
    d: DataFrame,
    m: dict,
    preds: List = [],
    limits: List = [],
    weights: dict = {},
    multi: list = [],
    bounds=("gr", "lo"),
) -> (np.ndarray, np.ndarray, np.ndarray):
    # respect the length of the modified signals by using preds instead of d
    signal = []
    for n in range(state.se, state.se + state.k):
        sg = {b: np.full(len(preds[0]), True) for b in bounds}
        for b in bounds:
            sg[b] = e_bounds_funcs[b](
                preds[n], params[f"{sgn}.{b}{n}"], sg[b], id=m["pair"]
            )
            signal.append(sg[b])
    return reduce(andf, signal), signal


def eval_signals_mode(
    state: tuple,
    sgn: str,
    params: dict,
    d: DataFrame,
    m: dict,
    preds: List,
    limits: dict,
    weights: dict,
    multi: dict,
    bounds=("gr", "lo"),
) -> (np.ndarray, np.ndarray, np.ndarray):
    n_preds = len(preds)
    s_preds = len(preds[0])
    if params and list(params.keys())[0] in weights or len(weights) < 4:
        weights.update(params)

    sg = {}
    sgw = []
    for b in bounds:
        sg[b] = np.full(s_preds, 0.0)
        for n in range(n_preds):
            k = f"{sgn}.{b}{n}"
            sg[b] = f_bounds_funcs[b](
                preds[n], limits[k], weights[k], sg[b], id=m["pair"],
            )
        sgw.append(sg[b] > weights["trigger"])

    return reduce(andf, sgw), sg


def eval_signals_multi_mode(
    state: tuple,
    sgn: str,
    params: dict,
    d: DataFrame,
    m: dict,
    preds: List,
    limits: dict,
    weights: dict,
    multi: dict,
    bounds=("gr", "lo"),
    count=(1, 1),
    mw=0.27,
):
    signal = []
    for nw, lw in enumerate(weights):
        nl, weg = lw
        lim = limits[nl]
        wsig, _ = eval_signals_mode(
            state, sgn, params, d, m, preds, lim, weg, {}, bounds=bounds,
        )
        signal.append(wsig)

    sg = []
    for b in bounds:
        msig = 0
        for n, s in enumerate(signal):
            msig += s.astype(float) * params[f"{sgn}.{b}{n}"]
        sg.append(b_bounds_funcs[b](msig, mw))

    return reduce(andf, sg), sg


def eval_signals_pair_mode(
    state: tuple,
    sgn: str,
    params: dict,
    d: DataFrame,
    m: dict,
    preds: List,
    limits: dict,
    weights: dict,
    multi: dict,
    bounds=("gr", "lo"),
    count=(1, 1),
    mw=0.26,
):

    sgp = []
    for n, mu_params in enumerate(multi):
        msig, _ = eval_signals_multi_mode(
            state,
            sgn,
            mu_params,
            d,
            m,
            preds,
            limits,
            weights,
            multi,
            bounds=bounds,
            count=count,
            mw=mw,
        )
        sgp.append(msig)

    sg = []
    for b in bounds:
        psig = 0
        for n, s in enumerate(sgp):
            psig += s.astype(float) * params[f"{sgn}.p{n}"]
        sg.append(b_bounds_funcs[b](psig, mw))
    return reduce(andf, sg), sg


def min_max(d: DataFrame, ind):
    updated = False
    if ind not in mi or mi[ind] > d[ind].min():
        mi[ind] = d[ind].min()
        updated = True
    if ind not in ma or ma[ind] < d[ind].max():
        ma[ind] = d[ind].max()
        updated = True
    if updated:
        with open(MEAN_DIR / "min_max") as mf:
            json.dump({f"{ind}_min": mi[ind], f"{ind}_max": ma[ind]})


def average(arr, label):
    labels[label]["me"].append(np.nanmean(arr))
    labels[label]["qt"].append(np.nanquantile(arr, 0.05))
    print(json.dumps(labels))
    # with open(f"{MEAN_DIR}/{label}_me_{pid}", "w") as m:
    # json.dump(labels, m)


def plusf(x, y):
    return x + y


def andf(x, y):
    return x & y


def orf(x, y):
    return x | y


# NOTE: id is used to diff cache


def sge(x, y, z, id=None):
    return (x >= y) & (z)


def sle(x, y, z, id=None):
    return (x <= y) & (z)


def sgo(x, y, z, id=None):
    return (x >= y) | (z)


def slo(x, y, z, id=None):
    return (x <= y) | (z)


def sgf(x, y, z, w=0, id=None):
    return (x >= y).astype(float) * z + w


def slf(x, y, z, w=0, id=None):
    return (x <= y).astype(float) * z + w


f_bounds_funcs = {"gr": sgf, "lo": slf}
e_bounds_funcs = {"gr": sge, "lo": sle}
b_bounds_funcs = {"gr": lambda x, y: x > y, "lo": lambda x, y: x < y}
