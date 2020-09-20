import json
from functools import reduce
from typing import Any, List, Tuple

import numpy as np
from pandas import DataFrame

from user_data.modules import toolkit as tk
from user_data.modules.toolkit import q

PICKLE_PATH = "/tmp/.freqtrade/testcondition/"
RESULTS_PATH = "user_data/hyperopt_results"
PARAMS_PATH = "params.json"
EVALS_PATH = f"{RESULTS_PATH}/evals.json"

plusf = lambda x, y: x + y.astype(int)
andf = lambda x, y: x & y
orf = lambda x, y: x | y

# eval condition
# def econd(inds: List, func: Callable, prefixes):


def fn(fn: str, v: int) -> List[List[List[Any]]]:
    if v == 1:
        return [[["fn", [0]], ["fn", [0]], ["fn", [0]], ["fn", [0]], ["fn", [0]]]]
    elif v == 2:
        return [
            [
                ["fn", [0, 1]],
                ["fn", [0, 1]],
                ["fn", [0, 1]],
                ["fn", [0, 1]],
                ["fn", [1, 0]],
                ["fn", [1, 0]],
            ]
        ]


def trend(v: int) -> List[List[List[Any]]]:
    if v == 1:
        return [
            [
                ["dt", [0]],
                ["dt", [0]],
                ["dt", [0]],
                ["dt", [0]],
                ["dt", [0]],
                ["ut", [0]],
                ["ut", [0]],
            ],
            [
                ["ut", [0]],
                ["ut", [0]],
                ["ut", [0]],
                ["ut", [0]],
                ["ut", [0]],
                ["dt", [0]],
                ["dt", [0]],
            ],
        ]
    elif v == 2:
        return [
            [["dt", [0]], ["dt", [0]], ["dt", [0]], ["dt", [0]], ["ut", [0]], ["ut", [0]]],
            [["ut", [1]], ["ut", [1]], ["ut", [1]], ["ut", [1]], ["dt", [1]], ["dt", [1]]],
        ]


def cross(v: int) -> List[List[List[Any]]]:
    if v == 1:
        return [
            [
                ["rcr0", [0, 1]],
                ["rcr0", [0, 1]],
                ["rcr0", [0, 1]],
                ["rcr0", [1, 0]],
                ["tcr0", [0, 1]],
                ["tcr0", [0, 1]],
                ["tcr0", [1, 0]],
                ["tcr0", [1, 0]],
            ]
        ]
    if v == 2:
        return [
            [
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [1, 0]],
                ["tcr", [1, 0]],
            ]
        ]
    if v == 22:
        return [
            [
                ["tcr", [0, 1]],
                ["tcr", [0, 1]],
                ["tcr", [0, 1]],
                ["tcr", [1, 0]],
                ["tcr", [1, 2]],
                ["tcr", [1, 2]],
                ["tcr", [1, 2]],
                ["tcr", [2, 1]],
            ]
        ]
    if v == 3:
        return [  # tn/ck/kj
            [
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [0, 1]],
                ["cr", [1, 0]],
                ["cr", [1, 0]],
                ["cr", [1, 2]],
                ["cr", [1, 2]],
                ["cr", [2, 1]],
                ["tcr", [1, 2]],
                ["tcr", [1, 0]],
            ],
            [
                ["tcr", [1, 2]],
                ["tcr", [1, 2]],
                ["tcr", [1, 2]],
                ["tcr", [1, 2]],
                ["tcr", [2, 1]],
                ["tcr", [2, 1]],
                ["tcr", [0, 1]],
                ["tcr", [0, 1]],
                ["tcr", [1, 0]],
                ["cr", [0, 1]],
                ["cr", [2, 1]],
            ],
        ]
    if v == 4:  ## wclp-/avgp/medp/typp
        return [
            [["cr", [0, 1]], ["cr", [0, 1]], ["cr", [0, 1]], ["cr", [0, 1]], ["cr", [1, 0]]],
            [["cr", [0, 2]], ["cr", [0, 2]], ["cr", [0, 2]], ["cr", [0, 2]], ["cr", [2, 0]]],
            [["cr", [0, 3]], ["cr", [0, 3]], ["cr", [0, 3]], ["cr", [0, 3]], ["cr", [3, 0]]],
        ]


def curve(v: int) -> List[List[List[Any]]]:
    if v == 0:
        return [
            [["tpr1", [0]], ["tpr2", [0]], ["prb2", [0]], ["prb1", [0]]],
            [["prb1", [0]], ["prb2", [0]], ["tpr2", [0]], ["tpr1", [0]]],
        ]
    if v == 1:
        return [
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["prb1", [0]],
                ["prb1", [0]],
                ["prb1", [0]],
                ["prb1", [0]],
                ["prb2", [0]],
                ["prb2", [0]],
                ["tpr2", [0]],
                ["tpr2", [0]],
                ["tpr1", [0]],
            ],
        ]
    elif v == 2:
        return [  # aru/ard - macd/macds - #mdi/pdi
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["prb1", [1]],
                ["prb1", [1]],
                ["prb1", [1]],
                ["prb2", [1]],
                ["prb2", [1]],
                ["tpr2", [1]],
                ["tpr1", [1]],
            ],
        ]
    elif v == 22:
        return [  # rsi/stok
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr2", [1]],
                ["prb2", [1]],
                ["prb1", [1]],
            ],
        ]
    elif v == 3:
        return [  # rsi/stok/stod
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr2", [1]],
                ["prb2", [1]],
                ["prb1", [1]],
            ],
            [
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr2", [2]],
                ["prb2", [2]],
                ["prb1", [2]],
            ],
        ]
    elif v == -3:
        return [  # rsi/stok/stod
            [
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr2", [1]],
                ["tpr2", [1]],
                ["tpr2", [1]],
                ["prb2", [1]],
                ["prb2", [1]],
                ["prb1", [1]],
            ],
            [
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr2", [2]],
                ["tpr2", [1]],
                ["tpr2", [1]],
                ["prb2", [2]],
                ["prb2", [2]],
                ["prb1", [2]],
            ],
            [
                ["prb1", [0]],
                ["prb1", [0]],
                ["prb1", [0]],
                ["prb1", [0]],
                ["prb2", [0]],
                ["prb2", [0]],
                ["prb2", [0]],
                ["tpr2", [0]],
                ["tpr2", [0]],
                ["tpr1", [0]],
            ],
        ]
    elif v == 4:  # ub/mb/lb
        return [
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr2", [1]],
                ["prb2", [1]],
                ["prb1", [1]],
            ],
            [
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr2", [2]],
                ["prb2", [2]],
                ["prb1", [2]],
            ],
            [
                ["prb1", [2]],
                ["prb1", [2]],
                ["prb1", [2]],
                ["prb2", [2]],
                ["tpr2", [2]],
                ["tpr1", [2]],
            ],
        ]
    elif v == 44:
        return [
            [
                ["prb1", [3]],
                ["prb1", [3]],
                ["prb1", [3]],
                ["prb2", [3]],
                ["tpr2", [3]],
                ["tpr1", [3]],
            ],
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr1", [2]],
                ["tpr2", [2]],
                ["prb2", [2]],
                ["prb1", [2]],
            ],
            [
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr2", [1]],
                ["prb2", [1]],
                ["prb1", [1]],
            ],
        ]
    elif v == 444:  # wclp/typp/avgp/medp
        return [
            [
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr1", [0]],
                ["tpr2", [0]],
                ["tpr2", [0]],
                ["prb2", [0]],
                ["prb2", [0]],
                ["prb1", [0]],
            ],
            [
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr1", [1]],
                ["tpr2", [1]],
                ["tpr2", [1]],
                ["prb2", [1]],
                ["prb2", [1]],
                ["prb1", [1]],
            ],
            [
                ["prb1", [2]],
                ["prb1", [2]],
                ["prb1", [2]],
                ["prb1", [2]],
                ["prb2", [2]],
                ["prb2", [2]],
                ["tpr2", [2]],
                ["tpr2", [2]],
                ["tpr1", [2]],
            ],
            [
                ["prb1", [3]],
                ["prb1", [3]],
                ["prb1", [3]],
                ["prb1", [3]],
                ["prb2", [3]],
                ["prb2", [3]],
                ["tpr2", [3]],
                ["tpr2", [3]],
                ["tpr1", [3]],
            ],
        ]

    return []


def polar(v: int) -> List[List[List[Any]]]:
    if v == 0:
        return [
            [
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["dv", [0, 1]],
            ]
        ]
    if v == 1:
        return [
            [
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
            ],
            [
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
            ],
        ]
    if v == 2:  # hloc
        return [
            [
                ["cv", [0, 3]],
                ["cv", [0, 3]],
                ["cv", [0, 2]],
                ["cv", [0, 2]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [3, 2]],
                ["cv", [3, 2]],
            ],
            [
                ["dv", [2, 1]],
                ["dv", [2, 1]],
                ["dv", [2, 1]],
                ["dv", [3, 1]],
                ["dv", [3, 1]],
                ["dv", [3, 1]],
                ["dv", [2, 3]],
                ["dv", [2, 3]],
            ],
        ]
    elif v == 3:  # sa/sb
        return [
            [
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["cvf", [0, 1]],
                ["cvf", [0, 1]],
                ["dvf", [1, 0]],
            ],
            [
                ["dv", [1, 0]],
                ["dv", [1, 0]],
                ["dv", [1, 0]],
                ["cv", [1, 0]],
                ["cv", [1, 0]],
                ["dvf", [1, 0]],
                ["dvf", [1, 0]],
                ["cvf", [0, 1]],
            ],
        ]
    if v == 4:  # hloc
        return [
            [
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["cv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
                ["dv", [0, 1]],
            ],
            [
                ["dv", [2, 3]],
                ["dv", [2, 3]],
                ["dv", [2, 3]],
                ["dv", [2, 3]],
                ["dv", [2, 3]],
                ["cv", [2, 3]],
                ["cv", [2, 3]],
                ["cv", [2, 3]],
                ["cv", [2, 3]],
                ["cv", [2, 3]],
            ],
        ]
    return []


def bop(d: DataFrame, cond) -> np.array:
    if cond == "bop>bop_min":
        ret = q("bop>bop_min")
    elif cond == "bop>=0":
        ret = q("bop>=0")
    elif cond == "bop>bop_max":
        ret = q("bop>bop_max")
    elif cond == "bop<bop_min":
        ret = q("bop<bop_min")
    elif cond == "bop<=0":
        ret = q("bop<=0")
    elif cond == "bop<bop_max":
        ret = q("bop<bop_max")
    elif cond == "bop_min><=0":
        ret = q("bop>bop_min") & q("bop<=0")
    elif cond == "0>=<bop_max":
        ret = q("bop>=0") & q("bop<bop_max")
    elif cond == "bop_min><bop_max":
        ret = q("bop>bop_min") & q("bop<bop_max")
    elif cond == "bop_min<>bop_max":
        ret = q("bop<bop_min") | q("bop<bop_max")
    elif cond == "bop_min<>=0":
        ret = q("bop<bop_min") | q("bop>=0")
    elif cond == "0<=>bop_max":
        ret = q("bop<=0") | q("bop>bop_max")
    return ret


def filter_and_join(orfp: List) -> str:
    return "|".join(list(filter(None, orfp)))


def replay(
    d: DataFrame, prev_params={}, sgn="", conds={}, given={}, op="&"
) -> Tuple[np.ndarray, List[np.ndarray], int, int, List[str]]:
    """ col: ['ind1'', 'ind2', 'ind3']
    order: [[(tpr', col.idx), (prb, col.idx)], ... ]
    """
    # write iters

    tk.d = d

    if not prev_params:
        with open(EVALS_PATH, "r") as fp:
            prev_params = json.load(fp)
    p_evals = prev_params["probability"] if "probability" in prev_params else 0.99
    if sgn:
        prev_params = prev_params[sgn]
    if not conds:
        conds = tconds

    ## initialize o to 0 because eval does not loop over the same params
    o = 0
    # the complete list of ands
    sent = []
    ln_d = len(d)
    ln_pp = len(prev_params)
    ln_cs = len(conds)
    ln_lpp = len(prev_params[-1]) if ln_pp > 0 else len(conds[0][1])
    ln_lcs = len(conds[-1][1])
    wrap = ln_pp == ln_cs and ln_lpp == ln_lcs
    # count the total number of flattened conditions
    n_evals = 0
    # signal if we have constructed all params evald so far
    stop = False
    stopnew = False
    # first run?
    if ln_pp == 0 and not wrap:
        n, o = 0, 0
        return np.full(ln_d, True), [np.full(ln_d, False) for _ in range(ln_lpp)], n, o, ln_lpp
    elif given:
        ln_lp = len(prev_params[given["n"]][0]) if ln_pp > given["n"] else 0  # length last param
    else:
        ln_lp = len(prev_params[-1][0]) if len(prev_params[-1]) > 0 else 0  # length last param
    ln_pr = len(conds[ln_pp - 1][1][0])  # length last predicate
    if ln_lp < ln_pr:
        # still needs to eval ORs if on the same conditions, increment o
        stop = True if not wrap else False
        n = ln_pp - 1
        o = ln_lp  ## the length is eq to the next index to eval
    # start eval a new condition, increment n, reset ands
    # print(f'tot cond is {tot_cond} and ln_pp: {ln_pp}, ln_cs: {ln_cs}')
    elif ln_lp == ln_pr and ln_pp < ln_cs:
        stopnew = True
        sno = 0
        snn = ln_pp

    econd = ""
    if given:
        n = given["n"]
        o = given["o"]
        if o < len(conds[n][1][0]):  # got to finish
            u_ands = prev_params[n] if ln_pp > n else [[] for i in range(len(conds[n][1]))]
            for andp in prev_params[:n]:
                for orfp in andp:
                    jor = filter_and_join(orfp)
                    if jor:
                        econd += f"({jor}){op}"
                        n_evals += 1
            for andp in prev_params[n + 1 :]:
                for orfp in andp:
                    jor = filter_and_join(orfp)
                    if jor:
                        econd += f"({jor}){op}"
                        n_evals += 1
            econd = econd.rstrip(op).replace(")", ").astype(int)", 1)
            if len(u_ands[0]) > 0:
                for k, orfp in enumerate(u_ands):
                    jor = filter_and_join(orfp)
                    u_ands[k] = eval(jor) if jor else False
            else:
                u_ands = [np.full(ln_d, False) for i in range(len(u_ands))]
            # if positions were given prev_params can't be empty
            sent = eval(econd) if econd != "" else np.full(ln_d, 0)
    elif stop:
        if ln_pp > 0:
            u_ands = prev_params.pop()  # unfinished ands
            # eval all prev_params except the last unfinished ands
            for andp in prev_params:
                for orfp in andp:
                    jor = filter_and_join(orfp)
                    if jor:
                        econd += f"({jor}){op}"
                        n_evals += 1
            econd = econd.rstrip(op).replace(")", ").astype(int)", 1)
            # eval each or of the unfishied ands separetely to be chained with addcond
            for k, orfp in enumerate(u_ands):
                jor = filter_and_join(orfp)
                u_ands[k] = eval(jor) if jor else False
                sent = eval(econd) if len(prev_params) > 0 else np.full(ln_d, n_evals)
        else:
            sent = np.full(ln_d, n_evals)  # def value for ANDs is true...
        # ands need to be finished (ored called by addcond)
    else:
        wrapped = False
        # loop over all prev params since we don't know which one is being evald
        for n, andp in enumerate(prev_params):
            # get the length of any of the ors
            ln_or = len(andp[0])
            # if we don't know where we wrapped yet and the length of any or
            # does not match the referenced condition, then it is the evaling position
            if not wrapped and wrap and ln_or < len(conds[n][1][0]):
                wrapped = True
                wn, wo = n, ln_or
                u_ands = andp
                # eval the previously run ORs of the current AND
                if ln_or > 0:
                    for k, orfp in enumerate(u_ands):
                        u_ands[k] = filter_and_join(orfp)
                else:
                    u_ands = [np.full(ln_d, False) for i in range(len(u_ands))]
            else:
                for orfp in andp:
                    if len(orfp) > 0:
                        jor = filter_and_join(orfp)
                        if jor:
                            econd += f"({jor}){op}"
                            n_evals += 1
        # cast the first and as type
        econd = econd.rstrip(op).replace(")", ").astype(int)", 1)
        sent = eval(econd) if econd != "" else np.full(ln_d, n_evals)
        # make sure the new ends is initialized to the correct number of
        # predicates of the next condition (conds[n])
        # ands are collections of ors so default to False
        if stopnew:
            ln_ands = len(conds[n][1])
            n, o = snn, sno
        # if we wrapped we have to set the position that we stored before
        elif wrapped:
            n, o = wn, wo
        # else there are no condition to eval anymore
        else:
            ln_ands, n, o = len(conds[0][1]), -1, 0
        # initialize u_ands for addcond
        if not wrapped:
            u_ands = [np.full(ln_d, False) for i in range(ln_ands)]
    return sent, u_ands, n, o, n_evals, p_evals


def sequencer(
    d: DataFrame, sgn: List, prev_params={}, conds={}
) -> Tuple[np.ndarray, List[np.ndarray], int, int, List[str]]:
    """ col: ['ind1'', 'ind2', 'ind3']
    order: [[(tpr', col.idx), (prb, col.idx)], ... ]
    """
    # write iters
    if not prev_params:
        with open(PARAMS_PATH, "r") as fp:
            prev_params = json.load(fp)

    if not conds:
        conds = tconds

    ## initialize o to -1 to compensate for the possible +1 for incomplete cond
    o = -1
    # the complete list of ands
    sent = []
    ln_pp = len(prev_params)
    ln_lpp = len(prev_params[-1]) if ln_pp > 0 else 0
    ln_cs = len(conds)
    ln_lcs = len(conds[-1][1][0])
    ld = len(d)
    # list to store evaluated conditions
    evals = {s: [[] for _ in prev_params] for s in sgn}
    # signal if we have constructed all params evald so far
    stop = False
    stopnew = False
    # first run?
    if ln_pp == 0:
        n, o = 0, 0
        inds, preds, pfxs = conds[n][0], conds[n][1], conds[n][2]
        ands = [[] for i in range(len(preds))]
    # are we looping again?
    wrap = True if ln_pp == ln_cs and ln_lpp == ln_lcs else False
    for n, cond in enumerate(prev_params):
        # each predicate is a list made of a callable (0) and
        # an ordered list of inds indexes to pass the corresponding inds as args (1)
        inds, preds, pfxs = conds[n][0], conds[n][1], conds[n][2]
        # part of the params naming
        subj = ".".join(inds)
        tot_pc = len(preds[0])  # ORs
        tot_and = len(preds)  # ANDs
        tot_cond = len(cond)
        # the ands of the current predicate
        ands = [[] for i in range(len(preds))]
        for s in sgn:
            evals[s][n] = [[] for i in range(len(preds))]
            # loop over evald conds appending one OR to each AND
            for o in range(0, tot_cond):
                for pn, andp in enumerate(preds):
                    # if any of the current ORs is empty, fill evals
                    # with empty strings and skip
                    if not cond[o] or list(cond[o])[0] == None:
                        evals[s][n] = [["" for n_or in range(tot_pc)] for n_ands in range(tot_and)]
                        break
                    args = []
                    # parts of the parameter name
                    pred = andp[o][0]
                    e = "tk." + pred + "("
                    # add the indicators as args
                    for ind in andp[o][1]:
                        args.append(inds[ind])
                        e += f"'{inds[ind]}',"
                    # add the parameters as args, prefix+ANDindex
                    for pfx in pfxs:
                        val = cond[o][f"{s}_{subj}:{pfx}:{pred}:{pn+1:02d}"]
                        if val is not None:
                            args.append(val)
                            e += f"{val},"
                        else:
                            break
                    e = e.rstrip(",")
                    # if any of the parameters value is empty, skip the predicate
                    # append null to remaining ands entries in evals
                    if val is None:
                        for rn in range(pn, tot_and):
                            evals[s][n][rn].append("")
                        break
                    # call the predicate
                    if ld != 0:
                        ands[pn].append(eval(f"tk.{pred}(*args)"))
                    else:
                        e = e + ")"
                        evals[s][n][pn].append(e)
        # if tot_cond are less than the referenced condition tot_pc
        # we have exhausted the evald results
        if tot_cond < tot_pc and ln_pp < n:
            # still needs to eval ORs if on the same conditions, increment o
            stop = True
            o += 1  # the counter for the next OR statement
            if wrap:  # if we are wrapping store the current position to return
                nn, no = n, o
            else:
                break  # it is the last iteration anyway
        else:
            if ld != 0:
                for andp in ands:
                    sent.append(reduce(orf, andp))
            # start eval a new condition, increment n, reset ands
            # print(f'tot cond is {tot_cond} and lpp: {lpp}, lcs: {lcs}')
            if tot_cond == tot_pc and not wrap:
                stopnew = True
                n += 1
                o = 0
    if len(sent) > 0:
        if op == "&":
            sent = reduce(andf, sent)  # reduce sent because we are done.
        elif op == "+":
            sent = reduce(plusf, sent)
    else:
        sent = np.full(len(d), True)  # def value for ANDs is true...
    if stop and o > 1:  # if o == 1 a new cond just started, no andp to reduce yet
        # ands need to be finished (ored called by addcond)
        if ld != 0:
            for pn, andp in enumerate(ands):
                ands[pn] = reduce(orf, andp)
    # make sure the new ends is initialized to the correct number of
    # predicates of the next condition (conds[n])
    # ands are collections of ors so default to False
    elif stopnew:
        ands = [np.full(len(d), False) for i in range(len(conds[n][1]))]

    return sent, ands, n, o, evals


## add a condition from the new testing ones from tconds list
def addcond(
    d: DataFrame, sgn: str, sent: list, ands: list, n: int, o: int, params: dict, p: dict, op="&"
):
    tk._p = p
    tk.d = d
    """add a new condition to the sequence"""
    inds, preds, pfxs = tconds[n][0], tconds[n][1], tconds[n][2]
    # when o is reset to 0 ands is  0 len, has to be reinitialized
    # but this should be handled by sequencer/replay functions
    # if o == 0:
    # ands = [np.full(len(d), False) for i in range(len(preds))]
    # add the next or to each andp from the conds config
    for pn, andp in enumerate(ands):
        args = []
        pred = preds[pn][o][0]  # fn to call
        subj = ".".join(inds)
        # print(f'n is {n}, pn is {pn}')
        # print(f'inds are {preds[pn][o][1]}')
        for ind in preds[pn][o][1]:
            args.append(inds[ind])
        for pfx in pfxs:
            args.append(params[f"{sgn}_{subj}:{pfx}:{pred}:{pn+1:02d}"])  # new params
        orp = eval(f"tk.{pred}(*args)")
        ands[pn] = andp | orp

    if op == "&":
        ands = reduce(andf, ands)
        ret = sent & ands
    elif op == "+":
        ands = reduce(plusf, ands)
        ret = sent + ands
    return ret


## list of tuples of sequencer args (inds', seq, prefx)
tconds = [
    [["ad"], curve(1), ["smp"]],
    [["vol"], curve(1), ["smp"]],
    [["tl"], curve(1), ["smp"]],
    [["trg"], curve(1), ["smp"]],
    [["cci"], curve(1), ["smp"]],
    [["cmo"], curve(1), ["smp"]],
    [["ub"], curve(1), ["smp"]],
    [["mb"], curve(1), ["smp"]],
    [["lb"], curve(1), ["smp"]],
    [["macdh"], curve(1), ["smp"]],
    [["macd"], curve(1), ["smp"]],
    [["macds"], curve(1), ["smp"]],
    [["adx"], curve(1), ["smp"]],
    [["pdi"], curve(1), ["smp"]],
    [["mdi"], curve(1), ["smp"]],
    [["mp"], curve(1), ["smp"]],
    [["mpp"], curve(1), ["smp"]],
    [["sar"], curve(1), ["smp"]],
    [["mom"], curve(1), ["smp"]],
    [["bop"], curve(1), ["smp"]],
    [["ppo"], curve(1), ["smp"]],
    [["htpe"], curve(1), ["smp"]],
    [["htph"], curve(1), ["smp"]],
    [["htin"], curve(1), ["smp"]],
    [["htqu"], curve(1), ["smp"]],
    [["htsi"], curve(1), ["smp"]],
    [["htls"], curve(1), ["smp"]],
    [["avgp"], curve(1), ["smp"]],
    [["medp"], curve(1), ["smp"]],
    [["typp"], curve(1), ["smp"]],
    [["wclp"], curve(1), ["smp"]],
    [["rsi"], curve(1), ["smp"]],
    [["stok"], curve(1), ["smp"]],
    [["stod"], curve(1), ["smp"]],
    [["ard"], curve(1), ["smp"]],
    [["aru"], curve(1), ["smp"]],
    [["h"], curve(1), ["smp"]],
    [["o"], curve(1), ["smp"]],
    [["l"], curve(1), ["smp"]],
    [["c"], curve(1), ["smp"]],
    [["sa"], curve(1), ["smp"]],
    [["sb"], curve(1), ["smp"]],
    [["tn"], curve(1), ["smp"]],
    [["kj"], curve(1), ["smp"]],
    [["sh"], curve(1), ["smp"]],
    [["lo"], curve(1), ["smp"]],
    [["emas"], curve(1), ["smp"]],
    [["emam"], curve(1), ["smp"]],
]

# tconds = [
#     [['rsi', 'stok', 'stod'], curve(-3), ['smp']],
#     [['ub', 'lb'], polar(1), ['smp']],
#     [['macd', 'macds'], polar(1), ['smp']],
#     [['macds', 'macd'], curve(2), ['smp']],
#     [['mp', 'mpp'], polar(0), ['smp']],
#     [['tl'], trend(1), ['smp']],
#     [['ub', 'mb', 'lb'], curve(4), ['smp']],
#     [['mdi', 'pdi'], curve(2), ['smp']],
#     [['ard', 'aru'], curve(2), ['smp']],
#     [['bop'], trend(1), ['smp', 'shift']],
#     [['sar'], trend(1), ['smp', 'shift']],
#     [['htph', 'htpe'], polar(1), ['smp']],
#     [['htin', 'htqu'], polar(1), ['smp']],
#     [['tn', 'c', 'kj'], cross(3), ['smp']],
#     [['sa', 'sb'], polar(3), ['smp']],
#     [['h', 'l', 'o', 'c'], polar(2), ['smp']],
# ]
