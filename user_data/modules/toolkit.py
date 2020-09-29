# toolkit
import re
import numpy as np
from pandas import DataFrame
from typing import List

d: DataFrame = None
qc: dict = {}
_p = {}

_v = ".values"
_stx = "([^a-zA-Z.S0-9]?[0-9]?)([a-z]*)([\.a-z]{0,2}@?[0-9]{,4})([^a-zA-Z.S0-9]{0,2})([0-9.]*)([_a-z]*)([\.a-z]{0,2}@?[0-9]{,4})([^a-zA-Z]*)"
_fn = {"s": "shift", "r": "rolling", "x": "max", "n": "min", "m": "sum", "b": "sub"}


def q(query, level=0, scope=locals()) -> np.ndarray:
    """
        Examples:
        q('adx>pdi') :: dataframe['adx'] > dataframe['pdi]
        q('adx1>pdi2') :: dataframe['adx].shift(1) > dataframe['pdi'].shift(2)
        q('|adx-pdi|') :: abs(dataframe['adx'] - dataframe['pdi'])
        q('2adx') :: 2 * dataframe['adx']
        q('mom>0') :: dataframe['mom'] > 0
        q('adx>pdi/2') :: dataframe['adx'] > dataframe['pdi] / 2
        q('adx.s3) :: dataframe['adx].shift(3)
        q('adx.s@3) :: dataframe['adx].shift(-3)

        Strict/Loose comparison operators:
        >+ :: strictly greater
        >- :: loosely greater
        <+ :: loosely lesser
        <- :: strictly lesser
        q('adx>+pdi') :: dataframe['adx'] / dataframe['pdi'] > _max
        where _max is a value >1 (1.005)
        q('pdi<-mdi') :: dataframe['pdi'] / dataframe['mdi] < _min
        where _min is a value <1 (0.995)

        Syntax notes:
        Absolute: for the result of the operations, not prior.
        Division: only for the second member if not the operator
        Multiplication: both first and second member
        Numbers: only as second members
        """
    # _max = d._max
    # _min = d._min
    # OR
    # d.COL_min, d.COL_max
    # where COL is a dataframe column
    # d = self.d
    # s, r, x , n... shortcuts for series functions defined in _fn

    if query in qc:
        # tname = type(qc[query]).__name__
        # if tname != 'ndarray':
        #     print(tname, query)
        return qc[query]

    parts = list(re.search(_stx, query).groups())
    # print(parts)
    if len(parts) < 3:
        print("wrong syntax: " + query)
        return

    if (
        (parts[3] == None)
        or (parts[1] == None)
        or (parts[5] == None and parts[4] == None)
    ):
        print("noop for: " + query)
        return

    m1 = parts[1]
    if parts[3] != "|":
        op = parts[3]
    else:
        op = ""
        end = ")"
    if parts[5] == "":
        ## comparisons against numbers, the sign is of the member
        if len(op) == 2 and op[1] in ["-", "+"]:
            m2 = op[1] + parts[4]
            op = op[0]
        else:
            m2 = parts[4]
    else:
        m2 = parts[5]

    ## compute recursion
    r = 0
    subcmd = m1 + parts[2] + op + m2 + parts[6]
    if level == 0 and op in ["-", "/", "*"]:
        if subcmd != query:
            q(subcmd, ++level)
            m1 = "q('" + subcmd + "')"
            # print(m1)
            op = ""
            m2 = ""
            parts[2] = ""
            parts[6] = ""
            r = 1

    ## (dataframe)
    # print(m1)
    # print(m2)
    o_m1, o_m2 = m1, m2
    d1, d2 = False, False
    lm1, lm2 = len(m1), len(m2)
    lp1 = len(parts[0])
    if (lp1 == 0 and op != "") or (op == "" and r == 0):
        if m1 not in scope and (lm1 < 2 or m1[1] != "."):
            m1 = "d['" + m1 + "']"
            d1 = True

        if m2 not in scope and m2 not in globals():
            if lm2 > 0 and m2[0] != "-" and not re.search("[0-9]", m2[0]):
                if lm2 < 2 or m2[1] != ".":
                    if m2 in d:
                        m2 = "d['" + m2 + "']"
                        d2 = True
                    else:
                        m2 = "d." + m2
        else:
            eval(f"global {m2}")

    def add_v1(s):
        if d1 is True:
            s += _v
        return s

    def add_v2(s):
        if d2 is True:
            s += _v
        return s

    ## compute query
    if parts[2] != None and parts[2] != "":
        am1 = add_v1(o_m1 + parts[2])
        afx1 = "qc['" + am1 + "']"
        if am1 not in qc:
            pos = parts[2].replace("@", "-")
            print(_fn, pos)
            qc[am1] = eval(m1 + "." + _fn[pos[1:2]] + "(" + pos[2:] + ")" + _v, scope)
    else:
        afx1 = ""
        am1 = add_v1(o_m1)

    def getc(c1, c2):
        if r != 0:
            return c1 + c2
        else:
            return c1 + "qc['" + c2 + "']"

    ##
    if parts[0] != None and parts[0] != "":
        if parts[0][0] == "|":
            if lp1 > 1:
                sm1 = parts[0][1:] + am1
                sfx1 = getc("abs(", sm1)
                if sm1 not in qc:
                    if afx1 != "":
                        qc[sm1] = eval(parts[0][1:] + "*" + "qc['" + am1 + "']", scope)
                    else:
                        qc[sm1] = eval(
                            parts[0][1:] + "*" + "d['" + o_m1 + "']" + _v, scope
                        )
            else:
                sm1 = parts[0] + am1
                sfx1 = getc("abs(", am1)
        else:
            sm1 = parts[0] + am1
            sfx1 = getc("", sm1)
            if sm1 not in qc:
                if r != 0:
                    qc[sm1] = eval(parts[0] + "*" + "qc['" + subcmd + "']", scope)
                else:
                    if afx1 != "":
                        qc[sm1] = eval(parts[0] + "*" + "qc['" + am1 + "']", scope)
                    else:
                        qc[sm1] = eval(parts[0] + "*" + "d['" + o_m1 + "']" + _v, scope)
    else:
        sfx1 = ""
        sm1 = ""

    ##
    if parts[6] != None and parts[6] != "":
        am2 = add_v2(o_m2 + parts[6])
        afx2 = "qc['" + am2 + "']"
        if am2 not in qc:
            pos = parts[6].replace("@", "-")
            qc[am2] = eval(m2 + "." + _fn[pos[1:2]] + "(" + pos[2:] + ")" + _v, scope)
    else:
        afx2 = ""
        am2 = add_v2(o_m2)

    ##
    if parts[4] != None and parts[4] != "" and parts[5] != "":
        sm2 = parts[4] + am2
        sfx2 = getc("", sm2)
        if sm2 not in qc:
            qc[sm2] = eval(parts[4] + "*" + "qc['" + am2 + "']", scope)
    else:
        sfx2 = ""
        sm2 = am2

    if parts[7] != None and parts[7] != "" or parts[3] == "|":
        if parts[7] == "|" or parts[3] == "|":
            efx2 = sfx2 + ")"
        else:
            em2 = sm2 + parts[7]
            efx2 = getc("", em2)
            if em2 not in qc:
                if r != 0:
                    qc[em2] = eval("qc['" + subcmd + "']" + parts[7], scope)
                else:
                    qc[em2] = eval("qc['" + sm2 + "']" + parts[7], scope)
    else:
        if "efx2" not in locals():
            efx2 = ""

    def cm1(m1):
        if sfx1 != "":
            m1 = sfx1
        else:
            if afx1 != "":
                m1 = afx1
            else:
                m1 = add_v1(m1)
        return m1

    def cm2(m2):
        if efx2 != "":
            m2 = efx2
        else:
            if sfx2 != "":
                m2 = sfx2
            else:
                if afx2 != "":
                    m2 = afx2
                else:
                    m2 = add_v2(m2)
        return m2

    ## compute op
    if len(op) == 2:
        if op[0] == "<" or op[0] == ">":
            if op[1] in ["-", "+"]:
                m1 = "q('" + o_m1 + parts[2] + "/" + o_m2 + parts[6] + "')"
                afx1 = ""
                afx2 = ""
                if mm_type[o_m2] is True:
                    pfx = o_m2
                else:
                    pfx = ""
                if op[1] == "-":
                    m2 = "d." + pfx + "_min" + _v
                else:  # +
                    m2 = "d." + pfx + "_max" + _v
                op = op[0]
            else:
                m1 = cm1(m1)
                m2 = cm2(m2)
        else:
            m1 = cm1(m1)
            m2 = cm2(m2)
    else:
        m1 = cm1(m1)
        m2 = cm2(m2)

    cmd = m1 + op + m2
    # print(cmd)
    # time.sleep(1)
    if query in qc:
        # tname = type(qc[query]).__name__
        # if tname != 'ndarray':
        #     print(tname, query)
        return qc[query]

    qc[query] = eval(cmd, scope)
    # tname = type(qc[query]).__name__
    # if tname != 'ndarray':
    #     print(tname, query)
    return qc[query]


## cache member ## needs ._v, d.scope
def cc(col, r, op, s=0, scope=locals()):
    rs = str(r)
    ss = str(s)
    cc_id = col + rs + op + ss
    if cc_id in qc:
        return qc[cc_id]
    else:
        cc_id_s = col + ss + ".sr"
        if cc_id_s not in qc:  ## cached shifted series
            qc[cc_id_s] = d[col].shift(s)
        cc_id_r = cc_id_s + rs
        if cc_id_r not in qc:  ## cached rolling
            if r == 0:
                qc[cc_id_r] = qc[cc_id_s]
            else:
                qc[cc_id_r] = qc[cc_id_s].rolling(r)
        if hasattr(qc[cc_id_r], "values") and hasattr(qc[cc_id_r], op):
            qc[cc_id] = eval("qc['" + cc_id_r + "']" + _v + "." + op + "()", scope)
        else:
            qc[cc_id] = eval("qc['" + cc_id_r + "']." + op + "()" + _v, scope)
        return qc[cc_id]


## rolling down w=3
def rd(col, smp=3):
    query = "rd_" + col + str(smp)
    if query not in qc:
        qc[query] = cc(col, smp, "min") < cc(col, smp, "max")
    return qc[query]


## wall ahead w=6 # ONLY use with sa/sb
def wa(col, smp=6, verse=">", bounds=False):
    query = "wa_" + col + str(smp) + str(verse)
    if query not in qc:
        if bounds:
            if verse == ">":
                qc[query] = (
                    cc(col, smp, "min") / cc(col, smp, "min", -smp) > d["_max"].values
                )
            elif verse == "<":
                qc[query] = (
                    cc(col, smp, "max") / cc(col, smp, "max", -smp) < d["_min"].values
                )
        else:
            if verse == ">":
                qc[query] = cc(col, smp, "min") > cc(col, smp, "min", -smp)
            elif verse == "<":
                qc[query] = cc(col, smp, "max") < cc(col, smp, "max", -smp)
    return qc[query]


## dip forward w=-8 ONLY sa/sb
def dpf(col, smp=-8):
    query = "dip_" + col + str(smp)
    if query not in qc:
        qc[query] = q(col) > q(col + ".n@" + str(-smp))
    return qc[query]


## dip
def dip(col, smp=8):
    query = "dp_" + col + str(smp)
    if query not in qc:
        qc[query] = d[col].values <= cc(col, smp, "min")
    return qc[query]


## spike w=3
def spk(col, smp=3, ratio=1.1):
    query = "spk_" + col + str(smp) + str(ratio)
    if query not in qc:
        qc[query] = d[col].values / cc(col, smp, "mean") > ratio
    return qc[query]


## dump w=3
def dmp(col, smp=3, ratio=0.9):
    query = "dmp_" + col + str(smp) + str(ratio)
    if query not in qc:
        qc[query] = d[col].values / cc(col, smp, "mean") < ratio
    return qc[query]


## downtrend w=12
def dt(col, smp=12, shift=1):
    query = "dt_" + col + str(smp) + str(shift)
    if query not in qc:
        qc[query] = cc(col, smp, "sum") < cc(col, smp, "sum", shift)
    return qc[query]


## downtrend forward w=12
def dtf(col, smp=12, shift=-1):
    query = "dtf_" + col + str(smp) + str(shift)
    if query not in qc:
        qc[query] = cc(col, smp, "sum") > cc(col, smp, "sum", shift)
    return qc[query]


## uptrend w=12
def ut(col, smp=12, shift=1):
    query = "ut_" + col + str(smp) + str(shift)
    if query not in qc:
        qc[query] = cc(col, smp, "sum") > cc(col, smp, "sum", shift)
    return qc[query]


## uptrend forward w=12
def utf(col, smp=12, shift=-1):
    query = "utf_" + col + str(smp) + str(shift)
    if query not in qc:
        qc[query] = cc(col, smp, "sum") < cc(col, smp, "sum", shift)
    return qc[query]


## horizon sa/sb
def hrz(col, smp=12):
    query = "hrz_" + col + str(smp)
    if query not in qc:
        qc[query] = q(col) == cc(col, smp, "max", -smp)
    return qc[query]


## previously lower w=6 ._min
def pl(col1, col2, smp=6, ratio=0.9):
    query = "md_" + col1 + col2 + str(smp) + str(ratio)
    if query not in qc:
        qc[query] = cc(col1, smp, "sum") / cc(col2, smp, "sum") < ratio
    return qc[query]


## quiet
def qt(col, smp=6, noise=None):
    query = "qt_" + str(col) + str(smp) + str(noise)
    if query not in qc:
        if noise == None:
            noise = d["_max"].values - d["_min"].values
        tmp = cc(col, smp, "mean") / q(col)
        qc[query] = (tmp > 1 - noise) & (tmp < 1 + noise)
    return qc[query]


## converging w=3
def cv(col1, col2, smp=3):
    query = "cv_" + col1 + col2 + str(smp)
    if query not in qc:
        qc[query] = cc(col1, smp, "sum") - cc(col2, smp, "sum") < cc(
            col1, smp, "sum", smp
        ) - cc(col2, smp, "sum", smp)
    return qc[query]


## diverging w=3
def dv(col1, col2, smp=3):
    query = "dv_" + col1 + col2 + str(smp)
    if query not in qc:
        qc[query] = abs(cc(col1, smp, "sum") - cc(col2, smp, "sum")) > abs(
            cc(col1, smp, "sum", smp) - cc(col2, smp, "sum", smp)
        )
    return qc[query]


## converging forward w=6 # ONLY sa/sb, col1 should be the higher one
def cvf(col1, col2, smp=6):
    query = "cvf_" + col1 + col2 + str(smp)
    if query not in qc:
        qc[query] = cc(col1, smp, "sum") - cc(col2, smp, "sum") > cc(
            col1, smp, "sum", -smp
        ) - cc(col2, smp, "sum", -smp)
    return qc[query]


## diverging forward w=6 # ONLY sa/sb, col1 should be the higher one
def dvf(col1, col2, smp=6):
    query = "dvf_" + col1 + col2 + str(smp)
    if query not in qc:
        qc[query] = cc(col1, smp, "sum") - cc(col2, smp, "sum") < cc(
            col1, smp, "sum", -smp
        ) - cc(col2, smp, "sum", -smp)
    return qc[query]


## above col1 // col2 (col1 over col2)
def abv(col1, col2, sp=3, ratio=0):
    query = "abv_" + col1 + col2 + str(ratio) + str(smp)
    if query not in qc:
        if ratio == 0:
            qc[query] = cc(col1, smp, "sum") > cc(col2, smp, "sum")
        else:
            qc[query] = cc(col1, smp, "sum") / cc(col2, smp, "sum") > ratio
    return qc[query]


## crossed col1 // col2
def cr(col1, col2, smp=3, ratio=0):
    query = "cr_" + col1 + col2 + str(ratio) + str(smp)
    if query not in qc:
        if ratio == 0:
            qc[query] = (cc(col1, smp, "sum") > cc(col2, smp, "sum")) & (
                cc(col1, smp, "sum", smp) < cc(col2, smp, "sum", smp)
            )
        else:
            qc[query] = (
                cc(col1, smp, "sum", smp) / cc(col2, smp, "sum", smp) > ratio
            ) & (cc(col1, smp, "sum", smp) / cc(col2, smp, "sum", smp) > ratio)
    return qc[query]


## to cross soon col1 \\ col2 (col1 going below col2)
def tcr(col1, col2, smp=3, ratio=0):
    query = "tcr_" + col1 + col2 + str(ratio) + str(smp)
    change1 = cc(col1, smp, "sum", smp) - cc(col1, smp, "sum")
    change2 = cc(col2, smp, "sum", smp) - cc(col2, smp, "sum")
    if query not in qc:
        if ratio == 0:
            qc[query] = cc(col1, smp, "sum") - change1 < cc(col2, smp, "sum") - change2
        else:
            qc[query] = (cc(col1, smp, "sum") - change1) / (
                cc(col2, smp, "sum") - change2
            ) < ratio
    return qc[query]


def tpr(col, hops=2, span=1):
    """ tapering w=3 """
    span = _p[span]
    span_str = str(span)
    query = "tpr_" + col + str(hops) + span_str
    c1 = col + "-" + col + ".s" + span_str
    c2 = col + ".s" + span_str + "-" + col + ".s" + str(2 * span)
    c3 = col + ".s" + str(2 * span) + "-" + col + ".s" + str(3 * span)
    if query not in qc:
        if hops == 2:
            qc[query] = (q(c1) < q(c2)) & (q(c2) < q(c3))
        else:
            if hops == 1:
                qc[query] = q(c1) < q(c2)
    return qc[query]


def tpr1(col, span=1):
    """ tpr with span=1 """
    return tpr(col, 1, span)


def tpr2(col, span=1):
    return tpr(col, 2, span)


## parabolic
def prb(col, hops=2, span=1):
    span_str = str(span)
    query = "prb_" + col + str(hops) + span_str
    c1 = col + "-" + col + ".s" + span_str
    c2 = col + ".s" + span_str + "-" + col + ".s" + str(2 * span)
    c3 = col + ".s" + str(2 * span) + "-" + col + ".s" + str(3 * span)
    if query not in qc:
        if hops == 2:
            qc[query] = (q(c1) > q(c2)) & (q(c2) > q(c3))
        else:
            if hops == 1:
                qc[query] = q(c1) > q(c2)
    return qc[query]


def prb2(col, span=1):
    return prb(col, 2, span)


def prb1(col, span=1):
    return prb(col, 1, span)


## parabolic forward sa/sb
def prbf(col, hops=2, span=1):
    span_str = str(span)
    query = "prb_" + col + str(hops) + span_str
    c1 = col + "-" + col + ".s@" + span_str
    c2 = col + ".s@" + span_str + "-" + col + ".s@" + str(2 * span)
    c3 = col + ".s@" + str(2 * span) + "-" + col + ".s@" + str(3 * span)
    if query not in qc:
        if hops == 2:
            qc[query] = (q(c1) < q(c2)) & (q(c2) < q(c3))
        else:
            if hops == 1:
                qc[query] = q(c1) < q(c2)
    return qc[query]


## switched to down side, args order from higher to lower
def swd(col1, col2, col3):
    query = "swi_" + col1 + col2 + col3
    if query not in qc:
        qc[query] = (q(col1 + "-" + col2) > q(col2 + "-" + col3)) & (
            q(col1 + ".s1-" + col2 + ".s1") < q(col2 + ".s1-" + col3 + ".s1")
        )
    return qc[query]


## less than
def lt(col, var, mult=1):
    query = "lt_" + col + str(var) + str(mult)
    if query not in qc:
        qc[query] = q(col) < mult * var
    return qc[query]


## more than median
def gt(col, var=1, mult=1):
    query = "gt_" + col + str(var) + str(mult)
    if query not in qc:
        qc[query] = q(col + ">" + mult * var)
    return qc[query]


## mean above
def mea(col, var, smp=3):
    query = "mab_" + col + str(var) + str(smp)
    if query not in qc:
        qc[query] = cc(col, smp, "mean") > var
    return qc[query]


## mean below
def meb(col, var, smp=3):
    query = "mab_" + col + str(var) + str(smp)
    if query not in qc:
        qc[query] = cc(col, smp, "mean") < var
    return qc[query]


def ltm(col, denom=1):
    query = "ltm_" + col + str(denom)
    if query not in qc:
        qc[query] = q(col) / cc(col, 0, "median") < denom
    return qc[query]
