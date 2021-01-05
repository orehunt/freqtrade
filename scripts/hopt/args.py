import argparse
from collections.abc import Mapping
from typing import Any


def parse_hopt_args() -> Any:
    parser = argparse.ArgumentParser()

    parser.add_argument("-prf", help="run freqtrade with profiler", default="")
    parser.add_argument("-clr", help="clear hyperopt", action="store_true")
    parser.add_argument("-res", help="reset hyperopt", action="store_true")
    parser.add_argument("-a", help="auto mode", action="store_true")
    parser.add_argument("-r", help="recover previous runs", action="store_true")
    parser.add_argument(
        "-w", help="wrap again from specified index", default=-1, type=int
    )
    parser.add_argument("-j", help="number of jobs", type=int, default=8)
    parser.add_argument("-e", help="number of epochs", type=int, default=0)
    parser.add_argument("-f", help="effort", default=1, type=float)
    parser.add_argument(
        "-path", help="specify a path to a file or directory", default="", type=str
    )
    parser.add_argument(
        "-merge",
        help="specify saved data to be merged as one",
        default=False,
        action="store_true",
    )
    parser.add_argument("-pcond", help="print conditions", default="")
    parser.add_argument("-b", help="clear backtest cache", action="store_true")
    parser.add_argument("-s", help="no sell signal", action="store_true")
    # amounts:
    # "[on|off]:[roi|trailing|stoploss]": use default amounts config, don't optimize
    parser.add_argument(
        "-amt", help="spaces for amounts", default="on:roi,stoploss,trailing", type=str
    )
    parser.add_argument("-risk", help="amounts config", default="", type=str)
    parser.add_argument("-pp", help="path for pairlist for profits tuning", default="")
    parser.add_argument("-ne", help="optimize evals number", type=str, default="")
    parser.add_argument(
        "-se",
        help="skip evals, in the caseevals are copied manually",
        action="store_true",
    )
    parser.add_argument(
        "-sc",
        help="skip addcond when testing conditions evaluation",
        action="store_true",
    )
    parser.add_argument(
        "-mp", help="max positions (max open trades)", action="store_true"
    )
    parser.add_argument("-mt", help="min trades")
    parser.add_argument("-mavp", help="minimum average profit", default=0)
    parser.add_argument(
        "-mx",
        help="max number of trials per worker before logging",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-tm",
        help="max number of seconds before logging by each worker",
        type=int,
        default=None,
    )
    parser.add_argument("-stack", help="position stacking", action="store_true")
    parser.add_argument("-d", help="number of days", type=int, default=0)
    parser.add_argument("-g", help="timerange", type=str, default="")
    parser.add_argument("-i", help="timeframe", type=str, default="1h")
    parser.add_argument("-cvg", help="max convergence ratio", type=float, default=0.5)
    parser.add_argument("-rand", help="random state", type=int)
    parser.add_argument("-alg", help="optimizer algo", type=str, default="skopt:rand")
    parser.add_argument("-smp", help="set number of samples", type=int, default=1000)
    parser.add_argument("-psco", help="print scores", action="store_true")
    parser.add_argument("-inst", help="trials instance name", type=str, default="")
    parser.add_argument(
        "-lo",
        help="loss function, if split the first part is for trials instance",
        type=str,
    )
    parser.add_argument("-z", help="test condition type", type=str)
    parser.add_argument("-noprint", help="dont print all", action="store_true")
    parser.add_argument("-nojson", help="dont print json", action="store_false")
    parser.add_argument("-nobest", help="dont print best at end", action="store_false")
    parser.add_argument("-sgn", help="signals buy/sell both", type=str, default="")
    parser.add_argument("-mode", help="use one optimizer", default="shared")
    parser.add_argument("-kn", help="k out of n cross validation", action="store_true")
    parser.add_argument(
        "-cv", help="load CV trials instead of optimized trials", action="store_true"
    )
    parser.add_argument("-lie", help="lie strategy", default="cl_min")
    parser.add_argument("-mtc", help="optimizer meta tags conflict ", default="warn")
    parser.add_argument("-pts", help="number of points per trial", type=int, default=1)
    parser.add_argument(
        "-rpt", help="number of initial (random) points", type=int, default=0
    )
    parser.add_argument("-log", help="log to file", action="store_true", default="")
    parser.add_argument("-dbg", help="debug", default=0, type=int)
    parser.add_argument(
        "-k", help="how many parmaters groups to eval in one run", type=int, default=0
    )
    parser.add_argument(
        "-cond", help="the condition predicate to eval", type=str, default=""
    )
    parser.add_argument("-lim", help="opt limits", type=str, default="")
    parser.add_argument("-weg", help="opt limits", type=str, default="")
    parser.add_argument("-rb", help="restore backed up config", action="store_true")
    parser.add_argument("-ind", help="choose ind mode", type=str, default="weights")
    parser.add_argument("-pa", help="pairs")
    parser.add_argument("-qt", help="quote", default="usdt")
    parser.add_argument("-ua", help="update amounts", default="")
    parser.add_argument("-upp", help="update params", default="", type=str)
    parser.add_argument(
        "-pl",
        help="pick a subset of limits from a limit file (path) and print it",
        default="",
        type=str,
    )
    parser.add_argument("-q", help="quit afterwards", action="store_true")
    parser.add_argument("-cat", help="concatenate pair amounts")
    parser.add_argument("-fee", help="concatenate pair amounts", default=None)
    parser.add_argument(
        "-exc",
        help="exchange name which maps to a config file",
        default="exchange_testing",
    )
    parser.add_argument("-ppars", help="show params of epoch number", default="")
    parser.add_argument(
        "-pconf", help="print config to output or to file (-path)", action="store_true"
    )

    return parser.parse_args()


def update(d, u):
    """ nested dictionary update """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
