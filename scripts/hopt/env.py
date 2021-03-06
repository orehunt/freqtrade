import os
from typing import Any
import glob


def set_environment(args: Any, paths: dict, config) -> None:
    if not args.r:
        list(map(os.remove, glob.glob(f"{paths['pickle']}/*")))
    if args.b:
        list(map(os.remove, glob.glob(f"{paths['cache']}/*.pickle")))
        list(map(os.remove, glob.glob(f"{paths['pickle']}/*.pickle")))
    if not os.path.exists(paths["data"]):
        os.makedirs(paths["data"])
    # set env vars
    # never use pair based amounts in hyperopt
    os.environ["FQT_PBA"] = "1"
    os.environ["FQT_TIMEFRAME"] = config.get("timeframe", "1h")
    if args.s:
        os.environ["FQT_NOSS"] = "1"
    if args.sc:
        os.environ["FQT_SKIP_COND"] = "1"
    if args.z:
        os.environ["FQT_TEST_COND"] = args.z
    if args.ind:
        os.environ["FQT_IND_MODE"] = args.ind
        os.environ["FQT_PREDS"] = "1"
    if args.amt != 0 or args.ne:
        os.environ["FQT_SKIP_COND"] = "1"
    if args.sgn:
        os.environ["FQT_SGN"] = args.sgn
    if args.k:
        os.environ["FQT_K"] = str(args.k)
    if args.exc:
        os.environ["FQT_EXC"] = args.exc
    if args.ne:
        os.environ["FQT_N_WEIGHTS"] = args.ne.split(",")[0].split(":")[1]
