#!/usr/bin/env python

import glob
import json
import pickle
import re
import os
import subprocess
import logging
from colorama import Fore, Style
from pathlib import Path
from math import inf
from shutil import copy
from time import sleep, mktime
from datetime import datetime
from pandas import Timedelta, DataFrame
from statistics import mean
from hashlib import sha1
from multiprocessing import Lock
from joblib.externals.loky import get_reusable_executor

from user_data.modules import testsequences as ts

from freqtrade.configuration.configuration import Configuration, RunMode
from freqtrade.optimize.hyperopt import Hyperopt
from freqtrade.loggers import setup_logging
from freqtrade.data.btanalysis import load_trades

from scripts.hopt.args import parse_hopt_args
from scripts.hopt.env import set_environment
from scripts.hopt.time import OPT_TR, CV_TR

logger = logging.getLogger(__name__)

os.environ["NUMEXPR_MAX_THREADS"] = "16"
# os.environ["OMP_NUM_THREADS"] = "16"
os.environ["LD_LIBRARY_PATH"] = "/freqtrade/.env/lib"

paths = {
    "user": "user_data",
    "pickle": "/tmp/.freqtrade/testcondition",
    "cache": "/tmp/.freqtrade/hyperopt_cache",
    "spaces": "spaces.json",
    "best": "best.json",
    "limits": "limits.json",
    "weights": "weights.json",
    "params": "params.json",
    "run": "run.json",
}
paths["data"] = f"f{paths['user']}/hyperopt_data"
paths["archive"] = f"{paths['user']}/hyperopt_archive"
paths["evals"] = f"{paths['data']}/evals.json"
paths["pos"] = f"{paths['pickle']}/pos.json"
paths["score"] = f"{paths['user']}/indicators/score.json"


ho: Hyperopt


class Main:

    args = parse_hopt_args()

    timeframe = args.i
    days = args.d
    pairlist_name = "10e-4_spread_10e4_vol"

    config_dir = "cfg/"
    hyperopt_config = f"{config_dir}/hyperopt.json"
    exchange_config = f"{config_dir}/{args.exc}.json"
    amounts_tuned_config = f"{config_dir}/amounts_backtesting_tuned.json"
    exchange_overrides = []
    for f in ["amounts_backtesting"]:
        if os.path.exists(f"{config_dir}/{args.exc}/{f}.json"):
            exchange_overrides.append(f"{config_dir}/{args.exc}/{f}.json")
        else:
            exchange_overrides.append(f"{config_dir}/{f}.json")

    roi_config = f"{config_dir}/roi/{timeframe}.json"
    config_files = [
        hyperopt_config,
        exchange_config,
        *exchange_overrides,
        f"{config_dir}/pairlists_static.json",
        f"{config_dir}/amounts/default.json",
        f"{config_dir}/live.json",
        f"{config_dir}/askbid.json",
        f"{config_dir}/paths.json",
    ]

    config = {}
    config["user_data_dir"] = Path(paths["user"])
    config["hyperopt_loss"] = "DecideCoreLoss" if not args.lo else args.lo
    config["hyperopt_clear"] = args.clr
    config["hyperopt_reset"] = args.res
    config["hyperopt_jobs"] = args.j
    config["hyperopt_trials_maxout"] = args.mx or int(args.j)
    config["hyperopt_trials_timeout"] = args.tm or 60
    config["hyperopt_trials_instance"] = args.inst or (
        "cv" if (args.cv and not args.kn) else None
    )
    config["epochs"] = args.e
    config["mode"] = args.mode
    config["lie_strat"] = args.lie
    config["effort"] = args.f
    config["ask_points"] = args.pts
    config["print_json"] = True
    config["print_all"] = not args.np
    config["print_colorized"] = True
    config["verbosity"] = args.dbg
    config["logfile"] = "hyperopt.log" if args.log else ""
    config["position_stacking"] = not args.ns
    config["use_max_market_positions"] = args.mp
    config["hyperopt_random_state"] = args.rand if args.rand else None
    config["timeframe"] = timeframe
    config["ticker_interval"] = timeframe
    config["fee"] = args.fee
    base_config = config

    # spaces config
    sgn = args.sgn.split(",")
    if sgn[0] == "":
        sgn = []
    spaces = set()
    spaces_amounts_types = [["roi"], ["stoploss", "trailing"]]

    # make a stub lock to save trials from outside hyperopt
    stub = lambda: None
    stub.lock = Lock()

    def calc_trades(self):
        self.calc_days()
        mintrades = (
            int(
                int((self.n_tickers * self.days) / 14)
                if not self.args.mt
                else self.args.mt
            )
            or 3
        )
        self.config["hyperopt_min_trades"] = mintrades
        if self.n_tickers > 1 and self.timerange == OPT_TR:
            logger.warn(
                f"Using mintrades {mintrades} with many pairs on older timerange."
            )
        self.maxtrades = int(self.days * 2)

    def calc_days(self):
        start, stop = self.timerange.split("-")
        if not start:
            start = datetime.now().timetuple()
            stop = datetime.strptime(stop, "%Y%m%d").timetuple()
        elif not stop:
            stop = datetime.now().timetuple()
            start = datetime.strptime(start, "%Y%m%d").timetuple()
        else:
            start, stop = (
                datetime.strptime(t, "%Y%m%d").timetuple() for t in (start, stop)
            )
        start, stop = (mktime(t) for t in (start, stop))
        self.days = int((stop - start) // (60 * 60 * 24))
        self.config["days"] = self.days

    def setup_timerange(self):
        if self.args.kn:
            self.timerange = OPT_TR
        elif self.args.d:
            self.timerange = (datetime.now() - Timedelta(f"{self.days}d")).strftime(
                "%Y%m%d-"
            )
        elif self.args.g:
            try:
                trs, n = self.args.g.split(":")
                self.timerange = CV_TR[int(n) - 1]
            except ValueError:
                self.timerange = self.args.g
        else:
            self.timerange = CV_TR[0]
        self.config["timerange"] = self.timerange

    def __init__(self):
        self.n_tickers = self.get_tickers_number()
        self.setup_timerange()
        self.calc_trades()

        self.setup_config()
        setup_logging(self.config)

        self.ho = Hyperopt(self.config)

    def setup_config(self, space_type=None):
        set_environment(self.args, paths, self.config)
        self.config["runmode"] = RunMode.HYPEROPT
        self.config["command"] = "hyperopt"
        self.config["datadir"] = (
            self.config["user_data_dir"]
            / "data"
            / self.read_json_file(self.exchange_config, "exchange")["name"]
        )
        self.config_spaces(space_type)
        config = Configuration.from_files(self.config_files)
        config.update(self.config)
        self.config = config
        self.config.update(self.read_json_file(self.roi_config))
        self.update_evals()

    def update_config(self, space_type=None):
        self.config["timerange"] = self.timerange
        self.config["timeframe"] = self.timeframe
        self.calc_trades()

        self.config_spaces(space_type)
        # don't prefer sell signal when optimizing buy space
        if "buy" in self.spaces:
            self.config["ask_strategy"]["prefer_sell_signal"] = False
            logger.warn("Disabling prefer_sell_signal")
            self.config["ask_strategy"]["ignore_roi_if_buy_signal"] = True
            logger.warn("Enabling ignore_roi_if_buy_signal")
        elif "sell" in self.spaces:
            self.config["ask_strategy"]["prefer_sell_signal"] = True
            logger.warn("Enabling prefer_sell_signal")
        self.update_evals()

    def parse_pairs(self, pa=args.pa):
        tp = ""
        if type(pa) == str:
            pas = pa.split(":")
            if len(pas) > 1:
                tp = pas[0]
                pairs = pas[1]
            else:
                pairs = pas[0]
            if tp == "p":
                return [pairs]
            elif tp == "l":
                return pairs.split(",")
            # keep n profitable pairs or discard n profitable pairs
            elif tp in ("b", "bk", "bd"):
                exportfilename = f"{self.config['user_data_dir']}/backtest_results/{self.timeframe}.json"
                trades = load_trades(
                    "file",
                    db_url=self.config.get("db_url"),
                    exportfilename=exportfilename,
                )
                pairs_count = int(pairs)
                pairs = trades["pair"].drop_duplicates().values.tolist()
                pairs_profit = {pair: 0 for pair in pairs}
                for pair in pairs:
                    pairs_profit[pair] = trades[trades.pair == pair].profitperc.sum()
                sorted_by_profit = sorted(pairs_profit, key=lambda k: pairs_profit[k])
                if tp == "bk":
                    return sorted_by_profit[-abs(pairs_count) :]
                else: # b, bd
                    return sorted_by_profit[: -abs(pairs_count)]
            elif tp == "e":
                exc = self.read_json_file(self.config_dir + pairs + ".json")
                return exc["exchange"]["pair_whitelist"]
            else:
                return self.read_json_file(pa)
        else:
            return pa

    def update_pairs(self, pa=args.pa, config=False):
        """ returns the previous pairs """
        if config:
            prev_pairs = self.config["exchange"]["pair_whitelist"]
            self.config["exchange"]["pair_whitelist"] = self.parse_pairs(pa)
            return prev_pairs
        else:
            data = self.read_json_file(self.exchange_config)
            prev_pairs = data["exchange"]["pair_whitelist"]
        if not pa:
            # construct pair file from vars
            pa = "{}/pairlists/{}_{}_{}.json".format(
                self.config_dir,
                self.config["exchange"]["name"].lower(),
                self.config["stake_currency"].lower(),
                self.pairlist_name,
            )
            data["exchange"]["pair_whitelist"] = self.parse_pairs(pa)
            self.write_json_file(data, self.exchange_config)
        else:
            data["exchange"]["pair_whitelist"] = self.parse_pairs(pa)
            self.write_json_file(data, self.exchange_config, update=False)
        print("wrote the pairslist to:", self.exchange_config)
        return prev_pairs

    def config_spaces(self, space_type=None, clear=False):
        if clear:
            self.write_json_file({}, paths["spaces"], update=False)
        spaces_dict = {"timeframe": self.args.i}
        config = {}
        if self.args.amt != 0 or self.args.ne:
            config["hyperopt_loss"] = "SharpeLoss"
            if self.args.amt != 0:
                # disable position stacking when optimizing amounts
                config["position_stacking"] = False
            # in -1 first run is all, then 1 by 1
            # in 1 only one run with all spaces
            if self.args.amt in (-1, 1) and space_type == None:
                self.spaces.update(["roi", "stoploss", "trailing"])
            elif space_type != None:
                config.update(self.read_json_file(self.amounts_tuned_config))
                # delete previous amounts when optimizing 1 by 1
                for s in ("roi", "stoploss", "trailing"):
                    self.spaces.discard(s)
                self.spaces.update(self.spaces_amounts_types[space_type])
            else:
                config.update(self.read_json_file(self.roi_config))
                self.spaces.update(self.sgn if self.sgn else ["buy"])
            if self.args.ne:
                if not self.args.lo:
                    config["hyperopt_loss"] = "DecideCoreLoss"
                self.spaces.update(self.sgn if self.sgn else ["buy", "sell"])
                # by opt_weights
                if hasattr(self, "ne"):
                    sig = self.ne.split(",")
                else:
                    sig = self.args.ne.split(",")
                cs = {}
                for s in sig:
                    name, count = s.split(":")
                    cs.update({name: count})
                spaces_dict["custom"] = cs
                self.spaces_dict = spaces_dict
            update = False
        else:
            config["hyperopt_loss"] = "DecideCoreLoss"
            self.spaces.update(self.sgn if self.sgn else ["buy"])
            update = True
        if not self.spaces:
            self.spaces.update(self.sgn if self.sgn else ["buy"])
        if self.args.lo:
            config["hyperopt_loss"] = self.args.lo
        self.write_json_file(spaces_dict, paths["spaces"], update=update)
        config["spaces"] = self.spaces
        self.config.update(config)

    def get_tickers_number(self):
        with open(self.exchange_config, "r") as fe:
            fj = "".join(line for line in fe if not re.match("\s*//", line))
            exc_cfg = json.loads(fj)
        n_tickers = len(exc_cfg["exchange"]["pair_whitelist"])
        os.environ["FQT_N_TICKERS"] = str(n_tickers)
        return n_tickers

    def run_hyperopt(self, space_type=None, cv=False):
        """ run freqtrade process and update the best result from hyperopt """
        # update spaces
        # go directly into CV mode from a previous run, so we skip the first optimization run
        if self.args.kn and self.args.mode == "cv" and not cv:
            self.opt_cv()
            return
        else:
            self.update_config(space_type)
        print(
            Fore.GREEN
            + Style.BRIGHT
            + "\n*** STARTING HYPEROPT ***\n"
            + Style.RESET_ALL
        )
        # when profiling run freqtrade directly
        if self.args.prf != "":
            # normalize variables
            self.config["user_data_dir"] = str(self.config["user_data_dir"])
            self.config["datadir"] = str(self.config["datadir"])
            self.config["exportfilename"] = str(self.config["exportfilename"])
            self.config["spaces"] = list(self.config["spaces"])
            if "runmode" in self.config:
                del self.config["runmode"]
            # write config
            self.write_json_file(self.config, paths["run"], update=False)
            if self.args.prf == "1":
                main_exec = ["freqtrade"]
            else:
                main_exec = [
                    "python",
                    "-m",
                    *self.args.prf.split(","),
                    "/freqtrade/.env/bin/freqtrade",
                ]
            fqargs = [
                *main_exec,
                "hyperopt",
                "-c={}".format(paths["run"]),
                "--mode={}".format(self.args.mode),
                f"--ask-points={self.args.pts}",
            ]
            sanitized_args = list(filter(None, fqargs))
            while True:
                try:
                    p = subprocess.Popen(sanitized_args)
                    p.wait()
                except RuntimeError:
                    pass
                if p.returncode == 0 or not self.args.a:
                    break
                else:
                    sleep(5)
        else:
            # shutdown joblib pool to cleanup global state
            get_reusable_executor().shutdown(wait=False)
            self.ho = Hyperopt(self.config)
            self.ho.start()
        # perform CV after an optimization run
        if self.args.kn and not cv:
            self.opt_cv()
            return

    def get_trials(
        self, wait=1, path=None, instance=None, ignore_empty=False, cv=False
    ) -> (DataFrame, dict, dict):
        # remove clear flag before setup if we are getting trials
        clear_status = self.ho.config["hyperopt_clear"]
        reset_status = self.ho.config["hyperopt_reset"]
        self.ho.config["hyperopt_clear"] = False
        self.ho.config["hyperopt_reset"] = False
        if not path:
            if not hasattr(self.ho, "trials_file"):
                self.ho.setup_trials(load_trials=False)
            path = self.ho.trials_file
        if not instance and hasattr(self.ho, "trials_instance"):
            instance = self.ho.trials_instance
        elif instance == "last":
            instance = self.ho.get_last_instance(self.ho.trials_instances_file, cv=cv)
        if wait:
            tries = 0
            while not os.path.exists(path) and tries < 3:
                sleep(wait)
                tries += 1
        if not os.path.exists(path):
            return {}, {}, {}
        if cv:
            instance = instance.replace("_cv", "") + "_cv"
        try:
            trials = self.ho.load_trials(path, instance)
        except EOFError:
            return {}, {}, {}
        finally:
            self.ho.config["hyperopt_clear"] = clear_status
            self.ho.config["hyperopt_reset"] = reset_status
        if len(trials) < 1:
            if not ignore_empty:
                raise ValueError("No trials found!")
            else:
                return [], {}, {}
        # the last epoch is needed to check correct paramters in case of continuing epochs
        last = self.ho.trials_to_dict(trials.iloc[-1:, :])[0]
        # sort is ascending, and loss min is better so iloc[:1] to get the best value
        best = self.ho.trials_to_dict(trials.sort_values("loss").iloc[:1, :])[0]
        return trials, best, last

    prev_cond, cond_best, run_best = {}, inf, inf

    def save_results(self, c: int, i: int, w: int, prev_params: []):
        _, best, last = self.get_trials()
        last_params = last["params_dict"]
        null_params = dict.fromkeys(last_params, None)
        best = null_params if "loss" not in best else best
        if "params_dict" not in best or not self.match_condition_at_index(
            c, last_params
        ):
            params = null_params
        else:
            params = best["params_dict"]
        preds = ts.tconds[c][1]
        len_preds = len(preds)
        skipor = False
        if (len(prev_params) > c and len(prev_params[c]) < len(preds[0])) or w >= 0:
            close_len_preds = (
                len_preds - 1
            )  # sub 1 to make sure i+1 is an existing index
            pred = preds[0]
            skipor = False
            parmatch = False
            predmatch = False
            for cnd in prev_params[c]:
                if params == cnd:
                    parmatch = True
                    break
            if prev_params[c][-1].keys() == last_params.keys():
                predmatch = True
            if parmatch and (
                (i > 0 and params == last_params)
                or (predmatch and params != last_params)
            ):
                # null if the same
                prev_params[c].append(null_params)
                # print(1)
            else:
                # the last best is not the absolute best
                if (
                    best != null_params
                    and best["loss"] <= self.cond_best
                    and last_params == params
                ):
                    prev_params[c].append(params)
                    # print(3, 1)
                else:
                    prev_params[c].append(null_params)
                    # print(3, 2)
            # null and skip next one if still the same
            if (
                i < close_len_preds
                and pred[i] == pred[i + 1]
                and prev_params[c][-1] == null_params
            ):
                skipor = True
                prev_params[c].append(null_params)
                # increase the or counter (o) because with skip the next eval
                self.update_pickle_cache(1)
                # print(4)
        else:
            prev_params.append([params])
            # reset best loss when testing a new condition
            # print(5)
        # save parameters
        self.write_json_file(prev_params, "params.json", update=False)
        self.update_evals(prev_params)
        # update the best loss for the condition
        if "loss" in best and best["loss"] < self.cond_best:
            self.cond_best = best["loss"]
            self.write_json_file({"cond_best": self.cond_best}, BEST_PATH, update=True)
        return prev_params, skipor

    def read_json_file(self, file_path="params.json", key=""):
        with open(file_path, "r") as fe:
            fj = "".join(line for line in fe if not re.match("\s*(//|/\*|\*/)", line))
            data = json.loads(fj)
            if key != "":
                return data[key] if key in data else None
            else:
                return data

    @staticmethod
    def read_pickle_file(file_path="params.pickle", key=""):
        with open(file_path, "rb") as fp:
            if key != "":
                data = pickle.load(fp)
                return data[key] if key in data else None
            else:
                return pickle.load(fp)

    def write_pickle_file(self, data={}, file_path="params.pickle"):
        with open(file_path, "bw") as fp:
            pickle.dump(data, fp)

    def write_json_file(self, data={}, file_path="params.json", update=True):
        if not os.path.exists(file_path):
            with open(file_path, "w") as fp:
                json.dump({}, fp)
        with open(file_path, "r+") as fp:
            if update:
                dataf = json.load(fp)
                dataf.update(data)
            else:
                dataf = data
            fp.seek(0)
            fp.truncate()
            json.dump(dataf, fp, indent=4)

    def update_evals(self, prev_params=[]):
        if (
            self.args.se or self.config["strategy"] != "DecideStrategy"
        ):  # only run if strategy is DecideStrategy
            return
        if prev_params == []:
            prev_params = self.read_json_file(paths["params"])
        evals = ts.sequencer({}, self.sgn if self.sgn else ["buy"], prev_params)[4]
        self.write_json_file(evals, paths["evals"], update=False)

    def update_pickle_cache(self, a=0) -> bool:
        try:
            for rt, dr, fl in os.walk(paths["pickle"]):
                for fp in fl:
                    with open(fp, "r+") as f:
                        ret = pickle.load(f)
                        ret[3] += a
                        pickle.dump(ret, f)
            return True
        except Exception as e:
            print(e)
            return False

    def hashcond(self, cond):
        return sha1(bytes(f"{cond}", "UTF-8")).hexdigest()

    def process_last_condition(self, c: int, prev_params: list, skipor: True) -> list:
        """ decide if the last full evald condition should be kept, and save
        indicator score """
        # it's worse so reset condition because not good
        if self.cond_best > self.run_best and not skipor:
            if self.wrap or self.args.w != -1 and self.prev_cond:
                # put back what was before
                prev_params[c] = self.prev_cond
            else:
                # we never null if we are not wrapping
                # even if the objective is below run best
                pass
            self.write_json_file(prev_params, paths["params"], update=False)
        else:  # it's better so we update the run best loss
            self.run_best = self.cond_best
            self.write_json_file({"run_best": self.run_best}, paths["best"])
        # copy params.json to always have a copy with completed conditions
        copy(paths["params"], paths["params"] + ".1")  # backup params config
        # save score of the evald indicator
        score = self.cond_best / self.run_best
        cond = ts.tconds[c]
        key = self.hashcond(cond)
        # above >1 good, <1 bad
        if self.args.j > 4:  # in order to avoid low jobs test runs
            scores = self.read_json_file(paths["score"])
            if key in scores:
                scores[key].append(score)
            else:
                scores[key] = [score]
            self.write_json_file(scores, paths["score"], update=True)
        # reset cond_best since switching condition
        self.cond_best = inf
        return prev_params

    def match_condition_at_index(self, c: int, last_params: dict):
        """ loop over the keys in the parameters to make sure the paramters
        are of the condition at the current index """
        for k in last_params:
            for part in k.split(":"):
                for s in self.spaces:
                    part = part.strip(f"{s}_")
                for ind in ts.tconds[c][0]:
                    if ind == part:
                        return True
        return False

    def print_scores(self):
        scores = self.read_json_file(paths["score"])
        for n, cond in enumerate(ts.tconds):
            chash = self.hashcond(cond)
            if chash in scores:
                print(mean(scores[chash]), n, cond[0])

    def print_params(self):
        mode, num = self.args.ppars.split(":")
        self.ho.config["hyperopt_list_filter"] = False
        if self.args.inst:
            instance = self.args.inst
        else:
            instance = self.ho.get_last_instance(self.ho.trials_instances_file)
        trials, _, _ = self.get_trials(instance=instance, cv=(mode == "cv"))
        idx = int(num) - 1  # human
        trial = self.ho.trials_to_dict(trials.iloc[idx : idx + 1, :])[0]
        self.ho.print_epoch_details(trial, len(trials), True)
        # print(json.dumps(trial["params_details"]))

    def print_conditions(self):
        """
        converts conditions optimized parameters into eval list for runtime
        """
        try:
            src, dst = self.args.p.split(":")
        except:
            src = "params.json.1"
            dst = self.args.p
            print(f"no src given, defaulting to {src}")
        with open(src, "r") as fp:
            prev_params = json.load(fp)
            evals = ts.sequencer(
                {}, prev_params=prev_params, sgn=self.args.sgn.split(",")
            )[4]
        for e in evals:  # delete empty conditions
            for i in e:
                for n in range(1, len(i)):
                    b = 1
                    while i[n - b] == "" and (n - b) > 0:
                        b += 1
                    i[n] = "" if i[n] == i[n - b] else i[n]
        with open(dst, "w") as fp:
            json.dump(evals, fp)

    def pick_limits(self):
        limits = self.read_json_file(self.args.path)
        picked = {}
        bounds, to_pick = self.args.pl.split(":")
        bounds = bounds.split(",")
        to_pick = to_pick.split(",")
        sgn = self.args.sgn.split(",")
        for sg in sgn:
            for b in bounds:
                c = 0
                for n in to_pick:
                    picked[f"{sg}.{b}{c}"] = limits[f"{sg}.{b}{n}"]
                    c += 1
        print(json.dumps(picked))

    def opt_limits(self):
        self.config_spaces()
        start, stop = self.args.lim.split(":")
        start = int(start) if start else 0
        stop = int(stop)
        if start < 1:
            self.write_json_file({}, paths["limits"], update=False)
        k = self.args.k or int(self.args.ne.split(",")[0].split(":")[1])
        se = start
        start = start // k + start % k
        stop = stop // k + stop % k
        for n in range(start, stop):
            os.environ["FQT_SE"] = str(se)
            if n + k > stop:
                k = stop - n
            os.environ["FQT_K"] = str(k)
            self.run_hyperopt()
            # reset hyperopt trials
            self.config["hyperopt_reset"] = True
            # shutdown joblib pool to cleanup global state
            get_reusable_executor().shutdown(wait=False)
            trials, best, _ = self.get_trials(cv=self.args.kn)
            # convert to int to allow serialization
            for v in best["params_dict"]:
                num = best["params_dict"][v]
                best["params_dict"][v] = float(num)
            self.write_json_file({se: best["params_dict"]}, paths["limits"])
            se += k

    def opt_weights(self):
        # check that a best is configured before optimizing further
        self.update_new_best(-inf)
        lim_count, start_stop = self.args.weg.split(",")
        start, stop = start_stop.split(":")
        total_start = int(start) if start else 0
        total_stop = int(stop)
        if total_start < 1:
            self.write_json_file({}, paths["weights"], update=False)
        template_names = [s.split(":")[0] for s in self.args.ne.split(",")]
        init_k = self.args.k or int(self.args.ne.split(",")[0].split(":")[1])
        init_se = int(start)
        init_start = total_start // init_k + total_start % init_k
        init_stop = total_stop // init_k + total_stop % init_k
        if total_start + init_k > total_stop:
            logger.warn(
                f"Number of replicas ({init_k}) too high for given range ({total_stop-total_start}), "
                f"setting it to {total_stop-total_start}"
            )
            init_k = total_stop - total_start
        path = Path(self.args.path)
        path_bak = f"{path}.bak"
        if not path:
            raise RuntimeError("Provide a path to backup existing weights")
        elif self.args.rb:
            if os.path.exists(path_bak):
                copy(path_bak, path)
            else:
                logger.error("Can't restore weights since there is no backup file.")
        else:
            saved_weights = self.read_json_file(path)
            copy(path, path_bak)
            logger.info(f"Wrote backup weights to {path_bak}")
        for nl in range(int(lim_count)):
            se = init_se
            k = init_k
            start = init_start
            stop = init_stop
            evals = ""
            for tn in template_names:
                evals = f"{evals}{nl}.{tn}:{k},"
            self.ne = evals.rstrip(",")
            for n in range(start, stop):
                os.environ["FQT_SE"] = str(se)
                # merge or reduce the last
                if total_stop - n * k < k + k // 2:
                    k = total_stop - n * k
                    evals = ""
                    for tn in template_names:
                        evals = f"{evals}{nl}.{tn}:{k},"
                    self.ne = evals.rstrip(",")
                os.environ["FQT_K"] = str(k)
                self.run_hyperopt()
                # reset hyperopt trials
                self.config["hyperopt_reset"] = True
                # shutdown joblib pool to cleanup global state
                get_reusable_executor().shutdown(wait=False)
                trials, best, _ = self.get_trials(cv=self.args.kn)
                if len(trials) < 1:
                    raise ValueError("No trials, probably wrong params")
                # convert to int to allow serialization
                for v in best["params_dict"]:
                    num = best["params_dict"][v]
                    best["params_dict"][v] = float(num)
                self.write_json_file({se: best["params_dict"]}, paths["weights"])
                logger.info("Updating params with: \n", best["params_dict"])
                self.write_json_file(best["params_dict"], self.args.path, update=True)
                se += int(k)

    def update_new_best(self, candidate: float):
        run_best = self.read_json_file(paths["best"], "run_best")
        if not run_best:
            raise RuntimeError(
                f"Run best at {paths['best']} is null, update with proper value."
            )
        if candidate < run_best:
            self.write_json_file({"run_best": candidate}, paths["best"])
            return True
        else:
            return False

    def opt_conditions(self, prev_params: dict, w: int = args.w) -> list:
        # recall last best loss if we are resuming
        if os.path.exists(paths["best"]):
            saved_best = self.read_json_file(paths["best"])
            if "cond_best" in saved_best:
                self.cond_best = saved_best["cond_best"] or inf
            if "run_best" in saved_best:
                self.run_best = saved_best["run_best"] or inf
        for c in range(max(0, w), len(ts.tconds)):
            ln_cp = len(prev_params)
            ln_tc = len(ts.tconds)
            ln_lcp = (
                len(prev_params[-1]) if (ln_cp > 0 and prev_params[-1] != None) else 0
            )
            ln_lcs = len(ts.tconds[-1][1][0])
            self.wrap = ln_cp == ln_tc and ln_lcp == ln_lcs
            # store the previous full condition best value for comparison
            if (
                c < ln_cp and c < w and w < 0
            ):  # if wrap is >0 it is set and we always loop
                print(f"{c} continue..")
                self.cond_best = inf  # reset cond best since switching condition
                continue
            skipor = False
            # used for naming scheme
            inds = ts.tconds[c][0]
            preds = ts.tconds[c][1]
            pfxs = ts.tconds[c][2]
            # members of the predicates group have to be all the same length
            ln_ors = len(preds[0])
            ln_cpc = len(prev_params[c]) if c < ln_cp else 0
            # if we are wrapping we have to clean up current condition for re eval
            if self.wrap or w >= 0:
                if w >= 0:  # its a wrap
                    if ln_ors != 0 and ln_ors == ln_cpc:
                        self.prev_cond = prev_params[c]
                        prev_params[c] = []
                    else:  # it's a new cond
                        self.prev_cond = []
                self.update_evals(prev_params)
            if c == ln_cp or (self.wrap and ln_ors == 0) or w != -1 or ln_cp == 0:
                print("resetting epochs...")
                self.clear = "--clear"  # reset epochs on new conditions predicates
                self.next_cond = True
                ln_cpc = 0
            else:
                print("continuing epochs...")
                self.next_cond = False
                ln_cpc = len(prev_params[c]) if c < ln_cp else 0
            for i in range(0, ln_ors):
                if (
                    i < ln_cpc or skipor
                ):  # skipor means a new null iteration was already appended
                    skipor = 0
                    skipand = 1  # skipand is used to bypass process last condition
                    continue
                skipand = 0
                # clear cache at every iteration because the sequencer
                # config is updated by save_results
                list(map(os.remove, glob.glob(f"{paths['pickle']}/*")))
                # write parameters to spaces.json to correctly generate
                # the search space according to the current condition
                dimensions = {
                    dim: [pfxs, inds, [fname[i][0] for fname in preds]]
                    for dim in self.spaces
                }
                self.write_json_file(dimensions, paths["spaces"])
                # write current evaling position to disk
                with open(paths["pos"], "w") as fp:
                    json.dump({"n": c, "o": i}, fp)
                self.run_hyperopt()
                prev_params, skipor = self.save_results(
                    c=c, i=i, w=w, prev_params=prev_params
                )
            # after a full loop we decide if the condition evaluation is worth keeping
            prev_params = self.process_last_condition(c, prev_params, skipand)
        return prev_params

    def opt_amounts(self, spaces=args.amt):
        # disable ignore roi
        self.config["ask_strategy"]["ignore_roi_if_buy_signal"] = False
        # reset best
        self.write_json_file({"run_best": None}, paths["best"], update=False)
        len_spa = len(self.spaces_amounts_types)
        if spaces == 1:
            self.run_hyperopt()
            return
        else:
            if (
                spaces == -1
            ):  # this does a first run with all the limit, then loop one by one
                self.run_hyperopt()
                spaces = len_spa
            space_type = spaces % len_spa
            while space_type < len_spa:
                self.run_hyperopt(space_type)
                self.update_amounts()
                space_type += 1

    def opt_amounts_per_pair(self):
        pairs_amounts_path = f"{self.config_dir}/amounts/pairs"
        # start from specified pair or from the beginning of the list
        start = ""
        path = self.args.pp.split(":")
        if len(path) > 1:
            start, path = path[0], path[1]
        pairs = self.read_json_file(path)
        if start:
            prev_pairs = pairs[: pairs.index(start)]
            pairs = pairs[pairs.index(start) :]
            # remove all previous pairs
            list(map(os.remove, glob.glob(f"{pairs_amounts_path}/*.json")))
        else:  # from beginning we first run an opt_amounts for all pairs
            self.update_pairs(path)
            self.opt_amounts(-1)
            # copy the config to the pair amounts folder
            amounts = self.read_json_file(self.amounts_tuned_config)
            self.write_json_file(
                amounts, f"{self.config_dir}/amounts/pairs/all.json", update=False
            )
            # remove previous pair amounts (here so to not delete amounts from resume)
            for p in pairs:
                p_name = p.replace("/", "-")
                try:
                    os.remove(f"{pairs_amounts_path}/{p_name}.json")
                except OSError:
                    pass
        for p in pairs:
            self.update_pairs(f"p:{p}")
            self.opt_amounts(-1)
            amounts = self.read_json_file(self.amounts_tuned_config)
            # save
            p_name = p.replace("/", "-")
            p_path = f"{self.config_dir}/amounts/pairs/{p_name}.json"
            self.write_json_file(amounts, p_path, update=False)

    def update_amounts(self, ua=args.ua, amounts_path=amounts_tuned_config):
        ua = ua.split(":")
        if ua[0] == "cv":
            ua = int(ua[1])
            cv = True
        else:
            ua = int(ua[0]) if ua[0] else 0
            cv = False
        if ua > 0:
            results, _, _ = self.get_trials(
                wait=True, instance=(self.args.inst or "last"), ignore_empty=True, cv=cv
            )
            best = results.iloc[[ua - 1]]
        else:
            _, best, _ = self.get_trials(wait=False, cv=cv, ignore_empty=True)

        if not best:
            logger.info("Skipping update amounts as no best trial was found")
            return
        best = self.ho.trials_to_dict(best)[0]
        prev_best = self.read_json_file(paths["best"], "run_best")
        if len(best) < 1:
            print("update_amounts: empty results,", best)
            return
        if not ua and prev_best and best["loss"] > prev_best:
            print(f"update_amounts: best not lower, {best['loss']} > {prev_best}")
            return

        params = {}
        amounts = best["params_details"]
        if "roi" in best:
            roi = json.loads(best["roi"])
            params.update({"minimal_roi": {str(k): v for k, v in roi.items()}})
        if "stoploss" in amounts:
            params.update(amounts["stoploss"])
        if "trailing" in amounts:
            params.update(amounts["trailing"])
        if params:
            print(f"updating amounts at {amounts_path}")
            self.write_json_file(params, amounts_path)
            self.write_json_file(
                {"run_best": best["loss"]}, paths["best"], update=False
            )
        if self.args.q:
            exit(0)
        return

    def update_params(self, epoch=args.upp, path=args.path, instance=args.inst):
        results, _, _ = self.get_trials(instance=(instance or "last"))
        epoch = epoch.split(":")
        cv = epoch[0] == "cv"
        if cv:
            epoch = int(epoch[1])
        else:
            epoch = int(epoch[0])
        path = path.split(":")
        idx = int(path[1]) if len(path) > 1 else None
        path = path[0]
        trial = self.ho.trials_to_dict(results.loc[results["current_epoch"] == epoch])[
            0
        ]
        params = trial["params_dict"]
        data = self.read_json_file(path)
        if idx is not None:
            data[idx].update(params)
        else:
            data.update(params)
        self.write_json_file(data, path, update=False)
        self.ho.print_epoch_details(trial, len(results), True)
        # print(
        #     f"Updated file {path} with params:\n",
        #     trial["params_dict"],
        #     "\n\n",
        #     trial["results_explanation"],
        # )

    def concat_pairs_amounts(self, am: str):
        base_cfg, pairs_dir = am.split(":")
        base = self.read_json_file(base_cfg)
        pairs_amounts = {}
        for rt, dr, fl in os.walk(pairs_dir):
            for fp in fl:
                pair = fp.replace("-", "/").replace(".json", "")
                amounts = self.read_json_file(f"{rt}{fp}")
                pairs_amounts[pair] = amounts
        base.update({"amounts": pairs_amounts})
        return self.write_json_file(base, "pairs_amounts.json", update=False)

    def cv_reduce(self):
        n_tr = len(CV_TR) - 1
        if not hasattr(self, "reduce"):
            self.reduce = {}

        config, k, r = self.config, "hyperopt_list_pct_best", "pct_red"
        if config[k] not in ("mean", "std"):
            if r not in self.reduce:
                self.reduce[r] = config[k] / n_tr
            new_pct = config[k] - self.reduce[r]
            config[k] = new_pct if new_pct > 0 else self.reduce[r]
        config, k, r = self.config, "hyperopt_list_cutoff_best", "cutoff_red"
        if config[k] not in ("mean", "std"):
            if r not in self.reduce:
                self.reduce[r] = config[k] / n_tr
            new_cut = config[k] + self.reduce[r]
            config[k] = new_cut if new_cut < 1 else self.reduce[r]
        config, k, r = self.config["hyperopt_list_step_values"], "range", "step_red"
        if config[k] not in ("mean", "std"):
            if r not in self.reduce:
                self.reduce[r] = config[k] / n_tr
            new_range = config[k] - self.reduce[r]
            config[k] = new_range if new_range > 1 else self.reduce[r]
        config, k, r = self.config, "hyperopt_list_min_avg_profit", "min_avg_p_red"
        min_avg_p = config[k] or -1
        if r not in self.reduce:
            self.reduce[r] = abs(min_avg_p / n_tr)
        config[k] = min_avg_p + self.reduce[r]

    def opt_cv(self):
        # make sure the optimization was on the first timerange
        # if the first run was the optimization
        if self.args.mode is not "cv":
            assert self.timerange == OPT_TR
        # cross validate on the other two time ranges
        self.config["mode"] = "cv"
        # on the first cv sample all the metrics for good trials
        self.config.update(
            {
                "hyperopt_clear": False,
                "hyperopt_reset": False,
                "hyperopt_trials_instance": self.args.inst,
                "hyperopt_list_filter": True,
                "hyperopt_list_dedup": False,
                "hyperopt_list_min_total_profit": None,
                "hyperopt_list_min_avg_profit": self.args.mavp
                if not re.match("_cv$|^cv$", self.args.inst)
                else 0,
                "hyperopt_list_min_trades": self.config["hyperopt_min_trades"],
                "hyperopt_list_step_values": {"range": "std"},
                "hyperopt_list_step_metric": ["all"],
                "hyperopt_list_sort_metric": ["all"],
                "hyperopt_list_best": ["sum", "ratio"],
                "hyperopt_list_pct_best": "std",
                "hyperopt_list_cutoff_best": "std",
                # "hyperopt_list_step_metric": ["profit"],
                # "hyperopt_list_best": [],
            }
        )
        # loop over timeranges split for CV
        # apply linear reduction based on number of timeranges
        for tr in CV_TR:
            self.timerange = tr
            self.run_hyperopt(cv=True)
            self.cv_reduce()
            # after first cv fetch from cv saved trials
            self.config["hyperopt_trials_instance"] = "cv"
            # only allow unprofitable on the first cv; after that
            # only allow unprofitable if different pairs are getting optimized next
            if not self.args.pa:
                self.config["hyperopt_list_min_avg_profit"] = 0

        # cv against multiple pairs, if given, skip as doesn't have enough pairs
        if self.args.pa:
            prev_pairs = self.update_pairs(config=True)
            for tr in CV_TR:
                self.timerange = tr
                self.cv_reduce()
                self.run_hyperopt(cv=True)
            # restore pairs
            self.update_pairs(pa=prev_pairs)
        # reset options
        self.timerange = OPT_TR
        # change mode if we started from cv (resume)
        self.config["mode"] = self.args.mode if self.args.mode != "cv" else "shared"
        self.config["hyperopt_clear"] = self.args.clr
        self.config["hyperopt_reset"] = self.args.res
        self.config["hyperopt_trials_instance"] = False
        self.config["hyperopt_list_filter"] = False
        self.config["skip_trials_setup"] = False

    def opt_conds(self):
        """
        needs a list of conditions to be evaluated, one by one
        """
        self.config_spaces(clear=True)
        copy(paths["params"] + ".1", paths["params"] + ".2")  # backup params config
        if not self.args.r:
            self.write_json_file(
                [], paths["params"], update=False
            )  # reset params config
            self.write_json_file({}, paths["best"], update=False)  # reset best
        prev_params = self.read_json_file(paths["params"])
        self.update_evals(prev_params)
        while True and os.path.exists(paths["params"]):  # wrap indefinitely
            prev_params = self.opt_conditions(prev_params, self.args.w)
            self.args.w = 0  # after the first run we reset wrap because we are **surely** starting from 0
            sleep(1)

    def start(self):
        """
        run mode:
        a: automatic
        p: print evals
        sp: optimize profits
        t: loop t times
        lim: limits
        """
        if self.args.lim:
            self.opt_limits()
        elif self.args.weg:
            self.opt_weights()
        elif self.args.a:
            self.opt_conds()
        elif self.args.pcond:
            self.print_conditions()
        elif self.args.psco:
            self.print_scores()
        elif self.args.ppars:
            self.print_params()
        elif self.args.cat:
            self.concat_pairs_amounts(self.args.cat)
        elif self.args.pp:
            self.opt_amounts_per_pair()
        elif self.args.amt != 0:
            self.opt_amounts(self.args.amt)
        elif self.args.pl:
            self.pick_limits()
        elif self.args.pa and self.args.q:
            self.update_pairs()
        elif self.args.ua and self.args.q:
            self.update_amounts(ua=self.args.ua)
        elif self.args.upp and self.args.q:
            self.update_params(epoch=self.args.upp, path=self.args.path)
        else:
            self.run_hyperopt()


if __name__ == "__main__":
    main = Main()
    try:
        main.start()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
