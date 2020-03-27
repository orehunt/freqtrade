import logging
import warnings
from collections import OrderedDict
from numpy import iinfo, int32
from pathlib import Path
from typing import List, Dict

from joblib import dump, load
from pandas import isna, json_normalize
from os import path
import io

from freqtrade.exceptions import OperationalException
from freqtrade.misc import plural, round_dict

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
import freqtrade.optimize.hyperopt_backend as backend

# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401

from freqtrade.optimize.hyperopt_constants import VOID_LOSS

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from skopt import Optimizer
# Additional regressors already pluggable into the optimizer
# from sklearn.linear_model import ARDRegression, BayesianRidge
# possibly interesting regressors that need predict method override
# from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBoostRegressor


logger = logging.getLogger(__name__)


class HyperoptData:
    """
    Data and state for Hyperopt
    """

    # run modes
    mode: str
    multi: bool
    shared: bool
    cv: bool

    # total number of candles being backtested
    n_candles = 0

    # a guessed number extracted by the space dimensions
    search_space_size: int

    # used by update_max_epoch
    avg_last_occurrence: int
    current_best_epoch = 0
    current_best_loss = VOID_LOSS
    epochs_since_last_best: List = [0, 0]

    min_epochs: int
    max_epoch: int

    # evaluations
    trials: List = []
    num_trials_saved = 0

    opt: Optimizer

    def __init__(self, config):
        self.config = config
        # epochs counting
        self.total_epochs = self.config["epochs"] if "epochs" in self.config else 0
        self.epochs_limit = lambda: self.total_epochs or self.max_epoch

        # paths
        self.trials_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_results.pickle"
        )
        self.data_pickle_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_tickerdata.pkl"
        )
        self.opts_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_optimizers.pickle"
        )
        self.cv_trials_file = (
            self.config["user_data_dir"] / "hyperopt_results" / "hyperopt_cv_results.pickle"
        )

    def clean_hyperopt(self) -> None:
        """
        Remove hyperopt pickle files to restart hyperopt.
        """
        for f in [self.data_pickle_file, self.trials_file, self.cv_trials_file, self.opts_file]:
            p = Path(f)
            if p.is_file():
                logger.info(f"Removing `{p}`.")
                p.unlink()

    def save_trials(self, final: bool = False) -> None:
        """
        Save hyperopt trials
        """
        if self.cv:
            trials_path = self.cv_trials_file
        else:
            trials_path = self.trials_file
        num_trials = len(self.trials)
        if num_trials > self.num_trials_saved:
            logger.debug(f"\nSaving {num_trials} {plural(num_trials, 'epoch')}.")
            # save_trials(self.trials, trials_path, self.num_trials_saved)
            dump(self.trials, trials_path)
            self.num_trials_saved = num_trials
            if self.mode in ("single", "shared", "multi"):
                self.save_opts()
        if final:
            logger.info(
                f"{num_trials} {plural(num_trials, 'epoch')} " f"saved to '{self.trials_file}'."
            )

    def save_opts(self) -> None:
        """
        Save optimizers state to disk. The minimum required state could also be constructed
        from the attributes [ models, space, rng ] with Xi, yi loaded from trials.
        All we really care about are [rng, Xi, yi] since models are never passed over queues
        and space is dependent on dimensions matching with hyperopt config
        """
        # synchronize with saved trials
        opts = []
        n_opts = 0
        if self.multi:
            while not backend.optimizers.empty():
                opt = backend.optimizers.get()
                opt = HyperoptData.opt_clear(opt)
                opts.append(opt)
            n_opts = len(opts)
            for opt in opts:
                backend.optimizers.put(opt)
        else:
            # when we clear the object for saving we have to make a copy to preserve state
            opt = HyperoptData.opt_rand(self.opt, seed=False)
            if self.opt:
                n_opts = 1
                opts = [HyperoptData.opt_clear(self.opt)]
            # (the optimizer copy function also fits a new model with the known points)
            self.opt = opt
        logger.debug(f"Saving {n_opts} {plural(n_opts, 'optimizer')}.")
        dump(opts, self.opts_file)

    @staticmethod
    def _read_trials(trials_file: Path) -> List:
        """
        Read hyperopt trials file
        """
        logger.info("Reading Trials from '%s'", trials_file)
        trials = load(trials_file)
        return trials

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params[space]
        # Round floats to `r` digits after the decimal point if requested
        return round_dict(d, r) if r else d

    @staticmethod
    def _params_update_for_json(result_dict, params, space: str) -> None:
        if space in params:
            space_params = HyperoptData._space_params(params, space)
            if space in ["buy", "sell"]:
                result_dict.setdefault("params", {}).update(space_params)
            elif space == "roi":
                # Convert keys in min_roi dict to strings because
                # rapidjson cannot dump dicts with integer keys...
                # OrderedDict is used to keep the numeric order of the items
                # in the dict.
                result_dict["minimal_roi"] = OrderedDict(
                    (str(k), v) for k, v in space_params.items()
                )
            else:  # 'stoploss', 'trailing'
                result_dict.update(space_params)

    @staticmethod
    def export_csv_file(
        config: dict, results: list, total_epochs: int, highlight_best: bool, csv_file: str
    ) -> None:
        """
        Log result to csv-file
        """
        if not results:
            return

        # Verification for overwrite
        if path.isfile(csv_file):
            logger.error("CSV-File already exists!")
            return

        try:
            io.open(csv_file, "w+").close()
        except IOError:
            logger.error("Filed to create CSV-File!")
            return

        trials = json_normalize(results, max_level=1)
        trials["Best"] = ""
        trials["Stake currency"] = config["stake_currency"]
        trials = trials[
            [
                "Best",
                "current_epoch",
                "results_metrics.trade_count",
                "results_metrics.avg_profit",
                "results_metrics.total_profit",
                "Stake currency",
                "results_metrics.profit",
                "results_metrics.duration",
                "loss",
                "is_initial_point",
                "is_best",
            ]
        ]
        trials.columns = [
            "Best",
            "Epoch",
            "Trades",
            "Avg profit",
            "Total profit",
            "Stake currency",
            "Profit",
            "Avg duration",
            "Objective",
            "is_initial_point",
            "is_best",
        ]
        trials["is_profit"] = False
        trials.loc[trials["is_initial_point"], "Best"] = "*"
        trials.loc[trials["is_best"], "Best"] = "Best"
        trials.loc[trials["Total profit"] > 0, "is_profit"] = True
        trials["Epoch"] = trials["Epoch"].astype(str)
        trials["Trades"] = trials["Trades"].astype(str)

        trials["Total profit"] = trials["Total profit"].apply(
            lambda x: "{:,.8f}".format(x) if x != 0.0 else ""
        )
        trials["Profit"] = trials["Profit"].apply(
            lambda x: "{:,.2f}".format(x) if not isna(x) else ""
        )
        trials["Avg profit"] = trials["Avg profit"].apply(
            lambda x: "{:,.2f}%".format(x) if not isna(x) else ""
        )
        trials["Avg duration"] = trials["Avg duration"].apply(
            lambda x: "{:,.1f} m".format(x) if not isna(x) else ""
        )
        trials["Objective"] = trials["Objective"].apply(
            lambda x: "{:,.5f}".format(x) if x != 100000 else ""
        )

        trials = trials.drop(columns=["is_initial_point", "is_best", "is_profit"])
        trials.to_csv(csv_file, index=False, header=True, mode="w", encoding="UTF-8")
        print("CSV-File created!")

    @staticmethod
    def load_previous_results(trials_file: Path) -> List:
        """
        Load data for epochs from the file if we have one
        """
        trials: List = []
        if trials_file.is_file() and trials_file.stat().st_size > 0:
            trials = HyperoptData._read_trials(trials_file)
            if trials[0].get("is_best") is None:
                raise OperationalException(
                    "The file with Hyperopt results is incompatible with this version "
                    "of Freqtrade and cannot be loaded."
                )
            logger.info(f"Loaded {len(trials)} previous evaluations from disk.")
        return trials

    @staticmethod
    def load_previous_optimizers(opts_file: Path) -> List:
        """ Load the state of previous optimizers from file """
        opts: List[Optimizer] = []
        if opts_file.is_file() and opts_file.stat().st_size > 0:
            opts = load(opts_file)
        n_opts = len(opts)
        if n_opts > 0 and type(opts[-1]) != Optimizer:
            raise OperationalException(
                "The file storing optimizers state might be corrupted " "and cannot be loaded."
            )
        else:
            logger.info(f"Loaded {n_opts} previous {plural(n_opts, 'optimizer')} from disk.")
        return opts

    def setup_optimizers(self):
        """ Setup the optimizers objects, try to load from disk, or create new ones """
        # try to load previous optimizers
        opts = self.load_previous_optimizers(self.opts_file)
        n_opts = len(opts)

        if self.multi:
            max_opts = self.n_jobs
            rngs = []
            # when sharing results there is only one optimizer that gets copied
            if self.shared:
                max_opts = 1
            # put the restored optimizers in the queue
            # only if they match the current number of jobs
            if n_opts == max_opts:
                for n in range(n_opts):
                    rngs.append(opts[n].rs)
                    # make sure to not store points and models in the optimizer
                    backend.optimizers.put(HyperoptData.opt_clear(opts[n]))
            # generate as many optimizers as are still needed to fill the job count
            remaining = max_opts - backend.optimizers.qsize()
            if remaining > 0:
                opt = self.get_optimizer()
                rngs = []
                for _ in range(remaining):  # generate optimizers
                    # random state is preserved
                    rs = opt.rng.randint(0, iinfo(int32).max)
                    opt_copy = opt.copy(random_state=rs)
                    opt_copy.void_loss = VOID_LOSS
                    opt_copy.void = False
                    opt_copy.rs = rs
                    rngs.append(rs)
                    backend.optimizers.put(opt_copy)
                del opt, opt_copy
            # reconstruct observed points from epochs
            # in shared mode each worker will remove the results once all the workers
            # have read it (counter < 1)
            counter = self.n_jobs

            def empty_dict():
                return {rs: [] for rs in rngs}

            self.opt_empty_tuple = lambda: {rs: ((), ()) for rs in rngs}
            self.Xi.update(empty_dict())
            self.yi.update(empty_dict())
            self.track_points()
            # this is needed to keep track of results discovered within the same batch
            # by each optimizer, use tuples! as the SyncManager doesn't handle nested dicts
            Xi, yi = self.Xi, self.yi
            results = {tuple(X): [yi[r][n], counter] for r in Xi for n, X in enumerate(Xi[r])}
            results.update(self.opt_empty_tuple())
            backend.results_shared = backend.manager.dict(results)
        else:
            # if we have more than 1 optimizer but are using single opt,
            # pick one discard the rest...
            if n_opts > 0:
                self.opt = opts[-1]
            else:
                self.opt = self.get_optimizer()
                self.opt.void_loss = VOID_LOSS
                self.opt.void = False
                self.opt.rs = self.random_state
            # in single mode restore the points directly to the optimizer
            # but delete first in case we have filtered the starting list of points
            self.opt = HyperoptData.opt_clear(self.opt)
            rs = self.random_state
            self.Xi[rs] = []
            self.track_points()
            if len(self.Xi[rs]) > 0:
                self.opt.tell(self.Xi[rs], self.yi[rs], fit=False)
            # delete points since in single mode the optimizer state sits in the main
            # process and is not discarded
            self.Xi, self.yi = {}, {}
        del opts[:]

    @staticmethod
    def opt_rand(opt: Optimizer, rand: int = None, seed: bool = True) -> Optimizer:
        """ return a new instance of the optimizer with modified rng """
        if seed:
            if not rand:
                rand = opt.rng.randint(0, VOID_LOSS)
            opt.rng.seed(rand)
        opt, opt.void_loss, opt.void, opt.rs = (
            opt.copy(random_state=opt.rng),
            opt.void_loss,
            opt.void,
            opt.rs,
        )
        return opt

    @staticmethod
    def opt_clear(opt: Optimizer):
        """ clear state from an optimizer object """
        del opt.models[:], opt.Xi[:], opt.yi[:]
        return opt
