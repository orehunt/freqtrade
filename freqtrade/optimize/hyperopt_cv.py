import warnings
from abc import abstractmethod

from joblib import Parallel, delayed, dump
from pandas import DataFrame

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.optimize.hyperopt_data import HyperoptData
from freqtrade.optimize.hyperopt_out import HyperoptOut
import freqtrade.optimize.hyperopt_backend as backend
from freqtrade.optimize.hyperopt_backend import TrialsState, Epochs

# Suppress scikit-learn FutureWarnings from skopt
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
# Additional regressors already pluggable into the optimizer
# from sklearn.linear_model import ARDRegression, BayesianRidge
# possibly interesting regressors that need predict method override
# from sklearn.ensemble import HistGradientBoostingRegressor
# from xgboost import XGBoostRegressor


class HyperoptCV:
    """ cross validation """

    target_trials: DataFrame
    trials_maxout: int
    trials_timeout: float


    @abstractmethod
    def parallel_objective(self, t: int, params, epochs: Epochs, trials_state: TrialsState):
        """ objective run in single opt mode, run the backtest and log the results """

    @abstractmethod
    def parallel_objective_sig_handler(
        self, t: int, params: list, epochs: Epochs, trials_state: TrialsState
    ):
        """ objective run in single opt mode, run the backtest and log the results """

    def trials_params(self, offset: int, jobs: int):
        # use the right names for dimensions
        if not backend.trials.exit:
            self.target_trials, params_cols = HyperoptData.alias_cols(
                self.target_trials, "params_dict"
            )
            # dump the parameters values to FS
            params = self.target_trials.loc[:, params_cols].to_dict("records")
            epochs = len(params)
            Xi = [list(p.values()) for p in params]
            n_dims = len(self.dimensions)
            dump(Xi, self.Xi_file)

            # params_Xi = np.memmap(Xi_file, dtype='float64', mode='r', shape=(epochs,n_dims))
            for t, X in enumerate(Xi[offset:]):
                HyperoptOut._print_progress(t, jobs, self.trials_maxout)
                yield t, []
        else:
            # loop over jobs to schedule the last dispatch to collect unsaved epochs
            for j in range(2 * jobs):
                HyperoptOut._print_progress(j, jobs, self.trials_maxout)
                yield j, []

    def run_cv_backtest_parallel(self, parallel: Parallel, jobs: int):
        """ evaluate a list of given parameters in parallel """
        parallel(
            delayed(self.parallel_objective_sig_handler)(
                t, params, backend.epochs, backend.trials, self.cls_file
            )
            for t, params in self.trials_params(0, jobs)
        )
