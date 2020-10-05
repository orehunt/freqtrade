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


class HyperoptCV(HyperoptOut):
    """ cross validation """

    target_trials: DataFrame
    trials_maxout: int
    trials_timeout: float
    use_progressbar = True

    def trials_params(self, offset: int, jobs: int):
        # use the right names for dimensions
        if not backend.trials.exit:
            # dump the parameters values to FS
            params = self.target_trials["params_dict"].values.tolist()
            epochs = len(params)
            # n_dims = len(self.dimensions)
            dump(params, self.Xi_file)
            # not needed anymore
            del params

            # params_Xi = np.memmap(Xi_file, dtype='float64', mode='r', shape=(epochs,n_dims))
            for t in range(offset, epochs):
                if self.use_progressbar:
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
            delayed(backend.parallel_sig_handler)(
                self.parallel_objective,
                self.cls_file,
                self.logger,
                t,
                params,
                epochs=backend.epochs,
                trials_state=backend.trials,
            )
            for t, params in self.trials_params(0, jobs)
        )
