import warnings
from typing import List
from abc import abstractmethod

from joblib import Parallel, delayed, wrap_non_picklable_objects

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
import freqtrade.optimize.hyperopt_backend as backend

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

    target_trials: List

    @abstractmethod
    def parallel_objective(self, asked, results_list: List = [], n=0):
        """ objective run in single opt mode, run the backtest, store the results into a queue """

    def trials_params(self, offset: int):
        for t in self.target_trials[offset:]:
            yield t["params_dict"]

    def run_cv_backtest_parallel(self, parallel: Parallel, tries: int, first_try: int, jobs: int):
        """ evaluate a list of given parameters in parallel """
        parallel(
            delayed(wrap_non_picklable_objects(self.parallel_objective))(
                params, backend.results_list, n=i
            )
            for params, i in zip(self.trials_params(first_try), range(first_try, first_try + tries))
        )
