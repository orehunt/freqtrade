import warnings
from typing import List
from abc import abstractmethod

from joblib import Parallel, delayed, wrap_non_picklable_objects

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
# from freqtrade.optimize.hyperopt_backend import Trial
from freqtrade.optimize.hyperopt_interface import IHyperOpt  # noqa: F401
from freqtrade.optimize.hyperopt_loss_interface import IHyperOptLoss  # noqa: F401
from freqtrade.optimize.hyperopt_data import HyperoptData
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
    def parallel_objective(self, asked, trials_list: List = [], n=0):
        """ objective run in single opt mode, run the backtest, store the trials into a queue """

    def trials_params(self, offset: int, jobs: int):
        # use the right names for dimensions
        self.target_trials, params_cols = HyperoptData.alias_cols(self.target_trials, "params_dict")
        Xi = self.target_trials.loc[:, params_cols].to_dict("records")
        for X in Xi[offset:]:
            yield X
        # loop over jobs to schedule the last dispatch to collect unsaved epochs
        for j in range(jobs):
            yield []

    def run_cv_backtest_parallel(self, parallel: Parallel, tries: int, first_try: int, jobs: int):
        """ evaluate a list of given parameters in parallel """
        parallel(
            delayed(wrap_non_picklable_objects(self.parallel_objective))(
                t, params, backend.epochs, backend.trials
            )
            for params, t in zip(self.trials_params(first_try, jobs), range(first_try, first_try + tries))
        )
