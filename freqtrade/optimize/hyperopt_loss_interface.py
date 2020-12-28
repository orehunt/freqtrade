"""
IHyperOptLoss interface
This module defines the interface for the loss-function for hyperopt
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from pandas import DataFrame

from freqtrade.optimize.vbt import BacktestResultTupleType


Objective = Dict[str, float]
ObjectiveTuple = Tuple[Tuple[str, float], ...]
jittedLoss = Callable[[List[BacktestResultTupleType]], ObjectiveTuple]


class IHyperOptLoss(ABC):
    """
    Interface for freqtrade hyperopt Loss functions.
    Defines the custom loss function (`hyperopt_loss_function()` which is evaluated every epoch.)
    """

    timeframe: str
    metrics: List[str]

    @abstractmethod
    def hyperopt_loss_function(
        self, results: DataFrame, results_metrics: Dict[str, Any], *args, **kwargs
    ) -> Objective:
        """
        Objective function, returns smaller number for better results
        """

    @abstractmethod
    def hyperopt_loss_function_nb(self) -> jittedLoss:
        """
        Called to generate the jitted loss function
        """
