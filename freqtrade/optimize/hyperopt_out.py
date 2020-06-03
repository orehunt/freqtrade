import locale
import warnings
import logging
import sys
import json
import io
from pprint import pprint
from typing import Any, Dict, Optional
from pathlib import Path

import rapidjson
from colorama import Fore, Style
from pandas import DataFrame, isna, json_normalize
import tabulate
import enlighten

from freqtrade.misc import round_dict

# Import IHyperOpt and IHyperOptLoss to allow unpickling classes from these modules
from freqtrade.optimize.hyperopt_backend import Epochs
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

logger = logging.getLogger(__name__)


class HyperoptOut(HyperoptData):
    """ Output routines for Hyperopt """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # print options
        self.print_all = self.config.get("print_all", False)
        self.print_colorized = self.config.get("print_colorized", False)
        self.print_json = self.config.get("print_json", False)

    @staticmethod
    def print_epoch_details(
        trial, total_epochs: int, print_json: bool, no_header: bool = False, header_str: str = None
    ) -> None:
        """
        Display details of the hyperopt result
        """
        params = trial.get("params_details", {})

        # Default header string
        if header_str is None:
            header_str = "Best result"

        if not no_header:
            explanation_str = HyperoptOut._format_explanation_string(trial, total_epochs)
            print(f"\n{header_str}:\n\n{explanation_str}\n")

        if print_json:
            result_dict: Dict = {}
            if "roi" in trial:
                result_dict["minimal_roi"] = json.loads(trial["roi"])
            for s in ["buy", "sell", "stoploss", "trailing"]:
                HyperoptData._params_update_for_json(result_dict, params, s)
            print(
                rapidjson.dumps(
                    result_dict, default=str, number_mode=(rapidjson.NM_NATIVE | rapidjson.NM_NAN)
                )
            )

        else:
            HyperoptOut._params_pretty_print(params, "buy", "Buy hyperspace params:")
            HyperoptOut._params_pretty_print(params, "sell", "Sell hyperspace params:")
            HyperoptOut._params_pretty_print(params, "roi", "ROI table:")
            HyperoptOut._params_pretty_print(params, "stoploss", "Stoploss:")
            HyperoptOut._params_pretty_print(params, "trailing", "Trailing stop:")

    @staticmethod
    def clear_line(columns: int):
        print("\r", " " * columns, end="\r")

    @staticmethod
    def reset_line():
        print(end="\r")

    @staticmethod
    def _params_pretty_print(params, space: str, header: str) -> None:
        if space in params:
            space_params = HyperoptOut._space_params(params, space, 5)
            if space == "stoploss":
                print(header, space_params.get("stoploss"))
            else:
                print(header)
                pprint(space_params, indent=4)

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params[space]
        # Round floats to `r` digits after the decimal point if requested
        return round_dict(d, r) if r else d

    def print_results(self, trials, table_header: int, epochs=None) -> None:
        """
        Log results if it is better than any previous evaluation
        """
        if not self.print_all:
            trials = trials.loc[trials["is_best"]]
        # needed by epochs_limit because max epochs reference the global reference to epochs
        # but the workers access the proxied namespace directly
        if epochs:
            backend.epochs = epochs
        print(
            self.get_result_table(
                self.config,
                trials,
                self.epochs_limit(),
                self.print_all,
                self.print_colorized,
                table_header,
            )
        )

    @staticmethod
    def print_results_explanation(
        results, total_epochs, highlight_best: bool, print_colorized: bool
    ) -> None:
        """
        Log results explanation string
        """
        explanation_str = HyperoptOut._format_explanation_string(results, total_epochs)
        # Colorize output
        if print_colorized:
            if results["total_profit"] > 0:
                explanation_str = Fore.GREEN + explanation_str
            if highlight_best and results["is_best"]:
                explanation_str = Style.BRIGHT + explanation_str
        print(explanation_str)

    @staticmethod
    def _format_explanation_string(results, total_epochs) -> str:
        return (
            ("*" if "is_initial_point" in results and results["is_initial_point"] else " ")
            + f"{results['current_epoch']:5d}/{total_epochs}: "
            + f"{results['results_explanation']} "
            + f"Objective: {results['loss']:.5f}"
        )

    @staticmethod
    def get_result_table(
        config: dict,
        trials: DataFrame,
        total_epochs: int,
        highlight_best: bool,
        print_colorized: bool,
        remove_header: int,
    ) -> str:
        """
        Log result table
        """
        if len(trials) < 1:
            return ""

        tabulate.PRESERVE_WHITESPACE = True

        logger.debug("Formatting results...")

        trials["Best"] = ""
        trials = trials.loc[
            :,
            [
                "Best",
                "current_epoch",
                "results_metrics.trade_count",
                "results_metrics.avg_profit",
                "results_metrics.total_profit",
                "results_metrics.profit",
                "results_metrics.duration",
                "loss",
                "is_initial_point",
                "is_best",
            ],
        ]
        trials.columns = [
            "Best",
            "Epoch",
            "Trades",
            "Avg profit",
            "Total profit",
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
        trials["Trades"] = trials["Trades"].astype(str)

        trials["Epoch"] = trials["Epoch"].apply(
            lambda x: "{}/{}".format(str(x).rjust(len(str(total_epochs)), " "), total_epochs)
        )
        trials["Avg profit"] = trials["Avg profit"].apply(
            lambda x: "{:,.2f}%".format(x).rjust(7, " ") if not isna(x) else "--".rjust(7, " ")
        )
        trials["Avg duration"] = trials["Avg duration"].apply(
            lambda x: "{:,.1f} m".format(x).rjust(7, " ") if not isna(x) else "--".rjust(7, " ")
        )
        trials["Objective"] = trials["Objective"].apply(
            lambda x: "{:,.5f}".format(x).rjust(8, " ") if x != 100000 else "N/A".rjust(8, " ")
        )

        trials["Profit"] = trials.apply(
            lambda x: "{:,.8f} {} {}".format(
                x["Total profit"],
                config["stake_currency"],
                "({:,.2f}%)".format(x["Profit"]).rjust(10, " "),
            ).rjust(25 + len(config["stake_currency"]))
            if x["Total profit"] != 0.0
            else "--".rjust(25 + len(config["stake_currency"])),
            axis=1,
        )
        trials = trials.drop(columns=["Total profit"])

        n_cols = len(trials.columns)
        n_trials = len(trials)
        if print_colorized:
            for i in range(n_trials):
                if trials["is_profit"].iat[i]:
                    for j in range(n_cols - 3):
                        trials.iat[i, j] = "{}{}{}".format(
                            Fore.GREEN, str(trials.iat[i, j]), Fore.RESET
                        )
                if trials["is_best"].iat[i] and highlight_best:
                    for j in range(n_cols - 3):
                        trials.iat[i, j] = "{}{}{}".format(
                            Style.BRIGHT, str(trials.iat[i, j]), Style.RESET_ALL
                        )

        trials = trials.drop(columns=["is_initial_point", "is_best", "is_profit"])
        if remove_header > 0:
            table = tabulate.tabulate(
                trials.to_dict(orient="list"), tablefmt="orgtbl", headers="keys", stralign="right"
            )

            table = table.split("\n", remove_header)[remove_header]
        elif remove_header < 0:
            table = tabulate.tabulate(
                trials.to_dict(orient="list"), tablefmt="psql", headers="keys", stralign="right"
            )
            table = "\n".join(table.split("\n")[0:remove_header])
        else:
            table = tabulate.tabulate(
                trials.to_dict(orient="list"), tablefmt="psql", headers="keys", stralign="right"
            )
        return table

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
        path = Path(csv_file)
        if path.isfile(csv_file):
            logger.error(f"CSV file already exists: {csv_file}")
            return

        try:
            io.open(csv_file, "w+").close()
        except IOError:
            logger.error(f"Failed to create CSV file: {csv_file}")
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
        trials.loc[trials["is_initial_point"] & trials["is_best"], "Best"] = "* Best"
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
        logger.info(f"CSV file created: {csv_file}")

    @staticmethod
    def _format_results_explanation_string(stake_cur: str, results_metrics: Dict) -> str:
        """
        Return the formatted results explanation in a string
        """
        return (
            (
                f"{results_metrics['trade_count']:6d} trades. "
                f"Avg profit {results_metrics['avg_profit']: 6.2f}%. "
                f"Total profit {results_metrics['total_profit']: 11.8f} {stake_cur} "
                f"({results_metrics['profit']: 7.2f}\N{GREEK CAPITAL LETTER SIGMA}%). "
                f"Avg duration {results_metrics['duration']:5.1f} min."
            )
            .encode(locale.getpreferredencoding(), "replace")
            .decode("utf-8")
        )

    @staticmethod
    def log_results_immediate(n, epochs: Epochs) -> None:
        """ Signals that a new job has been scheduled"""
        # get lock to avoid messing up the console
        if epochs.lock.acquire(False):
            print(".", end="")
            sys.stdout.flush()
            epochs.lock.release()

    @staticmethod
    def _init_progressbar(print_colorized: bool, total_epochs: Optional[int], cv=False):
        global logger
        if backend.pbar:
            backend.pbar["total"].close()
        color = "green"
        opt_format = (
            (
                "Avg: {Style.BRIGHT}{Fore.BLUE}{backend.epochs.current_best_epoch}"
                "+{backend.epochs.avg_last_occurrence} "
                "{Style.RESET_ALL}"
                "Exp: {Style.BRIGHT}{Fore.CYAN}{backend.epochs.explo} "
                "{Style.RESET_ALL}"
                "Cvg: {Style.BRIGHT}{Fore.MAGENTA}{backend.epochs.convergence} "
                "{Style.RESET_ALL}"
            )
            if not cv
            else ""
        )
        # TODO:
        # - Buf reports negative values at the end when it's flushing the remaining trials
        # - Exp goes below(above) the actual number of workers after sometime (gc related?)
        # - max/minimizing tmux panes bugs the pbar once in a while requiring a tmux session restart
        # or causes a very rare exception with a resize lock held by the enlighten manager
        trials_format = (
            "Best: "
            "{Style.BRIGHT}{backend.epochs.current_best_epoch}"
            "[{backend.epochs.current_best_loss:.02}] "
            "{Style.RESET_ALL}"
            f"{opt_format}"
            "Buf: {Style.BRIGHT}{Fore.YELLOW}{backend.trials.num_done} "
            "{Style.RESET_ALL}"
            "Empty: {Style.BRIGHT}{Fore.RED}{backend.trials.empty_strikes} "
            "{Style.RESET_ALL}["
        )
        bar_format = (
            f"{trials_format}"
            "{desc_pad}{percentage:3.0f}%|{bar}| "
            "{count:{len_total}d}/{total:d} "
            "[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]"
        )
        counter_format = (
            f"{trials_format}"
            "{desc_pad}{count:d} {unit}{unit_pad}{elapsed},"
            " {rate:.2f}{unit_pad}{unit}/s]{fill}"
        )
        backend.pbar["manager"] = enlighten.Manager()
        backend.pbar["total"] = backend.pbar["manager"].counter(
            count=backend.trials.num_saved,
            total=total_epochs,
            unit="epochs",
            bar_format=bar_format,
            counter_format=counter_format,
            color=color,
            additional_fields={"Style": Style, "Fore": Fore, "backend": backend},
            min_delta=0.5
        )

    @staticmethod
    def _print_progress(t: int, jobs: int, maxout: int, finish=False):
        backend.pbar["total"].update(backend.trials.num_saved - backend.pbar["total"].count)
        if finish:
            backend.pbar["total"].close()
