import locale
import warnings
import logging
import sys
import json
import io
from pprint import pformat
from typing import Any, Dict, Optional, Union
from pathlib import Path

import rapidjson
from colorama import Fore, Style
from pandas import DataFrame, isna, json_normalize
import tabulate
import enlighten
import numba as nb
import numpy as np

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
        trial,
        total_epochs: int,
        print_json: bool,
        no_header: bool = False,
        header_str: str = None,
    ) -> None:
        """
        Display details of the hyperopt result
        """
        params = trial.get("params_details", {})

        # Default header string
        if header_str is None:
            header_str = "Best result"

        if not no_header:
            explanation_str = HyperoptOut._format_explanation_string(
                trial, total_epochs
            )
            print(f"\n{header_str}:\n\n{explanation_str}\n")

        if print_json:
            result_dict: Dict = {}
            if "roi" in trial:
                result_dict["minimal_roi"] = json.loads(trial["roi"])
            for s in ["buy", "sell", "stoploss", "trailing"]:
                HyperoptData._params_update_for_json(result_dict, params, s)
            print(
                rapidjson.dumps(
                    result_dict,
                    default=str,
                    number_mode=(rapidjson.NM_NATIVE | rapidjson.NM_NAN),
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

    def _params_pretty_print(params, space: str, header: str) -> None:
        if space in params:
            space_params = HyperoptOut._space_params(params, space, 5)
            print(f"\n    # {header}")
            if space == "stoploss":
                print("    stoploss =", space_params.get("stoploss"))
            else:
                params_result = pformat(space_params, indent=4).replace("}", "\n}")
                params_result = params_result.replace("{", "{\n ").replace(
                    "\n", "\n    "
                )
                print(f"    {space}_params = {params_result}")

    @staticmethod
    def _space_params(params, space: str, r: int = None) -> Dict:
        d = params[space]
        # Round floats to `r` digits after the decimal point if requested
        return round_dict(d, r) if r else d

    @staticmethod
    def limit_num_len(num: Union[float, int], fnum: str, limit=16, decimals=8):
        return fnum.format(num) if len(f"{num:.{decimals}f}") < limit else f"{num:.2e}"

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
                self.epochs_limit,
                self.print_all,
                self.print_colorized,
                table_header,
            )
        )

    @staticmethod
    def _format_explanation_string(results, total_epochs) -> str:
        return (
            (
                "*"
                if "is_initial_point" in results and results["is_initial_point"]
                else " "
            )
            + f"{results['current_epoch']:5d}/{total_epochs}: "
            + f"{results['results_explanation']} "
        )

    @staticmethod
    @nb.jit(cache=True)
    def format_results_nb(arr, col):
        if col == "Epoch":
            for n, e in enumerate(arr):
                arr[n] = (
                    limit_num_len(e, "{:,.2f}%").rjust(7, " ")
                    if not np.isnan(e)
                    else "--".rjust(7, " ")
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
        # pd.set_option('max_columns', 9)

        trials["Best"] = ""
        trials = trials.loc[
            :,
            [
                "Best",
                "current_epoch",
                "trade_count_mid",
                "trade_ratio_mid",
                "win_ratio_mid",
                "avg_profit_mid",
                "med_profit_mid",
                "total_profit_mid",
                "trade_duration_mid",
                "is_initial_point",
                "is_best",
                "loss",
            ],
        ]
        trials.columns = [
            "B",
            "Epoch",
            "Trades",
            "Returns R.",
            "Win R.",
            "Avg. P.",
            "Med. P.",
            "Tot. P.",
            "Avg. Dur.",
            "is_initial_point",
            "is_best",
            "loss",
        ]
        trials["is_profit"] = False
        trials.loc[trials["is_initial_point"], "B"] = "*"
        trials.loc[trials["is_best"], "B"] = "!"
        trials.loc[trials["Tot. P."].values > 0, "is_profit"] = True
        trials["Trades"] = trials["Trades"].astype(str)

        is_best = trials["is_best"].values
        is_profit = trials["is_profit"].values

        w = 10
        l = HyperoptOut.limit_num_len
        c = config["stake_currency"]
        # currency width
        c_w = len(c)
        # profit width
        p_w = w + c_w + 2
        # duration width
        d_w = 5
        # ratio width
        r_w = 8

        trials["Epoch"] = trials["Epoch"].apply(
            lambda x: "{}/{}".format(
                # for a maximum of 99999/99999 epochs
                str(x).rjust(11, " "),
                total_epochs,
            )
        )
        trials["Avg. P."] = trials["Avg. P."].apply(
            lambda x: l(x, "{:,.5g} {}".format(x, c)).rjust(p_w, " ")
            if not isna(x)
            else "--".rjust(p_w, " ")
        )
        trials["Med. P."] = trials["Med. P."].apply(
            lambda x: l(x, "{:,.5g} {}".format(x, c)).rjust(p_w, " ")
            if not isna(x)
            else "--".rjust(p_w, " ")
        )
        trials["Tot. P."] = trials.apply(
            lambda x: "{} {}".format(
                l(x["Tot. P."], "{:.5g}"),
                c,
            ).rjust(p_w, " ")
            if not isna(x["Tot. P."])
            else "--".rjust(p_w, " "),
            axis=1,
        )
        trials["Avg. Dur."] = trials["Avg. Dur."].apply(
            lambda x: l(x, "{:,.1f} m").rjust(d_w, " ")
            if not isna(x)
            else "--".rjust(d_w, " ")
        )
        trials["Win R."] = trials["Win R."].apply(
            lambda x: l(x, "{:.4g}").rjust(r_w, " ")
            if not isna(x)
            else "--".rjust(r_w, " ")
        )
        trials["Returns R."] = trials["Returns R."].apply(
            lambda x: l(x, "{:.4g}").rjust(r_w, " ")
            if not isna(x)
            else "--".rjust(r_w, " ")
        )

        trials.drop(columns=["loss"], inplace=True)
        for col in trials.columns.difference(
            ["is_profit", "is_best", "is_initial_point"]
        ):
            trials[col] = trials[col].astype(str)

        n_cols = len(trials.columns)
        if print_colorized:
            for i in range(len(trials)):
                if is_profit[i]:
                    for j in range(n_cols):
                        trials.iat[i, j] = "{}{}{}".format(
                            Fore.GREEN, trials.iat[i, j], Fore.RESET
                        )
                if is_best[i] and highlight_best:
                    for j in range(n_cols - 3):
                        trials.iat[i, j] = "{}{}{}".format(
                            Style.BRIGHT, trials.iat[i, j], Style.RESET_ALL
                        )

        trials.drop(columns=["is_initial_point", "is_best", "is_profit"], inplace=True)
        if remove_header > 0:
            table = tabulate.tabulate(
                trials.to_dict(orient="list"),
                tablefmt="orgtbl",
                headers="keys",
                stralign="right",
            )

            table = table.split("\n", remove_header)[remove_header]
        elif remove_header < 0:
            table = tabulate.tabulate(
                trials.to_dict(orient="list"),
                tablefmt="psql",
                headers="keys",
                stralign="right",
            )
            table = "\n".join(table.split("\n")[0:remove_header])
        else:
            table = tabulate.tabulate(
                trials.to_dict(orient="list"),
                tablefmt="psql",
                headers="keys",
                stralign="right",
            )
        return table

    @staticmethod
    def _format_results_explanation_string(stake_cur: str, r: Dict) -> str:
        """
        Return the formatted results explanation in a string
        """
        return (
            (
                f"trades: {r['trade_count_mid']} | "
                f"profits ({stake_cur}): (tot) {r['total_profit_mid']:.2f} "
                f"(rti) {r['trade_ratio_mid']:.2f} "
                f"(med) {r['med_profit_mid']:.2f} | "
                f"duration: {r['trade_duration_mid']:.1f}"
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
                "Imp: {Style.BRIGHT}{Fore.WHITE}{backend.epochs.improvement:.3} "
                "{Style.RESET_ALL}"
                "Awt: {Style.BRIGHT}{Fore.BLUE}{backend.epochs.avg_wait_time:.1f} "
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
            "{Style.BRIGHT}{backend.epochs.last_best_epoch} "
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
            min_delta=0.5,
        )

    @staticmethod
    def _print_progress(t: int, jobs: int, maxout: int, finish=False):
        try:
            backend.pbar["total"].update(
                backend.trials.num_saved - backend.pbar["total"].count
            )
            if finish:
                backend.pbar["total"].close()
        # for the enlighten manager
        except:
            logger.debug("failed to print progress, probably just forcefully killed.")
