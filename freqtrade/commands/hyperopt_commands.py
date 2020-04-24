import logging
from typing import Any, Dict

from colorama import init as colorama_init

from freqtrade.configuration import setup_utils_configuration
from freqtrade.exceptions import OperationalException
from freqtrade.state import RunMode

logger = logging.getLogger(__name__)


def start_hyperopt_list(args: Dict[str, Any]) -> None:
    """
    List hyperopt epochs previously evaluated
    """
    from freqtrade.optimize.hyperopt_out import HyperoptOut

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_colorized = config.get("print_colorized", False)
    print_json = config.get("print_json", False)
    export_csv = config.get("export_csv", None)
    no_details = config.get("hyperopt_list_no_details", False)
    no_header = False

    ho = HyperoptOut(config)

    filters = ho._filter_options(config)

    trials_file = config.get("hyperopt_list_trials_file", ho.get_trials_file(config, ho.trials_dir))
    trials_instances_file = config.get(
        "hyperopt_list_trials_instances_file", ho.trials_instances_file
    )

    # Previous evaluations
    trials = ho.load_trials(
        trials_file,
        ho.get_last_instance(trials_instances_file, config.get("hyperopt_trials_instance")),
    )
    total_epochs = len(trials)

    trials = ho.filter_trials(trials, config).copy()
    n_trials = len(trials)

    if print_colorized:
        colorama_init(autoreset=True)

    if not export_csv:
        try:
            print(
                ho.get_result_table(
                    config, trials, total_epochs, not filters["best"], print_colorized, 0
                )
            )
        except KeyboardInterrupt:
            print("User interrupted..")

    if n_trials and not no_details:
        best = trials.sort_values("loss").iloc[0:]
        ho.print_epoch_details(ho.trials_to_dict(best)[0], total_epochs, print_json, no_header)

    if n_trials and export_csv:
        ho.export_csv_file(config, trials, total_epochs, not filters["best"], export_csv)


def start_hyperopt_show(args: Dict[str, Any]) -> None:
    """
    Show details of a hyperopt epoch previously evaluated
    """
    from freqtrade.optimize.hyperopt import HyperoptOut
    from freqtrade.optimize.hyperopt_data import HyperoptData as hd

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    print_json = config.get("print_json", False)
    no_header = config.get("hyperopt_show_no_header", False)

    ho = HyperoptOut(config)

    trials_file = config.get("hyperopt_list_trials_file", ho.get_trials_file(config, ho.trials_dir))
    trials_instances_file = config.get(
        "hyperopt_list_trials_instances_file", ho.trials_instances_file
    )
    n = config.get("hyperopt_show_index", -1)

    # Previous evaluations
    trials = hd.load_trials(trials_file)
    i = 1
    for c, t in enumerate(trials):
        if t["current_epoch"] == trials[c - 1]["current_epoch"]:
            print(t["current_epoch"], i, trials[c - 1]["current_epoch"])
        i += 1
    total_epochs = len(trials)

    trials = hd.filter_trials(trials, config)
    trials_epochs = len(trials)

    if n > trials_epochs:
        raise OperationalException(
            f"The index of the epoch to show should be less than {trials_epochs + 1}."
        )
    if n < -trials_epochs:
        raise OperationalException(
            f"The index of the epoch to show should be greater than {-trials_epochs - 1}."
        )

    # Translate epoch index from human-readable format to pythonic
    if n > 0:
        n -= 1

    if trials:
        val = trials[n]
        HyperoptOut.print_epoch_details(
            val, total_epochs, print_json, no_header, header_str="Epoch details"
        )
