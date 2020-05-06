import logging
import sys

from logging import Formatter
from logging.handlers import RotatingFileHandler, SysLogHandler
from typing import Any, Dict, List
from colorama import Fore, Style, init

from freqtrade.exceptions import OperationalException


logger = logging.getLogger(__name__)


def _set_loggers(verbosity: int = 0) -> None:
    """
    Set the logging level for third party libraries
    :return: None
    """

    logging.getLogger("requests").setLevel(logging.INFO if verbosity <= 1 else logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO if verbosity <= 1 else logging.DEBUG)
    logging.getLogger("ccxt.base.exchange").setLevel(
        logging.INFO if verbosity <= 2 else logging.DEBUG
    )
    logging.getLogger("telegram").setLevel(logging.INFO)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Process -v/--verbose, --logfile options
    """
    # Log level
    verbosity = config["verbosity"]

    # Log to stderr
    log_handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    logfile = config.get("logfile")
    if logfile:
        s = logfile.split(":")
        if s[0] == "syslog":
            # Address can be either a string (socket filename) for Unix domain socket or
            # a tuple (hostname, port) for UDP socket.
            # Address can be omitted (i.e. simple 'syslog' used as the value of
            # config['logfilename']), which defaults to '/dev/log', applicable for most
            # of the systems.
            address = (s[1], int(s[2])) if len(s) > 2 else s[1] if len(s) > 1 else "/dev/log"
            handler = SysLogHandler(address=address)
            # No datetime field for logging into syslog, to allow syslog
            # to perform reduction of repeating messages if this is set in the
            # syslog config. The messages should be equal for this.
            handler.setFormatter(Formatter("%(name)s - %(levelname)s - %(message)s"))
            log_handlers.append(handler)
        elif s[0] == "journald":
            try:
                from systemd.journal import JournaldLogHandler
            except ImportError:
                raise OperationalException(
                    "You need the systemd python package be installed in "
                    "order to use logging to journald."
                )
            handler = JournaldLogHandler()
            # No datetime field for logging into journald, to allow syslog
            # to perform reduction of repeating messages if this is set in the
            # syslog config. The messages should be equal for this.
            handler.setFormatter(Formatter("%(name)s - %(levelname)s - %(message)s"))
            log_handlers.append(handler)
        else:
            log_handlers.append(
                RotatingFileHandler(logfile, maxBytes=1024 * 1024, backupCount=10)  # 1Mb
            )

    logging.basicConfig(
        level=logging.INFO if verbosity < 1 else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=log_handlers,
    )
    _set_loggers(verbosity)
    logger.info("Verbosity set to %s", verbosity)
    if config.get("print_colorized"):
        init(autoreset=True)
        logging.addLevelName(
            logging.WARNING,
            Fore.LIGHTYELLOW_EX
            + Style.BRIGHT
            + logging.getLevelName(logging.WARNING)
            + Style.RESET_ALL,
        )
        logging.addLevelName(
            logging.DEBUG,
            Fore.LIGHTMAGENTA_EX
            + Style.BRIGHT
            + logging.getLevelName(logging.DEBUG)
            + Style.RESET_ALL,
        )
        logging.addLevelName(
            logging.ERROR,
            Fore.LIGHTRED_EX + Style.BRIGHT + logging.getLevelName(logging.ERROR) + Style.RESET_ALL,
        )
        logging.addLevelName(
            logging.INFO,
            Fore.LIGHTWHITE_EX
            + Style.BRIGHT
            + logging.getLevelName(logging.INFO)
            + Style.RESET_ALL,
        )
