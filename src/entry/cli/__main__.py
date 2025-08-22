import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv

from config import GlobalConfig

load_dotenv()

from rich.logging import RichHandler
from rich.traceback import install

dev_mode = os.getenv("AIC25_DEV", "false").lower() == "true"

FORMAT = "%(message)s"
DATE_FORMAT = "[%X]"
logging.basicConfig(
    level=logging.DEBUG if dev_mode else logging.INFO,
    format=FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

install(show_locals=dev_mode)

from entry.cli import commands


def main():
    work_dir = Path.cwd() / "aic25_workspace"
    GlobalConfig.initialize(work_dir)
    parser = ArgumentParser(
        description="Command Line Interface of Team Past Beggar of AIC25."
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
    )
    parser.add_argument(
        "-w",
        "--workspace-dir",
        dest="work_dir",
        type=Path,
        default=work_dir,
        help="Path to workspace",
    )
    subparser = parser.add_subparsers(help="command", dest="command", required=True)

    for command_cls in commands.available_commands:
        command = command_cls(work_dir)
        command.add_args(subparser)

    args = parser.parse_args()
    args_dict = vars(args)
    command = args_dict.pop("command")

    actual_work_dir = args_dict.get("work_dir", work_dir)

    if "func" not in args_dict:
        parser.print_help()
        return

    logger.info(f"Start running command: {command}")
    logger.debug(f"Working directory: {actual_work_dir}")

    func = args_dict.pop("func")

    if not args_dict.get("verbose"):
        logging.disable(logging.CRITICAL)

    if "work_dir" in args_dict:
        args_dict["work_dir"] = actual_work_dir

    func(**args_dict)


if __name__ == "__main__":
    main()
