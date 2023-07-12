import argparse
import json
import logging
import os
import signal
import sys
from types import FrameType
from typing import Optional

import hojichar
from hojichar.utils.io_iter import fileout_from_iter, stdin_iter, stdout_from_iter
from hojichar.utils.load_compose import load_compose
from hojichar.utils.process import process_iter, reject_iter

FILTER: hojichar.Compose
logger = logging.getLogger("hojichar.__main__")


def finalize() -> None:
    print(json.dumps(FILTER.statistics), file=sys.stderr, end="")


# Typing of signal handler: https://github.com/python/typing/discussions/1042
def sigint_handler(signum: int, frame: Optional[FrameType]) -> None:
    print(file=sys.stderr)
    finalize()
    sys.exit(0)


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        "-p",
        required=True,
        metavar="<profile.py>",
        help="Path to a Python file that implements your custom filter.",
    )
    parser.add_argument(
        "--args",
        default=[],
        nargs="+",
        help="Pass additional arguments to the profile.\
            Use it like `--args arg1 arg2` etc. The arguments should be space-separated.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Specifies the path for the output file. Defaults to standard output.",
    )
    parser.add_argument(
        "--dump-stats",
        default=None,
        metavar="<path to stats.json>",
        help="Dump statistics to file. If the file exists, it will be appended.",
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="Exit if an exception occurs during filtering.\
            Useful for debugging custom filters.",
    )
    parser.add_argument(
        "--redirect-stdout",
        default=None,
        help="This option is used to redirect standard output to a specified file during the \
        profile. By default, it redirects to /dev/null.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="A flag that specifies whether to include discarded samples. \
            This is useful when inspecting discarded samples.",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    global FILTER
    signal.signal(signal.SIGINT, sigint_handler)
    args = argparser()
    if args.redirect_stdout is None:
        stdout_fp = open(os.devnull, "w")
    else:
        stdout_fp = open(args.redirect_stdout, "w")

    FILTER = load_compose(
        args.profile,
        *tuple(args.args),
    )
    input_iter = stdin_iter()
    out_doc_iter = process_iter(
        input_iter=input_iter, filter=FILTER, exit_on_error=args.exit_on_error, stdout_fp=stdout_fp
    )
    out_str_iter = reject_iter(input_iter=out_doc_iter, discard_rejected=not args.all)
    if args.output:
        with open(args.output, "w") as fp:
            fileout_from_iter(out_str_iter, fp)
    else:
        stdout_from_iter(out_str_iter)
    finalize()
    if args.dump_stats:
        with open(args.dump_stats, "a") as fp:
            fp.write(json.dumps(FILTER.statistics) + "\n")


if __name__ == "__main__":
    main()
