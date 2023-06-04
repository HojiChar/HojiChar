import argparse
import json
import logging
import signal
import sys

import hojichar
from hojichar.utils.io_iter import fileout_from_iter, stdin_iter, stdout_from_iter
from hojichar.utils.load_compose import load_compose
from hojichar.utils.process import process_iter

FILTER: hojichar.Compose
logger = logging.getLogger("hojichar.__main__")


def finalize() -> None:
    print(json.dumps(FILTER.statistics), file=sys.stderr, end="")


def sigint_handler(signum, frame) -> None:
    print(file=sys.stderr)
    finalize()
    sys.exit(0)


def argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        "-p",
        required=True,
        metavar="<your_filter.py>",
        help="Path to a Python file that implements your custom filter.\
            hojichar.Compose must be defined as FILTER variable in the file.",
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output file path. If not given, stdout is used."
    )
    parser.add_argument(
        "--dump-stats",
        default=None,
        metavar="<path to stats.json>",
        help="Dump statistics to a file.",
    )
    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        help="Exit if an exception occurs during filtering.\
            Useful for debugging custom filters.",
    )
    parser.add_argument(
        "--args", default=[], nargs="+", help="Argument for the profile which receives arguments."
    )
    args = parser.parse_args()
    return args


def main() -> None:
    global FILTER
    signal.signal(signal.SIGINT, sigint_handler)
    args = argparser()
    FILTER = load_compose(
        args.profile,
        *tuple(args.args),
    )

    input_iter = stdin_iter()
    out_str_iter = process_iter(
        input_iter=input_iter, filter=FILTER, exit_on_error=args.exit_on_error
    )
    if args.output:
        with open(args.output, "w") as fp:
            fileout_from_iter(out_str_iter, fp)
    else:
        stdout_from_iter(out_str_iter)
    finalize()
    if args.dump_stats:
        with open(args.dump_stats, "w") as fp:
            fp.write(json.dumps(FILTER.statistics))


if __name__ == "__main__":
    main()
