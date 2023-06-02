import argparse
import importlib.util
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Iterator

import hojichar
from hojichar.utils.io_iter import stdin_iter, stdout_iter

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
        metavar="your_filter.py",
        help="Path to a Python file that implements your custom filter.\
            hojichar.Compose must be defined as FILTER variable in the file.",
    )
    parser.add_argument("--dump-stats", default=None, help="Dump statistics to a file.")
    args = parser.parse_args()
    return args


def load_compose_from_file(profile_path: str) -> hojichar.Compose:
    """_summary_

    Args:
        profile_path (str): Path to a Python file that implements your custom filter.
        hojichar.Compose must be defined as FILTER variable in the file.

    Raises:
        NotImplementedError:

    Returns:
        hojichar.Compose:
    """
    profile = Path(profile_path)
    module_name = profile.stem
    spec = importlib.util.spec_from_file_location(module_name, profile)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "FILTER"):
        filter = getattr(module, "FILTER")
        if not isinstance(filter, hojichar.Compose):
            raise TypeError("FILTER must be hojichar.Compose object.")
        return filter
    else:
        raise NotImplementedError("FILTER must be implemented in the profile.")


def process_iter(input_iter: Iterator[str], filter: hojichar.Compose) -> Iterator[str]:
    for line in input_iter:
        try:
            doc = filter.apply(hojichar.Document(line))
            if not doc.is_rejected:
                yield doc.text
        except Exception as e:
            logger.error(f"{e}. Skip processing the line: {line}")
            continue


def main() -> None:
    global FILTER
    signal.signal(signal.SIGINT, handler=sigint_handler)
    args = argparser()
    FILTER = load_compose_from_file(args.profile)

    input_iter = stdin_iter()
    out_str_iter = process_iter(input_iter, FILTER)
    stdout_iter(out_str_iter)
    finalize()
    if args.dump_stats:
        with open(args.dump_stats, "w") as fp:
            fp.write(json.dumps(FILTER.statistics))


if __name__ == "__main__":
    main()