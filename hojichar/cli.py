import argparse
import functools
import json
import logging
import os
import sys
from typing import Callable, Iterator, Optional

import tqdm

import hojichar
from hojichar.core.parallel import Parallel
from hojichar.utils.io_iter import fileout_from_iter, stdin_iter, stdout_from_iter
from hojichar.utils.load_compose import load_compose

MAIN_FILTER: hojichar.Compose
CLI_ARGS: argparse.Namespace

logger = logging.getLogger("hojichar.cli")


class ProgressBarIteratorWrapper:
    def __init__(self, iter: Iterator[str], total_bytes: Optional[int] = None) -> None:
        self.iter = iter
        self.pbar = tqdm.tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> str:
        next = self.iter.__next__()
        delta_bytes = len(next.encode("utf-8"))
        self.pbar.update(delta_bytes)
        return next


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
        "--input",
        "-i",
        default=None,
        help="Specifies the path for the input file. Defaults to standard input.\
            If set this path, the progress bar is enabled.",
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
        "--all",
        action="store_true",
        default=False,
        help="A flag that specifies whether to include discarded samples. \
            This is useful when inspecting discarded samples.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        default=None,
        type=int,
        help="The number ob parallel jobs. By default, the nuber of the CPU core.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=hojichar.__version__,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    file_in = None
    file_out = None
    args = argparser()

    input_iter: Iterator[str]
    if args.input:
        file_in = open(args.input)
        input_iter = ProgressBarIteratorWrapper(
            file_in,
            total_bytes=os.path.getsize(args.input),
        )
    else:
        input_iter = ProgressBarIteratorWrapper(stdin_iter())

    writer: Callable[[Iterator[str]], None]
    if args.output:
        file_out = open(args.output, "w")
        writer = functools.partial(fileout_from_iter, file_out)
    else:
        writer = stdout_from_iter

    input_doc_iter = (hojichar.Document(s) for s in input_iter)
    filter = load_compose(args.profile, *tuple(args.args))
    try:
        with Parallel(
            filter=filter, num_jobs=args.jobs, ignore_errors=not args.exit_on_error
        ) as parallel_filter:
            out_doc_iter = parallel_filter.imap_apply(input_doc_iter)
            out_str_iter = (
                (doc.text for doc in out_doc_iter)
                if args.all
                else (doc.text for doc in out_doc_iter if not doc.is_rejected)
            )
            writer(out_str_iter)
    finally:
        input_iter.pbar.close()
        if file_in:
            file_in.close()
        if file_out:
            file_out.close()

    stats = filter.statistics_obj
    print(
        json.dumps(stats.get_human_readable_values(), ensure_ascii=False, indent=2),
        file=sys.stderr,
    )
    if args.dump_stats:
        with open(args.dump_stats, "a") as fp:
            fp.write(json.dumps(stats.get_human_readable_values(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
