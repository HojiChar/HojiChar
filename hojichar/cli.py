import argparse
import logging
import multiprocessing

import hojichar
from hojichar.utils.io_iter import fileout_from_iter, stdin_iter, stdout_from_iter
from hojichar.utils.load_compose import load_compose
from hojichar.utils.process import reject_iter

MAIN_FILTER: hojichar.Compose
CLI_ARGS: argparse.Namespace
logger = logging.getLogger("hojichar.__main__")


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
    args = parser.parse_args()
    return args


def init_worker(filter: hojichar.Compose, args: argparse.Namespace) -> None:
    global MAIN_FILTER, CLI_ARGS
    MAIN_FILTER = filter
    CLI_ARGS = args


def worker(doc: hojichar.Document) -> hojichar.Document:
    global MAIN_FILTER, CLI_ARGS
    try:
        return MAIN_FILTER.apply(doc)
    except Exception as e:
        if CLI_ARGS.exit_on_error:
            raise e
        else:
            logger.error(f"Caught {type(e)}. Skip processing the line: {doc.text}")
            return hojichar.Document("", is_rejected=True)


def main() -> None:
    args = argparser()
    input_iter = stdin_iter()
    input_doc_iter = (hojichar.Document(s) for s in input_iter)
    filter = load_compose(
        args.profile,
        *tuple(args.args),
    )
    with multiprocessing.Pool(
        processes=args.jobs, initializer=init_worker, initargs=(filter, args)
    ) as pool:
        out_doc_iter = pool.imap_unordered(
            worker,
            input_doc_iter,
        )

        out_str_iter = reject_iter(input_iter=out_doc_iter, discard_rejected=not args.all)
        if args.output:
            with open(args.output, "w") as fp:
                fileout_from_iter(out_str_iter, fp)
        else:
            stdout_from_iter(out_str_iter)


if __name__ == "__main__":
    main()
