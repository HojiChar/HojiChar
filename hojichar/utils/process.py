import logging
import os
from contextlib import redirect_stdout
from typing import Iterator, Optional, TextIO

import hojichar

logger = logging.getLogger(__name__)


def process_iter(
    input_iter: Iterator[str],
    filter: hojichar.Compose,
    exit_on_error: bool,
    stdout_fp: Optional[TextIO] = None,
) -> Iterator[hojichar.Document]:
    """
    Getting an iterator of string, processing by given hojichar.Compose filter,
    and iterate processed string.
    Standard output of each filter is redirected to stdout_fp. Default is /dev/null.

    Args:
        input_iter (Iterator[str]): Input iterator.
        filter (hojichar.Compose): Processing filter.
        exit_on_error (bool): Halt with error while processing.
        stdout_fp: Redirection of stdout in each filter.

    Raises:
        e: Exception raised during processing in hojichar.Compose

    Yields:
        Iterator[hojichar.Document]: Processed text
    """
    if stdout_fp is None:
        stdout_fp = open(os.devnull, "w")
    with redirect_stdout(stdout_fp):
        for line in input_iter:
            try:
                doc = filter.apply(hojichar.Document(line))
                yield doc
            except Exception as e:
                if exit_on_error:
                    raise e
                else:
                    logger.error(f"Caught {type(e)}. Skip processing the line: `{line}`")
                    continue


def reject_iter(input_iter: Iterator[hojichar.Document], discard_rejected: bool) -> Iterator[str]:
    """_summary_

    Args:
        input_iter (Iterator[hojichar.Document]): Input iterator of the documents.
        discard_rejected (bool): Whether discarding `is_rejected` docs or not.

    Yields:
        Iterator[str]: Output iterator of the string.
    """
    for doc in input_iter:
        if doc.is_rejected and discard_rejected:
            pass
        else:
            yield doc.text
