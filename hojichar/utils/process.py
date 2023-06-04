import logging
from typing import Iterator

import hojichar

logger = logging.getLogger(__name__)


def process_iter(
    input_iter: Iterator[str],
    filter: hojichar.Compose,
    exit_on_error: bool,
) -> Iterator[str]:
    for line in input_iter:
        try:
            doc = filter.apply(hojichar.Document(line))
            if not doc.is_rejected:
                yield doc.text
        except Exception as e:
            if exit_on_error:
                raise e
            else:
                logger.error(f"Caught {type(e)}. Skip processing the line: `{line}`")
                continue
