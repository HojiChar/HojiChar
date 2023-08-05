from __future__ import annotations

import functools
import logging
import multiprocessing
import os
import signal
from copy import copy
from typing import Iterator

import hojichar
from hojichar.core.inspection import StatsContainer

logger = logging.getLogger(__name__)


PARALLEL_BASE_FILTER: hojichar.Compose
WORKER_PARAM_IGNORE_ERRORS: bool


def _init_worker(filter: hojichar.Compose, ignore_errors: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global PARALLEL_BASE_FILTER, WORKER_PARAM_IGNORE_ERRORS
    PARALLEL_BASE_FILTER = hojichar.Compose(copy(filter.filters))  # TODO random state treatment
    WORKER_PARAM_IGNORE_ERRORS = ignore_errors


def _worker(
    doc: hojichar.Document,
) -> tuple[hojichar.Document, int, StatsContainer, str | None]:
    global PARALLEL_BASE_FILTER, WORKER_PARAM_IGNORE_ERRORS
    ignore_errors = WORKER_PARAM_IGNORE_ERRORS
    error_message = None
    try:
        result = PARALLEL_BASE_FILTER.apply(doc)
    except Exception as e:
        if ignore_errors:
            logger.error(e)
            error_message = str(e)
            result = hojichar.Document("", is_rejected=True)
        else:
            raise e  # If we're not ignoring errors, let this one propagate
    return result, os.getpid(), PARALLEL_BASE_FILTER.statistics_obj, error_message


class Parallel:
    """
    The Parallel class provides a way to apply a hojichar.Compose filter
    to an iterator of documents in a parallel manner using a specified
    number of worker processes. This class should be used as a context
    manager with a 'with' statement.

    Example:

    doc_iter = (hojichar.Document(d) for d in open("my_text.txt"))
    with Parallel(my_filter, num_jobs=8) as pfilter:
        for doc in pfilter.imap_apply(doc_iter):
            pass  # Process the filtered document as needed.
    """

    def __init__(
        self, filter: hojichar.Compose, num_jobs: int | None = None, ignore_errors: bool = False
    ):
        """
        Initializes a new instance of the Parallel class.

        Args:
            filter (hojichar.Compose): A composed filter object that specifies the
                processing operations to apply to each document in parallel.
                A copy of the filter is made within a 'with' statement. When the 'with'
                block terminates,the statistical information obtained through `filter.statistics`
                or`filter.statistics_obj` is replaced with the total value of the statistical
                information processed within the 'with' block.

            num_jobs (int | None, optional): The number of worker processes to use.
                If None, then the number returned by os.cpu_count() is used. Defaults to None.
            ignore_errors (bool, optional): If set to True, any exceptions thrown during
                the processing of a document will be caught and logged, but will not
                stop the processing of further documents. If set to False, the first
                exception thrown will terminate the entire parallel processing operation.
                Defaults to False.
        """
        self.filter = filter
        self.num_jobs = num_jobs
        self.ignore_errors = ignore_errors

        self._pool: multiprocessing.pool.Pool | None = None
        self._pid_stats: dict[int, StatsContainer] | None = None

    def __enter__(self) -> Parallel:
        self._pool = multiprocessing.Pool(
            processes=self.num_jobs,
            initializer=_init_worker,
            initargs=(self.filter, self.ignore_errors),
        )
        self._pid_stats = dict()
        return self

    def imap_apply(self, docs: Iterator[hojichar.Document]) -> Iterator[hojichar.Document]:
        """
        Takes an iterator of Documents and applies the Compose filter to
        each Document in a parallel manner. This is a generator method
        that yields processed Documents.

        Args:
            docs (Iterator[hojichar.Document]): An iterator of Documents to be processed.

        Raises:
            RuntimeError: If the Parallel instance is not properly initialized. This
                generally happens when the method is called outside of a 'with' statement.
            Exception: If any exceptions are raised within the worker processes.

        Yields:
            Iterator[hojichar.Document]: An iterator that yields processed Documents.
        """
        if self._pool is None or self._pid_stats is None:
            raise RuntimeError(
                "Parallel instance not properly initialized. Use within a 'with' statement."
            )
        try:
            for doc, pid, stats_obj, err_msg in self._pool.imap_unordered(_worker, docs):
                self._pid_stats[pid] = stats_obj
                if err_msg is not None:
                    logger.error(f"Error in worker {pid}: {err_msg}")
                yield doc
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        if self._pool:
            self._pool.terminate()
            self._pool.join()
        if self._pid_stats:
            self.filter._statistics.stats = self.filter._statistics.stats + functools.reduce(
                lambda x, y: x + y, self._pid_stats.values()
            )

    @property
    def statistics_obj(self) -> StatsContainer:
        """
        Returns a statistics object of the total statistical
        values processed within the Parallel block.

        Returns:
            StatsContainer: Statistics object
        """
        if self._pid_stats:
            stats: StatsContainer = functools.reduce(lambda x, y: x + y, self._pid_stats.values())
        else:
            stats = copy(self.filter.statistics_obj).reset()
        return stats

    @property
    def statistics(self) -> dict:
        """
        Returns a statistics dict which friendly with human of the total statistical
        values processed within the Parallel block.

        Returns:
            dict: Human readable statistics values
        """
        return self.statistics_obj.get_human_readable_values()
