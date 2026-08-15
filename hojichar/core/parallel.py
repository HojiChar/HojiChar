from __future__ import annotations

import functools
import logging
import multiprocessing
import os
import signal
import threading
from copy import copy
from multiprocessing.pool import Pool
from typing import Iterator, List

import hojichar
from hojichar.core import inspection
from hojichar.core.models import Statistics

logger = logging.getLogger(__name__)


_START_METHOD_ENV_VAR = "HOJICHAR_MP_START_METHOD"

PARALLEL_BASE_FILTER: hojichar.Compose
WORKER_PARAM_IGNORE_ERRORS: bool


def _get_parallel_context() -> multiprocessing.context.BaseContext:
    """Return the multiprocessing context used by :class:`Parallel`.

    ``fork`` is selected explicitly when it is available so worker processes can
    inherit filters which cannot be pickled.  The environment variable is an
    escape hatch for applications where forking is unsafe; ``default`` delegates
    the choice back to Python's global/default multiprocessing context.
    """
    configured_method = os.getenv(_START_METHOD_ENV_VAR)
    if configured_method is None:
        start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else None
    elif configured_method == "default":
        start_method = None
    else:
        start_method = configured_method

    try:
        return multiprocessing.get_context(start_method)
    except ValueError as error:
        supported_methods = ["default", *multiprocessing.get_all_start_methods()]
        raise ValueError(
            f"Invalid {_START_METHOD_ENV_VAR}={configured_method!r}. "
            f"Choose one of: {', '.join(supported_methods)}."
        ) from error


def _init_worker(filter: hojichar.Compose, ignore_errors: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global PARALLEL_BASE_FILTER, WORKER_PARAM_IGNORE_ERRORS
    PARALLEL_BASE_FILTER = hojichar.Compose(copy(filter.filters))  # TODO random state treatment
    WORKER_PARAM_IGNORE_ERRORS = ignore_errors


def _worker(
    doc: hojichar.Document,
) -> tuple[hojichar.Document, int, List[Statistics], str | None]:
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
    return result, os.getpid(), PARALLEL_BASE_FILTER.get_total_statistics(), error_message


class _InFlightGate:
    """Bounds documents drawn from the input but not yet returned to the caller.

    ``feed`` wraps the input iterator and acquires one permit *before* each
    document is drawn (acquiring afterwards would hold one extra pre-fetched
    document while waiting). ``imap_apply`` releases the permit only after the
    corresponding result has been handed back to the caller, so at most
    ``max_in_flight`` documents exist anywhere between the input iterator and
    the caller at any moment.

    ``feed`` runs inside Pool's task-handler thread. The polling acquire
    observes :meth:`stop` before and after each successful acquire (handing
    the permit back if stopped), and abnormal exits stop the gate before
    returning their permit, so the feeder never draws from the source after
    consumption ended and pool shutdown never blocks on the gate.
    """

    def __init__(self, max_in_flight: int) -> None:
        self._semaphore = threading.Semaphore(max_in_flight)
        self._stop_feeding = threading.Event()

    def feed(self, docs: Iterator[hojichar.Document]) -> Iterator[hojichar.Document]:
        iterator = iter(docs)
        while self._acquire():
            try:
                doc = next(iterator)
            except StopIteration:
                self.release()
                return
            yield doc

    def release(self) -> None:
        self._semaphore.release()

    def stop(self) -> None:
        """Unblock the feeder; called when consumption ends and at shutdown."""
        self._stop_feeding.set()

    def _acquire(self) -> bool:
        while not self._stop_feeding.is_set():
            if self._semaphore.acquire(timeout=0.1):
                if self._stop_feeding.is_set():
                    # Stopped while blocked on (or right after) the acquire:
                    # hand the permit back instead of drawing one more
                    # document from a possibly blocking source.
                    self._semaphore.release()
                    return False
                return True
        return False


class Parallel:
    """
    The Parallel class provides a way to apply a hojichar.Compose filter
    to an iterator of documents in a parallel manner using a specified
    number of worker processes. This class should be used as a context
    manager with a 'with' statement.

    On platforms which support it, Parallel explicitly uses the ``fork`` start
    method so filters that cannot be pickled can be inherited by workers. Set
    ``HOJICHAR_MP_START_METHOD`` to ``spawn``, ``forkserver``, or ``default`` to
    choose another context. Non-fork contexts require the Compose object and its
    filters to be picklable.

    Example:

    doc_iter = (hojichar.Document(d) for d in open("my_text.txt"))
    with Parallel(my_filter, num_jobs=8) as pfilter:
        for doc in pfilter.imap_apply(doc_iter):
            pass  # Process the filtered document as needed.
    """

    def __init__(
        self,
        filter: hojichar.Compose,
        num_jobs: int | None = None,
        ignore_errors: bool = False,
        ordered: bool = False,
        max_in_flight: int | None = None,
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
            ordered (bool, optional): If set to True, processed documents are yielded in
                the same order as the input documents. If set to False, documents are
                yielded as soon as their processing completes. Defaults to False.
            max_in_flight (int | None, optional): Strict upper bound on the number of
                documents drawn from the input iterator whose results have not yet been
                handed back to the caller. A yielded document keeps its permit until the
                caller requests the next one, so with `max_in_flight=1` the pool holds a
                single document end to end (no pipelining). Without a bound, Pool's
                task-handler thread drains the input as fast as the worker pipe accepts,
                so a producer that outruns the filters can buffer a large number of
                documents. If None, no explicit bound is applied. Defaults to None.
        """
        if max_in_flight is not None and max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.filter = filter
        self.num_jobs = num_jobs
        self.ignore_errors = ignore_errors
        self.ordered = ordered
        self.max_in_flight = max_in_flight

        self._pool: Pool | None = None
        self._pid_stats: dict[int, List[Statistics]] | None = None
        self._gates: list[_InFlightGate] = []

    def __enter__(self) -> Parallel:
        context = _get_parallel_context()
        self._pool = context.Pool(
            processes=self.num_jobs,
            initializer=_init_worker,
            initargs=(self.filter, self.ignore_errors),
        )
        self._pid_stats = dict()
        self._gates = []
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
        gate: _InFlightGate | None = None
        if self.max_in_flight is not None:
            gate = _InFlightGate(self.max_in_flight)
            self._gates.append(gate)
            docs = gate.feed(docs)
        try:
            results = (
                self._pool.imap(_worker, docs)
                if self.ordered
                else self._pool.imap_unordered(_worker, docs)
            )
            for doc, pid, stat, err_msg in results:
                self._pid_stats[pid] = stat
                if err_msg is not None:
                    logger.error(f"Error in worker {pid}: {err_msg}")
                # The permit is returned only once the caller has taken the
                # document: releasing before the yield would let the feeder
                # momentarily draw max_in_flight + 1 documents. On an abnormal
                # exit (close/throw at the yield point) the gate is stopped
                # *before* the permit is returned, so a feeder woken by the
                # release always observes the stop and cannot draw again.
                try:
                    yield doc
                except BaseException:
                    if gate is not None:
                        gate.stop()
                        gate.release()
                    raise
                else:
                    if gate is not None:
                        gate.release()
        except Exception:
            self.__exit__(None, None, None)
            raise
        finally:
            if gate is not None:
                gate.stop()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        # Feeders blocked on a max_in_flight gate must exit before the pool is
        # joined, or shutdown would wait on them forever.
        for gate in self._gates:
            gate.stop()
        if self._pool:
            self._pool.terminate()
            self._pool.join()
        if self._pid_stats:
            total_stats = functools.reduce(
                lambda x, y: Statistics.add_list_of_stats(x, y), self._pid_stats.values()
            )
            self.filter._statistics.update(Statistics.get_filter("Total", total_stats))
            for stat in total_stats:
                for filt in self.filter.filters:
                    if stat.name == filt.name:
                        filt._statistics.update(stat)
                        break

    def get_total_statistics(self) -> List[Statistics]:
        """
        Returns a statistics object of the total statistical
        values processed within the Parallel block.

        Returns:
            StatsContainer: Statistics object
        """
        if self._pid_stats:
            total_stats = functools.reduce(
                lambda x, y: Statistics.add_list_of_stats(x, y), self._pid_stats.values()
            )
            return total_stats
        else:
            return []

    def get_total_statistics_map(self) -> List[dict]:
        return [stat.to_dict() for stat in self.get_total_statistics()]

    @property
    def statistics_obj(self) -> inspection.StatsContainer:
        """
        Returns the statistics object of the Parallel instance.
        This is a StatsContainer object which contains the statistics
        of the Parallel instance and sub filters.

        Returns:
            StatsContainer: Statistics object
        """
        return inspection.statistics_obj_adapter(self.get_total_statistics())  # type: ignore
