import json
import logging
import pprint
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.models import Document
from hojichar.utils.warn_deprecation import deprecated_since


class Compose(Filter):
    def __init__(
        self,
        filters: List[Union[Filter, TokenFilter]],
        random_state: Optional[Union[int, np.random.Generator]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Compose a filter from pre-defined filter-objects.
        Filter which has `skip_rejected` flag ignores a document which has `is_rejected` flag.
        By doing so, Compose avoid applying filters that do not affect the output.

        Parameters
        ----------
        filters : List[Union[Filter, TokenFilter]]
            Filter instances which apply to the corpus.

        random_state : Union[None, int, np.random.Generator], optional
            Default = None
            Seed for applying filters randomly.
            `random_state` must be int or np.random.Generator instance.
        """
        super().__init__(random_state=random_state, *args, **kwargs)
        self.set_filters(filters)
        self.logger = logging.getLogger("hojichar.Compose")

        self._total_stats: Optional[List[Dict[str, Any]]] = None

    def set_filters(self, filters: List[Union[Filter, TokenFilter]]) -> None:
        """
        Set the filter to a Compose object. The filter is expanded if the
        list of filters in the argument contains a filter bound by Compose.

        Args:
            filters (List[Union[Filter, TokenFilter]]): Target filters
        """
        self.filters: List[Union[Filter, TokenFilter]] = []
        for filter in filters:
            if isinstance(filter, Compose):
                sub_filters = filter.filters
                for filt in sub_filters:
                    filt.set_rng_if_not_initialized(self._rng)
                    self.filters.append(filt)
            else:
                filter.set_rng_if_not_initialized(self._rng)
                self.filters.append(filter)

    def __call__(self, text: str, **kwargs: Any) -> str:
        document = Document(text)
        document = self.apply(document)
        if document.is_rejected:
            return ""
        else:
            return document.text

    def apply(self, document: Document) -> Document:
        stat = self.record_stats(document)
        for i, filt in enumerate(self.filters):
            document = filt._apply(document)
        new_stat = self.record_stats(document)
        diff_stat = self.diff_stats(stat, new_stat)
        self._statistics.update(diff_stat)
        return document

    def apply_batch(self, batch: Sequence[Document]) -> List[Document]:
        stats = [self.record_stats(doc) for doc in batch]
        for i, filt in enumerate(self.filters):
            batch = filt._apply_batch(batch)
        batch = self._finalize_batch(batch, stats)
        return list(batch)

    def apply_stream(self, stream: Iterable[Document]) -> Iterable[Document]:
        stream = self._count_input_stats(stream)
        for i, filt in enumerate(self.filters):
            stream = filt.apply_stream(stream)

        for doc in stream:
            out_stat = self.record_stats(doc)
            self._statistics.update(
                {
                    "num_output": 0 if doc.is_rejected else 1,
                    "output_bytes": 0 if doc.is_rejected else out_stat["bytes"],
                    "output_chars": 0 if doc.is_rejected else len(doc.text),
                    "num_discard": 1 if doc.is_rejected else 0,
                    "diff_bytes": -out_stat["bytes"]
                    if doc.is_rejected
                    else out_stat["bytes"] - doc.extras.get("__input_bytes", out_stat["bytes"]),
                    "diff_chars": -out_stat["num_chars"]
                    if doc.is_rejected
                    else out_stat["num_chars"]
                    - doc.extras.get("__input_chars", out_stat["num_chars"]),
                    "cumulative_time_ns": out_stat["time_ns"]
                    - doc.extras.get("__start_ns", out_stat["time_ns"]),
                }
            )
            del doc.extras["__start_ns"]
            del doc.extras["__input_bytes"]
            del doc.extras["__input_chars"]
            yield doc

    def _count_input_stats(self, stream: Iterable[Document]) -> Iterable[Document]:
        """
        Count the statistics of the input documents.
        """
        for doc in stream:
            stat = self.record_stats(doc)
            self._statistics.update(
                {
                    "num_input": 1,
                    "input_bytes": stat["bytes"],
                    "input_chars": stat["num_chars"],
                }
            )
            doc.extras["__start_ns"] = stat["time_ns"]
            doc.extras["__input_bytes"] = stat["bytes"]
            doc.extras["__input_chars"] = stat["num_chars"]
            yield doc

    def get_total_statistics(self) -> List[Dict[str, Any]]:
        if self._total_stats is None:
            stats = []
            stats.append({"name": "total", **self.get_statistics()})
            for i, filt in enumerate(self.filters):
                stats.append({"name": f"{i}-{filt.name}", **filt.get_statistics()})

            return stats
        else:
            return self._total_stats

    def set_total_statistics(self, stats: List[Dict[str, Any]]) -> None:
        """
        Set the total statistics for the Compose object.
        This is used to set pre-computed statistics.
        """
        self._total_stats = stats

    @staticmethod
    def merge_total_stats(
        x: List[Dict[str, Any]],
        y: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge two statistics lists by 'name'.
        同じ 'name' をもつエントリがあれば、'name' 以外のキーは数値として加算します。
        """
        merged: Dict[str, Dict[str, Any]] = {}
        for stat in x:
            merged[stat["name"]] = stat.copy()
        for stat in y:
            name = stat["name"]
            if name not in merged:
                merged[name] = stat.copy()
            else:
                base = merged[name]
                for k, v in stat.items():
                    if k == "name":
                        continue
                    if isinstance(v, (int, float)):
                        base[k] = base.get(k, 0) + v
                    else:
                        base[k] = v
        return list(merged.values())
