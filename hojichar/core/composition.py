import json
import logging
import pprint
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.models import Document


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
        super().__init__(*args, **kwargs)
        self.set_filters(filters)
        self.logger = logging.getLogger("hojichar.Compose")

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

    def apply_stream(
        self, stream: Iterable[Document], batch_size: int = 128
    ) -> Iterable[Document]:
        for i, filt in enumerate(self.filters):
            stream = filt.apply_stream(stream)

        for doc in stream:
            out_stat = self.record_stats(doc)
            self._statistics.update(
                {
                    "num_output": 1,
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
                    "input_chars": len(doc.text),
                }
            )
            doc.extras["__start_ns"] = stat["time_ns"]
            doc.extras["__input_bytes"] = stat["bytes"]
            doc.extras["__input_chars"] = stat["num_chars"]
            yield doc

    @property
    def statistics(self) -> dict:
        return self.get_statistics()

    def summary(self, format: str = "print") -> None:
        info = [
            {
                "layer": i,
                "name": filt.name,
                "doc": filt.__doc__,
            }
            for i, filt in enumerate(self.filters)
        ]

        def to_json(filter_info: dict) -> dict:
            filter_info["doc"] = "".join(d.strip() for d in filter_info["doc"].split("\n"))
            return filter_info

        if format == "json":
            print(json.dumps(list(map(to_json, info)), ensure_ascii=False, indent="\t"))
        if format == "print":
            for layer in info:
                print(f"[{layer['layer']}] {layer['name']}")
                pprint.pprint(layer["doc"])
