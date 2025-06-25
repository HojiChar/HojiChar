import json
import logging
import pprint
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from hojichar.core import inspection
from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.models import DocInfo, Document, Statistics
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
        self.logger = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

        self._statistics.name = "Total"

    def set_filters(self, filters: List[Union[Filter, TokenFilter]]) -> None:
        """
        Set the filter to a Compose object. The filter is expanded if the
        list of filters in the argument contains a filter bound by Compose.

        Args:
            filters (List[Union[Filter, TokenFilter]]): Target filters
        """
        self.filters: List[Union[Filter, TokenFilter]] = []

        filter_idx = 0
        for f in filters:
            if isinstance(f, Compose):
                for sub in f.filters:
                    sub.set_rng_if_not_initialized(self._rng)
                    name = f"{filter_idx}-{sub.__class__.__name__}"
                    sub.name = name
                    sub._statistics.name = name
                    self.filters.append(sub)
                    filter_idx += 1
            else:
                f.set_rng_if_not_initialized(self._rng)
                name = f"{filter_idx}-{f.__class__.__name__}"
                f.name = name
                f._statistics.name = name
                self.filters.append(f)
                filter_idx += 1

    def __call__(self, text: str, **kwargs: Any) -> str:
        document = Document(text, **kwargs)
        document = self.apply(document)
        if document.is_rejected:
            return ""
        else:
            return document.text

    def apply(self, document: Document) -> Document:
        stat = DocInfo(document)
        for i, filt in enumerate(self.filters):
            document = filt._apply(document)
        new_stat = DocInfo(document)
        diff_stat = Statistics.from_diff(stat, new_stat)
        self._statistics.update(diff_stat)
        return document

    def apply_batch(self, batch: Sequence[Document]) -> List[Document]:
        stats = [DocInfo(doc) for doc in batch]
        for i, filt in enumerate(self.filters):
            batch = filt._apply_batch(batch)
        batch = self._finalize_batch(batch, stats)
        return list(batch)

    def apply_stream(self, stream: Iterable[Document]) -> Iterable[Document]:
        stream = self._count_input_stats(stream)
        for i, filt in enumerate(self.filters):
            stream = filt.apply_stream(stream)

        for doc in stream:
            in_stat = DocInfo.from_dict(doc.extras["__init_stats"])
            in_stat = DocInfo(doc)
            out_stat = DocInfo(doc)

            diff_stat = Statistics.from_diff(in_stat, out_stat)
            self._statistics.update(diff_stat)
            del doc.extras["__init_stats"]
            yield doc

    def _count_input_stats(self, stream: Iterable[Document]) -> Iterable[Document]:
        """
        Count the statistics of the input documents.
        """
        for doc in stream:
            stat = DocInfo(doc)
            doc.extras["__init_stats"] = asdict(stat)
            yield doc

    def get_total_statistics(self) -> List[Statistics]:
        """
        Get the total statistics of the Compose object and sub filters.
        """
        stats = []
        stats.append(self.get_statistics())
        for i, filt in enumerate(self.filters):
            stats.append(filt.get_statistics())
        return stats

    def get_total_statistics_map(self) -> List[Dict[str, Any]]:
        """
        Get the total statistics of the Compose object and sub filters.
        """
        stats = self.get_total_statistics()
        return [stat.to_dict() for stat in stats]

    def shutdown(self) -> None:
        for f in self.filters:
            f.shutdown()

        super().shutdown()

    @property
    def statistics(self) -> dict:
        return inspection.statistics_obj_adapter(  # type: ignore
            self.get_total_statistics()
        ).get_human_readable_values()

    @property
    def statistics_obj(self) -> inspection.StatsContainer:
        """
        Get the statistics of the Compose object and sub filters.
        This method returns a StatsContainer object which contains the statistics
        of the Compose object and sub filters.
        """
        return inspection.statistics_obj_adapter(self.get_total_statistics())  # type: ignore

    @deprecated_since("1.0.0", "get_total_statistics")
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
