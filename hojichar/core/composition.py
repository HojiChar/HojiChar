import json
import logging
import pprint
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from hojichar.core import inspection
from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.models import Document, Statistics, get_doc_info
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
                    sub._set_rng_if_not_initialized(self._rng)
                    name = f"{filter_idx}-{sub.__class__.__name__}"
                    sub.name = name
                    sub._statistics.name = name
                    self.filters.append(sub)
                    filter_idx += 1
            else:
                f._set_rng_if_not_initialized(self._rng)
                name = f"{filter_idx}-{f.__class__.__name__}"
                f.name = name
                f._statistics.name = name
                self.filters.append(f)
                filter_idx += 1

    def __call__(self, text: str, **kwargs: Any) -> str:
        """
        Apply the composed filter to a text and return the processed text.
        If the document is rejected, return an empty string.
        """
        document = Document(text, **kwargs)
        document = self.apply(document)
        if document.is_rejected:
            return ""
        else:
            return document.text

    def apply(self, document: Document) -> Document:
        """
        Apply the composed filter to a document and return the processed document.
        """
        stat = get_doc_info(document)
        for i, filt in enumerate(self.filters):
            document = filt._apply(document)
        new_stat = get_doc_info(document)
        self._statistics.update_by_diff(stat, new_stat)
        return document

    def apply_batch(self, batch: Sequence[Document]) -> List[Document]:
        """
        Apply the composed filter to a batch of documents and return the processed documents.
        The `apply_batch` method implemented in sub-filters is called in order.
        """

        stats = [get_doc_info(doc) for doc in batch]
        for i, filt in enumerate(self.filters):
            batch = filt._apply_batch(batch)
        batch = self._finalize_batch(batch, stats)
        return list(batch)

    def apply_stream(self, stream: Iterable[Document]) -> Iterable[Document]:
        """
        Apply the composed filter to a stream of documents and return the processed documents.
        The `apply_stream` method implemented in sub-filters is called in order.


        In a sub-filter, if `apply_batch` is overridden and implemented, you need to set `use_batch`
        to True at that filter to utilize that implementation. Otherwise, the
        method implemented in `apply` will be applied to the stream.
        """
        stream = self._count_input_stats(stream)
        for i, filt in enumerate(self.filters):
            stream = filt.apply_stream(stream)

        for doc in stream:
            in_stat = doc.extras["__init_stats"]
            out_stat = get_doc_info(doc)

            self._statistics.update_by_diff(in_stat, out_stat)
            del doc.extras["__init_stats"]
            yield doc

    def _count_input_stats(self, stream: Iterable[Document]) -> Iterable[Document]:
        for doc in stream:
            doc.extras["__init_stats"] = get_doc_info(doc)
            yield doc

    def get_total_statistics(self) -> List[Statistics]:
        """
        Get the statistics of the Compose object and sub filters.

        The statistics of the Compose class are stored in an object with the name "Total",
        and sub-filters's are stored with names in the format {filter_index}-{filter class name}.
        """
        stats = []
        stats.append(self.get_statistics())
        for i, filt in enumerate(self.filters):
            stats.append(filt.get_statistics())
        return stats

    def get_total_statistics_map(self) -> List[Dict[str, Any]]:
        """
        Get the statistics of the Compose object and sub filters as a list of dictionaries.
        """
        stats = self.get_total_statistics()
        return [stat.to_dict() for stat in stats]

    def shutdown(self) -> None:
        for f in self.filters:
            f.shutdown()

        super().shutdown()

    @property
    def statistics(self) -> dict:
        """
        Deprecated

        Get the statistics of the Compose object and sub filters.

        This property is retained for compatibility with previous versions.
        Please use `get_total_statistics` or `get_total_statistics_map` instead.
        """
        return inspection.statistics_obj_adapter(  # type: ignore
            self.get_total_statistics()
        ).get_human_readable_values()

    @property
    def statistics_obj(self) -> inspection.StatsContainer:
        """
        Deprecated

        Get the statistics of the Compose object and sub filters.
        This method returns a StatsContainer object which contains the statistics
        of the Compose object and sub filters.

        This property is retained for compatibility with previous versions.
        Please use `get_total_statistics` or `get_total_statistics_map` instead.
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
