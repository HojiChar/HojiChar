import json
import logging
import pprint
from typing import Any, List, Optional, Union

import numpy as np

from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.inspection import Inspector, StatisticsCounter, StatsContainer
from hojichar.core.models import Document


class BeforeProcessFilter(Filter):
    def apply(self, doc: Document) -> Document:
        return doc


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
        self.filters = filters
        self.logger = logging.getLogger("hojichar.Compose")
        self.before_process_inspector = Inspector(
            target_filter=BeforeProcessFilter(), filter_idx=-1
        )
        self.inspectors = [
            Inspector(target_filter=filter, filter_idx=idx)
            for idx, filter in enumerate(self.filters)
        ]
        self._statistics = StatisticsCounter(self.inspectors)

        # Turn random_state into a `np.random.Generator` instance.
        if random_state is None:
            self.rng = np.random.default_rng()
        elif isinstance(random_state, int):
            self.rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            raise ValueError(f"{random_state} cannot be used to seed.")

    def __call__(self, text: str) -> str:
        document = Document(text)
        document = self.apply(document)
        if document.is_rejected:
            return ""
        else:
            return document.text

    def _apply_filter(self, filt: Union[Filter, TokenFilter], document: Document) -> Document:
        if document.is_rejected and filt.skip_rejected:
            pass
        else:
            if filt.p == 1:
                document = filt.apply_filter(document)
            else:
                if self.rng.random() < filt.p:
                    document = filt.apply_filter(document)
        return document

    def apply(self, document: Document) -> Document:
        document = self.before_process_inspector.apply(document)
        previous_inspector = self.before_process_inspector
        for i, filt in enumerate(self.filters):
            inspector = self.inspectors[i]
            document = self._apply_filter(filt=filt, document=document)
            document = inspector.apply(document)
            if (not previous_inspector.is_rejected) and inspector.is_rejected:
                document.reject_reason = filt.get_jsonalbe_vars(exclude_keys={"skip_rejected"})
            previous_inspector = inspector

        self._statistics.update_changes(document, self.before_process_inspector, self.inspectors)
        return document

    @property
    def statistics(self) -> dict:
        return self._statistics.get_statistics()

    @property
    def statistics_obj(self) -> StatsContainer:
        return self._statistics.stats

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
