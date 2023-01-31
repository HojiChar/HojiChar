import json
import logging
import numbers
import pprint
from typing import List, Union

import numpy as np

from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.inspection import Inspector, StatisticsCounter
from hojichar.core.models import Document


class Compose(Filter):
    def __init__(
        self,
        filters: List[Union[Filter, TokenFilter]],
        ignore_filtered: bool = True,
        random_state: Union[None, int, np.random.Generator] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Compose a filter from pre-defined filter-objects.

        Parameters
        ----------
        filters : List[Union[Filter, TokenFilter]]
            Filter instances which apply to the corpus.

        ignore_filtered : bool, optional
            Default = True
            If True, filters ignore a document which has `is_rejected` flag.
            By doing so, we avoid applying filters that do not affect the output.

            However, for the purpose of corpus survay, e.g., to determine how many documents
            in the corpus are removed by a certain filter, it is preferable not to be
            affected by the upstream filter. In such a case, you should
            set `ignore_filter` to False.

        random_state : Union[None, int, np.random.Generator], optional
            Default = None
            Seed for applying filters randomly.
            `random_state` must be int or np.random.Generator instance.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.filter_name_list = [f"{i}-{filt.name}" for i, filt in enumerate(self.filters)]
        self.logger = logging.getLogger("hojichar.Compose")
        self.ignore_filtered = ignore_filtered
        self.before_process_inspector = Inspector(
            target="before_process", ignore_filtered=ignore_filtered
        )
        self.inspectors = [
            Inspector(target=name, ignore_filtered=ignore_filtered)
            for name in self.filter_name_list
        ]
        self._statistics = StatisticsCounter(self.inspectors, ignore_filtered=self.ignore_filtered)

        # Turn random_state into a `np.random.Generator` instance.
        if random_state is None:
            self.rng = np.random.default_rng()
        elif isinstance(random_state, numbers.Integral):
            self.rng = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            raise ValueError(f"{random_state} cannot be used to seed.")

    def __call__(self, text: str) -> str:
        document = Document(text)
        document = self.apply(document)
        return document.text

    def apply(self, document: Document) -> Document:
        document = self.before_process_inspector.apply(document)
        for i, filt in enumerate(self.filters):
            if (filt.p == 1) or (filt.p is None):
                document = filt.filter_apply(document)
            else:
                if self.rng.random() < filt.p:
                    document = filt.filter_apply(document)

            # If `ignore_filtered==False`, filters process a doc which is rejected
            # Under the setting, each inspector always re-set `doc.is_rejected` to `False`.
            document = self.inspectors[i].apply(document)

        document.processed_text = document.text
        if sum([inspector.is_rejected for inspector in self.inspectors]):
            document.is_rejected = True
            document.text = ""

        self._statistics.update_changes(document, self.before_process_inspector, self.inspectors)
        return document

    @property
    def statistics(self) -> dict:
        return self._statistics.get_statistics()

    def summary(self, format="print"):
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
