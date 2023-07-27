from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, List, Union

from hojichar.core.filter_interface import Filter, TokenFilter
from hojichar.core.models import Document

logger = logging.getLogger(__name__)


class Inspector(Filter):
    def __init__(
        self, target_filter: Union[Filter, TokenFilter], filter_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("hojichar.Inspector")
        self.target_filter = target_filter
        self.filter_idx = filter_idx
        self.target = f"{filter_idx}-{target_filter.name}"

        self.is_rejected = False
        self.text_hash = 0
        self.tokens_hash = 0

    def apply(self, document: Document) -> Document:
        self.inspect(document)
        return document

    def inspect(self, document: Document) -> None:
        self.is_rejected = False
        self.is_rejected = document.is_rejected
        self.bytes = len(document.text.encode("utf-8"))
        self.time_ns = time.perf_counter_ns()


@dataclasses.dataclass
class FilterStatistics:
    name: str
    discard_num: int = 0
    diff_bytes: int = 0
    cumulative_time_ns: int = 0
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def get_human_readable_values(self) -> dict:
        ret = {
            "name": self.name,
            "discard_num": self.discard_num,
            "diff_MB": (self.diff_bytes / 1048576),  # 1024**2
            "cumulative_time": (self.cumulative_time_ns / 10**9),
            "params": self.params,
        }
        return ret

    def __add__(self, other: FilterStatistics) -> FilterStatistics:
        assert self.name == other.name, "Layer names must match"
        return FilterStatistics(
            self.name,
            self.discard_num + other.discard_num,
            self.diff_bytes + other.diff_bytes,
            self.cumulative_time_ns + other.cumulative_time_ns,
            self.params,
        )


@dataclasses.dataclass
class DocStatistics:
    processed_num: int = 0
    discard_num: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    cumulative_time_ns: int = 0
    total_token_num: int = 0

    def get_human_readable_values(self) -> dict:
        ret = {
            "processed_num": self.processed_num,
            "discard_num": self.discard_num,
            "input_MB": (self.input_bytes / 1000**2),
            "output_MB": (self.output_bytes / 1000**2),
            "cumulative_time": (self.cumulative_time_ns / 10**9),
            "total_token_num": self.total_token_num,
        }
        return ret

    def __add__(self, other: DocStatistics) -> DocStatistics:
        return DocStatistics(
            self.processed_num + other.processed_num,
            self.discard_num + other.discard_num,
            self.input_bytes + other.input_bytes,
            self.output_bytes + other.output_bytes,
            self.cumulative_time_ns + other.cumulative_time_ns,
            self.total_token_num + other.total_token_num,
        )


@dataclasses.dataclass
class StatsContainer:
    total_info: DocStatistics
    layers_info: Dict[str, FilterStatistics]  # Key of the dict is filter name.

    def __add__(self, other: StatsContainer) -> StatsContainer:
        assert self.layers_info.keys() == other.layers_info.keys(), "Layer names must match"
        return StatsContainer(
            self.total_info + other.total_info,
            {k: v + other.layers_info[k] for k, v in self.layers_info.items()},
        )

    def get_human_readable_values(self) -> dict:
        return {
            "total_info": self.total_info.get_human_readable_values(),
            "layers_info": [
                layer.get_human_readable_values() for layer in self.layers_info.values()
            ],
        }


class StatisticsCounter:
    def __init__(self, inspectors: List[Inspector]) -> None:
        counts = dict()
        for inspector in inspectors:
            counts[inspector.target] = FilterStatistics(
                name=inspector.target,
                params=inspector.target_filter.get_jsonalbe_vars(),
            )
        self.stats = StatsContainer(
            DocStatistics(),
            counts,
        )

    def update_changes(
        self,
        document: Document,
        before_process_inspector: Inspector,
        inspectors: List[Inspector],
    ) -> None:

        # Counting statistics for each filter
        previous_inspector = before_process_inspector
        for idx, inspector in enumerate(inspectors):
            # Logging how many docs are discarded in each filter
            if (not previous_inspector.is_rejected) and inspector.is_rejected:
                self.stats.layers_info[inspector.target].discard_num += 1

            # logging how much volume of docs are changed in each filter.
            if (not previous_inspector.is_rejected) and inspector.is_rejected:
                diff_bytes = -inspector.bytes
            elif previous_inspector.is_rejected and inspector.is_rejected:
                diff_bytes = 0
            else:
                diff_bytes = inspector.bytes - previous_inspector.bytes

            self.stats.layers_info[inspector.target].diff_bytes += diff_bytes

            process_time_ns = inspector.time_ns - previous_inspector.time_ns
            self.stats.layers_info[inspector.target].cumulative_time_ns += process_time_ns

            previous_inspector = inspector

        # Counting total statistics
        self.stats.total_info.processed_num += 1
        self.stats.total_info.discard_num += (
            1 if any([inspector.is_rejected for inspector in inspectors]) > 0 else 0
        )
        self.stats.total_info.input_bytes += len(document.original.encode("utf-8"))
        self.stats.total_info.output_bytes += (
            0 if document.is_rejected else len(document.text.encode("utf-8"))
        )
        self.stats.total_info.cumulative_time_ns += inspectors[-1].time_ns - inspectors[0].time_ns
        self.stats.total_info.total_token_num += len(document.tokens)

    def get_statistics(self) -> dict:
        return self.stats.get_human_readable_values()
