import dataclasses
import logging
import time
from typing import List

from hojichar.core.filter_interface import Filter
from hojichar.core.models import Document

logger = logging.getLogger(__name__)


class Inspector(Filter):
    def __init__(self, target: str, ignore_filtered, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("hojichar.Inspector")
        self.target = target
        self.ignore_filtered = ignore_filtered

        self.is_rejected = False
        self.text_hash = 0
        self.tokens_hash = 0

    def apply(self, document: Document) -> Document:
        self.inspect(document)
        if not self.ignore_filtered:
            document.is_rejected = False
        return document

    def inspect(self, document) -> None:
        self.is_rejected = False
        self.is_rejected = document.is_rejected
        self.bytes = len(document.text.encode("utf-8"))
        self.time_ns = time.perf_counter_ns()


@dataclasses.dataclass
class FilterStatistics:
    discard_num: int = 0
    diff_bytes: int = 0
    cumulative_time_ns: int = 0

    def get_human_readable_values(self) -> dict:
        ret = {
            "discard_num": self.discard_num,
            "diff_MB": (self.diff_bytes / 1048576),
            "cumlative_time": (self.cumulative_time_ns / 10**9),
        }
        return ret


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


class StatisticsCounter:
    def __init__(self, inspectors: List[Inspector], ignore_filtered):
        counts = dict()
        for inspector in inspectors:
            counts[inspector.target] = FilterStatistics()
        self.counts = counts
        self.doc_counts = DocStatistics()
        self.ignore_filtered = ignore_filtered

    def update_changes(
        self,
        document: Document,
        before_process_inspector: Inspector,
        inspectors: List[Inspector],
    ) -> None:

        previous_inspector = before_process_inspector
        for idx, inspector in enumerate(inspectors):
            # logging how many docs are discarded in each filter.
            if self.ignore_filtered:
                if (not previous_inspector.is_rejected) and inspector.is_rejected:
                    self.counts[inspector.target].discard_num += 1
            else:
                if inspector.is_rejected:
                    self.counts[inspector.target].discard_num += 1

            # logging how much volume of docs are changed in each filter.
            if self.ignore_filtered:
                if (not previous_inspector.is_rejected) and inspector.is_rejected:
                    diff_bytes = -inspector.bytes
                elif previous_inspector.is_rejected and inspector.is_rejected:
                    diff_bytes = 0
                else:
                    diff_bytes = inspector.bytes - previous_inspector.bytes
            else:
                if inspector.is_rejected:
                    diff_bytes = -inspector.bytes
                else:
                    diff_bytes = inspector.bytes - previous_inspector.bytes

            self.counts[inspector.target].diff_bytes += diff_bytes

            process_time_ns = inspector.time_ns - previous_inspector.time_ns
            self.counts[inspector.target].cumulative_time_ns += process_time_ns

            previous_inspector = inspector

        self.doc_counts.processed_num += 1
        self.doc_counts.discard_num += (
            1 if sum([inspector.is_rejected for inspector in inspectors]) > 0 else 0
        )
        self.doc_counts.input_bytes += len(document.original.encode("utf-8"))
        self.doc_counts.output_bytes += len(document.text.encode("utf-8"))
        self.doc_counts.cumulative_time_ns += inspectors[-1].time_ns - inspectors[0].time_ns
        self.doc_counts.total_token_num += len(document.tokens)

    def get_statistics(self) -> dict:
        # about_layers = dict()
        about_layers = []
        for filter_name, stats in self.counts.items():
            # about_layers[key] = self.counts[key].get_human_readable_values()
            item = dict()
            item["name"] = filter_name
            stats = self.counts[filter_name]
            for key, stat in stats.get_human_readable_values().items():
                item[key] = stat
            about_layers.append(item)

        return {
            "total_info": self.doc_counts.get_human_readable_values(),
            "layers_info": about_layers,
        }
