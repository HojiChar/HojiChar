import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

from hojichar.utils.warn_deprecation import deprecated_since


@deprecated_since("0.1.0", "Document")
class Token:
    def __init__(self, text: str, is_rejected: bool = False) -> None:
        self.text = text
        self.__original = text
        self.is_rejected = is_rejected

    @property
    def original(self) -> str:
        return self.__original

    def __str__(self) -> str:
        return self.text


class Document:
    """
    Document class represents a text document with metadata.
    It contains the text of the document, a flag indicating whether it is rejected,
     and additional metadata stored in the `extras` dictionary.

    The `tokens` attribute will be deprecated in future versions,
    and users are encouraged to use the `extras` dictionary to store token-related information.

    Attributes:
        text (str): The text content of the document.
        is_rejected (bool): A flag indicating whether the document is rejected.
        extras (Dict[str, Any]): A dictionary to store additional metadata about the document.
        reject_reason (Dict[str, Any]): A dictionary to store the reason for rejection. The
          filter class and the member name and value will logged at the filter is logged here.

    Next attributes will be deprecated in future versions:
        tokens (List[Token]): A list of tokens extracted from the document.
    """

    def __init__(
        self,
        text: str,
        is_rejected: bool = False,
        tokens: Optional[List[Token]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.text = text
        self.__original = text
        self.is_rejected = is_rejected
        if tokens is None:
            self.tokens: List[Token] = []
        else:
            self.tokens = tokens

        if extras is None:
            self.extras: Dict[str, Any] = {}
        else:
            self.extras = extras

        self.reject_reason: Dict[str, Any] = {}

    @property
    def original(self) -> str:
        return self.__original

    @deprecated_since("1.0.0")
    def set_tokens(self, tokens: List[str]) -> None:
        self.tokens = [Token(token) for token in tokens]

    @deprecated_since("1.0.0")
    def get_tokens(self) -> List[str]:
        return [token.text for token in self.tokens]

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return (
            f"Document(text={self.text!r}, is_rejected={self.is_rejected}, extras={self.extras})"  # noqa
        )


@dataclass
class Statistics:
    """
    Statistics class to track the performance of the document processing pipeline.
    """

    name: Optional[str] = None
    input_num: int = 0
    input_bytes: int = 0
    input_chars: int = 0
    output_num: int = 0
    output_bytes: int = 0
    output_chars: int = 0
    discard_num: int = 0
    diff_bytes: int = 0
    diff_chars: int = 0
    cumulative_time_ns: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Statistics object to a dictionary.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def update(self, other: "Statistics") -> None:
        """
        Update the statistics by adding another Statistics object.
        """
        self.input_num += other.input_num
        self.input_bytes += other.input_bytes
        self.input_chars += other.input_chars
        self.output_num += other.output_num
        self.output_bytes += other.output_bytes
        self.output_chars += other.output_chars
        self.discard_num += other.discard_num
        self.diff_bytes += other.diff_bytes
        self.diff_chars += other.diff_chars
        self.cumulative_time_ns += other.cumulative_time_ns
        self.errors += other.errors

    def reset(self) -> "Statistics":
        """
        Reset the statistics to their initial values.
        """
        self.input_num = 0
        self.input_bytes = 0
        self.input_chars = 0
        self.output_num = 0
        self.output_bytes = 0
        self.output_chars = 0
        self.discard_num = 0
        self.diff_bytes = 0
        self.diff_chars = 0
        self.cumulative_time_ns = 0
        self.errors = 0
        return self

    def update_by_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:
        """
        Update the statistics by calculating the differences between two Doc-info mappings.
        This method is used to update the statistics after a filter is applied.
        """
        if not before["is_rejected"] and after["is_rejected"]:
            # Document is rejected after the filter is applied
            self.input_num += 1
            self.input_bytes += before["bytes"]
            self.input_chars += before["chars"]
            self.discard_num += 1
            self.diff_bytes -= before["bytes"]
            self.diff_chars -= before["chars"]
            self.cumulative_time_ns += after["time_ns"] - before["time_ns"]
        else:
            # Document is not rejected or still not rejected after the filter is applied
            self.input_num += 1
            self.input_bytes += before["bytes"]
            self.input_chars += before["chars"]
            self.output_num += 1
            self.output_bytes += after["bytes"]
            self.output_chars += after["chars"]
            self.diff_bytes += after["bytes"] - before["bytes"]
            self.diff_chars += after["chars"] - before["chars"]
            self.cumulative_time_ns += after["time_ns"] - before["time_ns"]

    @staticmethod
    def add(x: "Statistics", y: "Statistics") -> "Statistics":
        """
        Add two Statistics objects together.
        This method assumes that the names of the two Statistics objects match.
        If they do not match, it will raise an AssertionError."""
        assert x.name == y.name, "Layer names must match"
        return Statistics(
            name=x.name,
            input_num=x.input_num + y.input_num,
            input_bytes=x.input_bytes + y.input_bytes,
            input_chars=x.input_chars + y.input_chars,
            output_num=x.output_num + y.output_num,
            output_bytes=x.output_bytes + y.output_bytes,
            output_chars=x.output_chars + y.output_chars,
            discard_num=x.discard_num + y.discard_num,
            diff_bytes=x.diff_bytes + y.diff_bytes,
            diff_chars=x.diff_chars + y.diff_chars,
            cumulative_time_ns=x.cumulative_time_ns + y.cumulative_time_ns,
            errors=x.errors + y.errors,
        )

    @staticmethod
    def add_list_of_stats(x: List["Statistics"], y: List["Statistics"]) -> List["Statistics"]:
        """
        Add FilterStatistics objects from two lists by matching their names.
        This method assumes that both lists contain FilterStatistics objects
        with the same names, and it will raise a ValueError if the sets of names
        in the two lists do not match.
        """
        # check if the names in both lists match
        names_x = {stat.name for stat in x}
        names_y = {stat.name for stat in y}
        if names_x != names_y:
            raise ValueError(f"name の集合が一致しません: {names_x} vs {names_y}")

        y_map = {stat.name: stat for stat in y}

        # keep the order of x and add corresponding y
        result: List[Statistics] = []
        for stat_x in x:
            stat_y = y_map[stat_x.name]
            result.append(Statistics.add(stat_x, stat_y))

        return result

    @staticmethod
    def get_filter(name: str, stats: List["Statistics"]) -> "Statistics":
        """
        Get a Statistics object by its name from a list of statistics.
        If the name is not found, return None.
        """
        for stat in stats:
            if stat.name == name:
                return stat
        raise KeyError(f"Statistics with name '{name}' not found in the list.")


def get_doc_info(document: Document) -> dict[str, Any]:
    """
    Create a document-info mapping from a Document instance.
    This function is used to extract metadata from the Document for statistics tracking.
    """
    return {
        "is_rejected": document.is_rejected,
        "bytes": len(document.text.encode("utf-8")),
        "chars": len(document.text),
        "time_ns": time.perf_counter_ns(),
    }
