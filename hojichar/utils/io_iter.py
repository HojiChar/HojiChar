import io
import sys
from typing import Iterator, TextIO


def stdin_iter() -> Iterator[str]:
    """
    Encodes standard output in utf-8 and iterates line by line.

    Wrapping sys.stdin as being opened file, for below reasons:
    - Specifying encoding.
    - Processing against invalid unicode inputs.

    Yields:
        Iterator[str]: Line of stdin stream.
    """
    stdin = io.TextIOWrapper(
        buffer=sys.stdin.buffer,
        encoding="utf-8",
        errors="replace",
    )
    return (line.rstrip("\n") for line in stdin)


def stdout_from_iter(iter: Iterator[str]) -> None:
    """
    Write iterators to standard output.

    Wrapping sys.out as being opened file for below reasons:
    - Specifying encoding.
    - Enforces line buffering for proper behavior in the program subsequently piped.
        - Line buffering is necessary for a case:
            <stdout program> | head -n 1
        If line buffering is not forced, the program runs with full buffering.
        The output is not passed to the subsequent pipe until the buffer is full.
        Thus, the result of the `head` command will not be output immediately.

        - Piped commands sometimes interrupt in the middle of output,
        as is the case with head, less, etc.
        These raise a BrokenPipeError exception and must be handled properly.
    """
    stdout = io.TextIOWrapper(
        buffer=sys.stdout.buffer,
        encoding="utf-8",
        line_buffering=True,
    )
    try:
        for line in iter:
            stdout.write(line + "\n")
    except BrokenPipeError:
        sys.exit(1)


def fileout_from_iter(iter: Iterator[str], fp: TextIO) -> None:
    for line in iter:
        fp.write(line + "\n")
