import io
import sys
from typing import Iterator, TextIO


def stdin_iter() -> Iterator:
    stdin = io.TextIOWrapper(
        buffer=sys.stdin.buffer,
        encoding="utf-8",
        errors="replace",
    )
    return (line.rstrip("\n") for line in stdin)


def stdout_from_iter(iter: Iterator) -> None:
    stdout = io.TextIOWrapper(
        buffer=sys.stdout.buffer,
        line_buffering=True,
    )
    try:
        for line in iter:
            stdout.write(line + "\n")
    except BrokenPipeError:
        sys.exit(1)


def fileout_from_iter(iter: Iterator, fp: TextIO) -> None:
    for line in iter:
        fp.write(line + "\n")
