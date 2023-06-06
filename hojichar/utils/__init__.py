# flake8: noqa
"""
Utilities mainly used by implementation of the CLI tool, but some are useful for users.

- `hojichar.utils.process` -- This utility applies filters to the input text iteratively.
- `hojichar.utils.load_compose` --   This utility allows users to load `hojichar.Compose` objects from a user-defined file. The profile specifications are described in the `Text Processing Profile` section.
- `hojichar.utils.io_iter` -- This is an IO wrapper used to handle standard input/output and file output in an iterator-oriented manner, streamlining the processing of large text files.
"""
