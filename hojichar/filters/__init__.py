# flake8: noqa
"""
Filters for text preprocess.

Each filter module can directly import from `hojichar`. i.e., you can inport as `from hojichar import document_filters`.

How we classify each filter is below:
- `hojichar.filters.document_filters`-- General text cleaners. 
- `hojichar.filters.deduplication`-- Approximate deduplicate processor, inspired by NEARDUP from https://arxiv.org/abs/2107.06499
- `hojichar.filters.token_filters`-- A per-token filter. For example, to process a specific part of speech.
- `hojichar.filters.tokenization`-- Tokenizer, which splits texts into tokens. Here, a token is an arbitrary unit for splitting a sentence and processing it with `hojichar.filters.token_filters`.
"""
