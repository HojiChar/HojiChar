# HojiChar: The Text Processing Pipeline

[![PyPI version](https://badge.fury.io/py/hojichar.svg)](https://badge.fury.io/py/hojichar)
[![Python Versions](https://img.shields.io/pypi/pyversions/hojichar.svg)](https://pypi.org/project/hojichar/)
[![CI wowkflow](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml/badge.svg)](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/HojiChar/HojiChar/branch/main/graph/badge.svg?token=16928I9U9Y)](https://codecov.io/gh/HojiChar/HojiChar)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hojichar)

Official docs: <https://hojichar.github.io/HojiChar/hojichar.html>

## Features

- HojiChar provides a way to combine multiple arbitrary text processing tasks into a streamlined pipeline.
- The sequence of operations can be described declaratively, ensuring portability.
- HojiChar allows users to gather detailed statistical information from large amounts of text during processing.
- It enables management of any Python text processing tasks, providing a Command Line Interface (CLI) capable of parallel processing.

## Background and what is for HojiChar

Text preprocessing is far from a one-size-fits-all process. Depending on the data source and the specific task at hand, various steps including normalization, noise removal, and filtering may be necessary. Not all texts require the same level of preprocessing. For instance, relatively clean texts may only need minimal filtering, while "dirtier" sources like Common Crawl data often require more thorough processing. As a result, the preprocessing profile has to be tailored to each specific domain.

Many preprocessing operations can be viewed as filters, taking string as input, applying a transformation, and outputting the processed string. Even though these operations might seem straightforward individually, managing them in a multi-layered, efficient manner can be challenging.

Inspired by [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) and [iver56/audiomentations](https://github.com/iver56/audiomentations), HojiChar addresses these challenges. It enables users to define each text processing step as a class inheriting from `hojichar.Filter` and use `hojichar.Compose` to chain them together into a single filter. By writing out the `Compose` recipe as a profile, the preprocessing process for a specific domain's text can be made portable. Moreover, `Compose` automatically logs various metrics for each filter, such as byte changes, processing time, and number of rejected texts. This allows users to assess the validity of each operation and consider trade-offs between computation time and performance.

While there are other text normalization tools available, most are designed to perform a specific set of operations. Text preprocessing, despite its importance in the LLM era, is often considered a mundane task compared to machine learning or artificial intelligence tasks. As a result, many existing solutions can be ad hoc, poorly maintained, or inadequately tested. Recognizing these issues, we developed HojiChar as a robust tool for configuring text preprocessing.

## Install

```
pip install hojichar
```

If you want to use the additional filters, install the package with the following command:

```
pip install 'hojichar[all]'
```

If you want to use `AsyncChatAPI` filter, install the package with the following command:

```
pip install 'hojichar[openai]'
```

If you want to use the near-deduplication (using MinHash LSH algorithm) filter, install the package with the following command:

```
pip install 'hojichar[dedup]'
```

## Defining a Compose Object

The [`Compose`](https://hojichar.github.io/HojiChar/hojichar.html#Compose) class in HojiChar allows you to create a sequence of text processing filters.

```Python
from hojichar import Compose, document_filters

cleaner = Compose([
    document_filters.JSONLoader(key="text"),
    document_filters.AcceptJapanese(),
    document_filters.DocumentLengthFilter(min_doc_len=0,max_doc_len=1000),
    document_filters.ExampleHojiChar(),
    document_filters.JSONDumper()
])
```

When a [`Compose`](https://hojichar.github.io/HojiChar/hojichar.html#Compose) object is called, it accepts a string and returns the processed string.

```Python
>>> cleaner('{"text": "こんにちは、"}')
{"text": "こんにちは、<hojichar>"}
```

The filter pipeline above accomplishes the following steps:

1. Extracts the value from the `'text'` key in the JSON object.
2. Discards the string if it's not in Japanese.
3. Rejects any text shorter than 0 characters or longer than 1000 characters.
4. Appends `<hojichar>` to the string.
5. Outputs the processed string as JSON with the key "text".

The filters used in the pipeline are predefined filters found in [`hojichar.filters`](https://hojichar.github.io/HojiChar/hojichar/filters.html).

While HojiChar provides some fundamental text processing filters and plans to add more in the future, users can also define their custom filters.

## User-defined Filters

A filter composing a [`Compose`](https://hojichar.github.io/HojiChar/hojichar.html#Compose) object is a class that inherits the [`Filter`](https://hojichar.github.io/HojiChar/hojichar.html#Filter) class and implements the text processing within the `apply` function.

```Python
from hojichar.core.filter_interface import Filter

class YourFilter(Filter):
    def apply(self, document):
        text = document.text
        """
        Write your text transformation...
        """
        document.text = text
        return document
```

The `apply` method accepts a `hojichar.Document` type as an argument and returns it after the transformations. The [`Document`](https://hojichar.github.io/HojiChar/hojichar.html#Document) is a class that encapsulates a string.

The Document class can have additional metadata via the extras attribute. This allows you to associate values with the document that can be utilized in subsequent filters.
**Reject documents**

- The `hojichar.Document` has an `is_rejected` attribute. If a filter sets this flag to `True`, `Compose` will discard the document during processing.

**Definition of `__init__` for custom filter**

When creating a user-defined class and applying a custom constructor, make sure to initialize the parent class.

```python
class YourFilter(Filter):
    def __init__(self, your_param, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.your_param = your_param

    def apply(self, document):
        text = document.text
        text = process(text, self.your_param)
        document.text = text
        return document
```

This is because The `Filter` class implicitly has several arguments, one of which is `p`.

```python
cleaner = Compose([
    document_filters.JSONLoader(key="text"),
    document_filters.AcceptJapanese(p=0.5),
    document_filters.JSONDumper()
])
```

The `p` argument passed to the `document_filters.AcceptJapanese` constructor determines the probability of applying the filter; with a probability of `1-p`, it acts as an identity function. This behavior is defined in the parent class `hojichar.Filter`.

## Batch and Stream Processing with `apply_batch` and `apply_stream`

The `Filter` and `Compose` classes support efficient batch and stream processing through the `apply_batch` and `apply_stream` methods.

### `apply_batch`

- The `apply_batch` method processes a list of `Document` objects in one go. By default, it applies the `apply` method to each document individually.
- Users can override `apply_batch` in custom filters for optimized batch operations.

    ```python
    class YourBatchFilter(Filter):
        def apply_batch(self, documents: Sequence[Document]) -> list[Document]:
            # Implement your batch processing logic here
            return documents
    ```


### `apply_stream`

The `apply_stream` method processes an iterable (e.g., generator) of `Document` objects, ideal for large datasets or stream-based processing. If the `use_batch` flag is set to `True` in a `Filter`'s constructor, its apply_batch implementation will be utilized during stream processing.

**Example Usage:**

```python
stream = (Document(f"text {i}") for i in range(10000))
processed_stream = cleaner.apply_stream(stream)

for doc in processed_stream:
    print(doc.text)
```

This allows HojiChar to efficiently process massive corpora while maintaining low memory consumption.

## Additional Notes on Compose

- Even though the behavior of a `Compose` object when called is a text-in, text-out function, `Compose` itself also inherits from the `Filter` class. Therefore, applying the `apply` method to a `Compose` object results in `hojihcar.Document` class being used as input and output.
- `Compose` class behaves like a Filter. If you add a Compose object as one of the filters in the constructor of Compose, the filter will be unfolded recursively.

## HojiChar running asynchronously

- HojiChar supports asynchronous processing of text data using the `AsyncCompose` class. This allows you to build pipelines that can handle out-of-CPU processing, such as making API calls.
- You can define async versions of filter using the `AsyncFilter` class.

    ```python
    from hojichar import AsyncFilter

    class YourAsyncFilter(AsyncFilter):
        async def apply(self, document):
            text = document.text
            # Perform asynchronous processing here
            document.text = text
            return document
    ```

- The `AsyncCompose` class accepts both `Filter` and `AsyncFilter` objects, allowing you to mix synchronous and asynchronous filters in a single pipeline.

### Example

Nowadays, text processing is enhanced by the intelligence of LLMs.

This example demonstrates how to use the `AsyncChatAPI` filter to process text data with OpenAI compatible APIs. This filter allows you to build high throughput of "Chain of LLMs" easily.

```python
import os

from hojichar import AsyncCompose
from hojichar.filters.document_filters import JSONLoader, JSONDumper
from hojichar.utils.async_handlers import write_stream_to_file


async_pipeline = AsyncCompose(
    [
        JSONLoader(input_key="text"),
        AsyncChatAPI(
            model_id="gpt-4o",
            openai_endpoint_url="https://api.openai.com/v1", 
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_concurrent_requests=128,
            output_key="llm_output",
            message_generator=lambda doc: [{"role": "user", "content": doc.text[:1000]}],
        ),
        JSONDumper(export_extras=True),
    ]
)

with open("input.jsonl") as f:
    with async_pipeline:
        async_output_stream = (str(doc) async for doc in async_pipeline.apply_stream(f))
        await write_stream_to_file(async_output_stream, "output.jsonl", chunk_size=128) # Write async-iterable to file efficiently
```

- You can use this filter by installing `'hojichar[openai]'`
- The filter works with OpenAI compatible APIs, like the endpoint hosted by vLLM. It's useful for text-augumentation tasks.
  - The AsyncChatAPI works 1K req/sec with optimized vLLM server. (We reccomend to use `uvloop` to get better throughput.)

## Get Metrics of processing

HojiChar tracks detailed statistics at both the filter and pipeline levels, helping you monitor and debug your processing pipeline.

Each `Filter` (including `Compose`) maintains a `Statistics` object containing information such as input size, output size, discarded document count, and processing time.

**Example: Getting Statistics from a Compose Object**

```python
stats = cleaner.get_total_statistics_map()
print(stats)
```

```python
[{'cumulative_time_ns': 337250,
  'diff_bytes': 10,
  'diff_chars': 10,
  'discard_num': 0,
  'input_bytes': 45,
  'input_chars': 23,
  'input_num': 1,
  'name': 'Total',
  'output_bytes': 55,
  'output_chars': 33,
  'output_num': 1},
 {'cumulative_time_ns': 80209,
  'diff_bytes': -12,
  'diff_chars': -12,
  'discard_num': 0,
  'input_bytes': 45,
  'input_chars': 23,
  'input_num': 1,
  'name': '0-JSONLoader',
  'output_bytes': 33,
  'output_chars': 11,
  'output_num': 1},
 {'cumulative_time_ns': 17500,
  'diff_bytes': 0,
  'diff_chars': 0,
  'discard_num': 0,
  'input_bytes': 33,
  'input_chars': 11,
  'input_num': 1,
  'name': '1-AcceptJapanese',
  'output_bytes': 33,
  'output_chars': 11,
  'output_num': 1},
 {'cumulative_time_ns': 8125,
  'diff_bytes': 0,
  'diff_chars': 0,
  'discard_num': 0,
  'input_bytes': 33,
  'input_chars': 11,
  'input_num': 1,
  'name': '2-DocumentLengthFilter',
  'output_bytes': 33,
  'output_chars': 11,
  'output_num': 1},
 {'cumulative_time_ns': 6042,
  'diff_bytes': 10,
  'diff_chars': 10,
  'discard_num': 0,
  'input_bytes': 33,
  'input_chars': 11,
  'input_num': 1,
  'name': '3-ExampleHojiChar',
  'output_bytes': 43,
  'output_chars': 21,
  'output_num': 1},
 {'cumulative_time_ns': 81125,
  'diff_bytes': 12,
  'diff_chars': 12,
  'discard_num': 0,
  'input_bytes': 43,
  'input_chars': 21,
  'input_num': 1,
  'name': '4-JSONDumper',
  'output_bytes': 55,
  'output_chars': 33,
  'output_num': 1}]
```

- Use `get_statistics()` to get the raw Statistics object for any filter.
- Use `get_total_statistics()` to get a list of statistics for all filters in a Compose pipeline.
- Use `get_total_statistics_map()` to retrieve the statistics as a list of dicts.

These tools allow granular monitoring of how each filter contributes to data reduction, rejection, or transformation.

## Parallel application of `Compose`

The `hojichar.Parallel` class allows for the application of `Compose` to an iterable of `Document` concurrently. This class empowers users to process vast collections of documents by harnessing the power of multiple CPU cores.

Example usage of `Parallel` class to proces a very large JSON Lines file concurrently.

```python
import hojichar

input_file = "your_text.jsonl"
input_doc_iter = (hojichar.Document(line) for line in open(input_file))

cleaner = hojichar.Compose([
    hojichar.document_filters.JSONLoader(),
    hojichar.document_filters.DocumentNormalizer(),
    # Insert your filters
    hojichar.document_filters.JSONDumper(),
])

with hojichar.Parallel(cleaner, num_jobs=10) as pfilter:
    out_doc_iter = pfilter.imap_apply(input_doc_iter)
    with open("your_processed_text.jsonl", "w") as fp:
        for doc in out_doc_iter:
            fp.write(doc.text + "\n")

```

- Always use the `Parallel` class within a `with` statement.
- `Parallel.imap_apply(doc_iter)` processes an iterator of `Document` and returns an iterator of the processed documents.
- For additional options and details about the `Parallel` class, please refer to the official documentation.

## CLI tool and preprocessing profile

- HojiChar provides CLI tools for text preprocess pipeline.
- User defines a series of preprocessing into a python file as profile.

- Example:

  ```bash
  cat <your_text.jsonl> | hojichar -p your_preprocessing_profile.py -o your_text_preprocessed.jsonl
  ```

- `hojichar --help`

  ```man
    usage: hojichar [-h] --profile <profile.py> [--args ARGS [ARGS ...]] [--output OUTPUT] [--input INPUT] [--dump-stats <path to stats.json>] [--exit-on-error] [--all] [--jobs JOBS]

    options:
    -h, --help            show this help message and exit
    --profile <profile.py>, -p <profile.py>
                            Path to a Python file that implements your custom filter.
    --args ARGS [ARGS ...]
                            Pass additional arguments to the profile. Use it like `--args arg1 arg2` etc. The arguments should be space-separated.
    --output OUTPUT, -o OUTPUT
                            Specifies the path for the output file. Defaults to standard output.
    --input INPUT, -i INPUT
                            Specifies the path for the input file. Defaults to standard input. If set this path, the progress bar is enabled.
    --dump-stats <path to stats.json>
                            Dump statistics to file. If the file exists, it will be appended.
    --exit-on-error       Exit if an exception occurs during filtering. Useful for debugging custom filters.
    --all                 A flag that specifies whether to include discarded samples. This is useful when inspecting discarded samples.
    --jobs JOBS, -j JOBS  The number ob parallel jobs. By default, the nuber of the CPU core.
  ```

## Definition of Profile

- HojiChar CLI receives a series of preprocessing as a profile.
- The preprocessing profile is provided as a Python file. Two patterns of the file are allowed.
- hojichar.utils.load_compose.load_compose() loads these profile.

### `FILTER` profile

- `hojichar.Compose` must be defined as `FILTER` variable.
- Example.

    ```python
    import json
    
    from hojichar import Compose, Filter
    from hojichar.filters.document_filters import ExampleHojiChar, JSONLoader
    
    
    class JSONDumper(Filter):
        def apply(self, document):
            text = document.text
            document.text = json.dumps({"text": text}, ensure_ascii=False)
            return document
    
    # FILTER must define Compose object.
    FILTER = Compose(
        [
            JSONLoader(),
            ExampleHojiChar(),
            JSONDumper(),
        ]
    )
    ```

  - Pass the texts to the filter you have defined using a pipe as follows.

    ```bash
    cat <your_file> | hojichar -p example_profile.py
    ```

- `hojichar.utils.load_compose.load_filter_from_file()` loads this type of profile.

### `FACTORY` profile

- A callable function that returns `hojichar.Compose` must be defined as `FACTORY` variable.
- The callable can receive arguments. In this way, parameters can be passed to the profile.
  - Some kinds of value are not preferred to static. For example, random seeds and some flags modify the behavior of a filter, etc
  - `FACTORY` provides a mechanism to pass those values as arguments to the preprocessing.
- Example.

  ```python
  import json
  
  from hojichar import Compose, Filter
  from hojichar.filters.document_filters import JSONLoader
  

  class AddSomething(Filter): #  Concat some value after every document.
      def __init__(self, something: str, *args, **kwargs) -> None:
          self.something = something

      def apply(self, document):
          text = document.text + self.something
          document.text = text
          return document

  class JSONDumper(Filter):
      def apply(self, document):
          text = document.text
          document.text = json.dumps({"text": text}, ensure_ascii=False)
          return document
  
  
  def callback(something):
      return Compose(
          [
              JSONLoader(),
              AddSomething(something),
              JSONDumper(),
          ]
      )
  
  # FACTORY must be callable which returns Compose object.
  FACTORY = callback
  ```

- Using `FACTORY` profile with arguments in CLI.

    ```bash
    cat <your_file> | hojichar -p example_profile.py --args arg1 arg2
    ```

- `hojichar.utils.load_compose.load_parametrized_filter_from_file()` or `load_factory_from_file` loads this type of profile.

## For Developers

### Installing from the Source Directory

To install the package, execute the following commands:

```
git clone https://github.com/HojiChar/HojiChar.git
cd HojiChar
uv sync --all-extras
```

### Testing

Some filters incorporate doctests. You can run these tests with the command:

```
pytest --doctest-modules .
```

This command should be executed from the root of the project.

### Code style

- HojiChar requires type hints for all code. Type checking is performed in continuous integration (CI) in addition to the pytest tests.
- HojiChar code is subject to inspection and formatting by the `ruff` Linter. For configuration details, please refer to `pyproject.toml`. You can perform linting and formatting from the root of the project using the following commands:

Linting

```bash
uvx ruff check .
```

Formatting

```bash
uvx ruff format .
```

### Building the Documentation

We use Pdoc for building the documentation. You can build the documentation using the following command:

```
pdoc -o docs hojichar
```

Run this command from the project root.

In practice, the process of building the documentation is automated by CI. When a Pull Request is merged into the main branch, the documentation is built in the `docs/` directory of the `docs` branch. This directory is then deployed to the official documentation site by GitHub Pages.

### Creating a Source Tarball

To create a source tarball, for instance, for packaging or distribution, run the following command:

```bash
uv build
```

The tarball will be created in the dist directory. This command will compile the source code, and the resulting tarball can be installed with no additional dependencies other than the Python standard library.

### Creating a Release and Uploading it to PyPI

Versions uploaded to PyPI are identified by git tags. The `__version__` variable in `__init__.py` or the `version` entry in `pyproject.toml` are ignored. The `uv-dynamic-versioning` plugin is used to implement this process.

The steps to push to PyPI are as follows, although in actuality, the process is automated by CI when a GitHub release is created from the tag.

```bash
git checkout v0.1.2
uv build
uv publish --index testpypi --token ${PYPI_TOKEN}
```

The actual task for the manager is to apply the appropriate tag to the commit to be released and to create the release from GitHub:

```bash
git tag -a v0.1.2 -m "Version 0.1.2"
git push origin v0.1.2
```
