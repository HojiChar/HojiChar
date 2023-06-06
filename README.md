# HojiChar

[![PyPI version](https://badge.fury.io/py/hojichar.svg)](https://badge.fury.io/py/hojichar)
[![Python Versions](https://img.shields.io/pypi/pyversions/hojichar.svg)](https://pypi.org/project/hojichar/)
[![CI wowkflow](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml/badge.svg)](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/HojiChar/HojiChar/branch/main/graph/badge.svg?token=16928I9U9Y)](https://codecov.io/gh/HojiChar/HojiChar)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hojichar)

Official docs: <https://hojichar.github.io/HojiChar/hojichar.html>

## Background and what is for HojiChar

Text preprocessing is far from a one-size-fits-all process. Depending on the data source and the specific task at hand, various steps including normalization, noise removal, and filtering may be necessary. Not all texts require the same level of preprocessing. For instance, relatively clean texts may only need minimal filtering, while "dirtier" sources like Common Crawl data often require more thorough processing. As a result, the preprocessing profile has to be tailored to each specific domain.

Many preprocessing operations can be viewed as filters, taking string as input, applying a transformation, and outputting the processed string. Even though these operations might seem straightforward individually, managing them in a multi-layered, efficient manner can be challenging.

Inspired by [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html) and [iver56/audiomentations](https://github.com/iver56/audiomentations), HojiChar addresses these challenges. It enables users to define each text processing step as a class inheriting from `hojichar.Filter` and use `hojichar.Compose` to chain them together into a single filter. By writing out the `Compose` recipe as a profile, the preprocessing process for a specific domain's text can be made portable. Moreover, `Compose` automatically logs various metrics for each filter, such as byte changes, processing time, and number of rejected texts. This allows users to assess the validity of each operation and consider trade-offs between computation time and performance.

While there are other text normalization tools available, most are designed to perform a specific set of operations. Text preprocessing, despite its importance, is often considered a mundane task compared to machine learning or artificial intelligence tasks. As a result, many existing solutions can be ad hoc, poorly maintained, or inadequately tested. Recognizing these issues, we developed HojiChar as a robust tool for configuring text preprocessing.

## Install

```
pip install hojichar
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

## Additional Notes on Compose

- Even though the behavior of a `Compose` object when called is a text-in, text-out function, `Compose` itself also inherits from the `Filter` class. Therefore, applying the `apply` method to a `Compose` object results in `hojihcar.Document` class being used as input and output.
- You can access various statistics regarding the processing performed by `Compose` through `Compose.statistics`, which returns a dictionary.

It might be helpful to add examples demonstrating the use of `Compose

## CLI tool and preprocessing profile

- HojiChar provides CLI tools for text preprocess pipeline.
- User defines a series of preprocessing into a python file as profile.

- Example:

  ```bash
  cat <your_text.jsonl> | hojichar -p your_preprocessing_profile.py -o your_text_preprocessed.jsonl
  ```

- `hojichar --help`

  ```man
  usage: hojichar [-h] --profile <your_filter.py> [--output OUTPUT] [--dump-stats <path to stats.json>] [--exit-on-error] [--args ARGS [ARGS ...]]
  
  options:
    -h, --help            show this help message and exit
    --profile <your_filter.py>, -p <your_filter.py>
                          Path to a Python file that implements your custom filter. hojichar.Compose must be defined as FILTER variable in the file.
    --output OUTPUT, -o OUTPUT
                          Output file path. If not given, stdout is used.
    --dump-stats <path to stats.json>
                          Dump statistics to a file.
    --exit-on-error       Exit if an exception occurs during filtering. Useful for debugging custom filters.
    --args ARGS [ARGS ...]
                          Argument for the profile which receives arguments.
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

**Local Installation with Poetry**

Requirements: `python >= 3.8, poetry >= 1.2`

To install the package, run the following commands:

```
git clone https://github.com/HojiChar/HojiChar.git
cd HojiChar
poetry install
```

For installing development-related packages, you can run:

```
poetry install --extras "dev lint test"
```

### Testing

You can run the tests with:

```
pytest --doctest-modules .
```

This will execute both mypy and pytest tests.

Linting can be done using:

```
poetry run task lint
```

And for formatting:

```
poetry run task format
```
