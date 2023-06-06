# HojiChar

[![PyPI version](https://badge.fury.io/py/hojichar.svg)](https://badge.fury.io/py/hojichar)
[![Python Versions](https://img.shields.io/pypi/pyversions/hojichar.svg)](https://pypi.org/project/hojichar/)
[![CI wowkflow](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml/badge.svg)](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/HojiChar/HojiChar/branch/main/graph/badge.svg?token=16928I9U9Y)](https://codecov.io/gh/HojiChar/HojiChar)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hojichar)

Official docs: <https://hojichar.github.io/HojiChar/hojichar.html>

## 概要

HojiChar はテキストデータの前処理のためのPythonモジュールです. 言語モデル構築時などにコーパスを前処理する目的で開発されました。

`hojichar.filters` で定義された、あるいはユーザーが定義したテキスト前処理フィルタを束ね、ひとつの前処理パイプラインとして構成することができように作られています。

この前処理パイプラインは、`torchvision.transforms` に着想を得て開発されました。

## 使い方

### インストール

```
pip install hojichar
```

### CLI tool and preprocessing profile

- HojiChar provides CLI tools for text preprocess pipeline.
- User defines a series of preprocessing into a python file as profile.

#### Usage

- Example:

  ```bash
  cat <your_text.jsonl> | hojichar -p your_preprocessing_profile.py -o your_text_preprocessed.jsonl
  ```

- See `hojichar --help`

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

### Definition of preprocessing profile

- HojiChar CLI receives a series of preprocessing as a profile.
- The preprocessing profile is provided as a Python file. Two patterns of the file are allowed.

#### `FILTER` profile

- `hojichar.Compose` must be defined as `FILTER` variable in the file.
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

#### `FACTORY` profile

- An alias for the function which returns `hojichar.Compose` is defined as `FACTORY` variable in the file.
- Parameters can be passed to the pre-processing profile.
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

  FACTORY = callback
  ```
  
  - Using `FACTORY` profile with arguments in CLI.

      ```bash
      cat <your_file> | hojichar -p example_profile.py --args hello
      ```

### フィルタ定義

`Compose` クラスを使ってフィルタを作成します.

```Python
from hojichar import Compose, document_filters

cleaner = Compose([
    document_filters.JSONLoader(key="text"),
    document_filters.AcceptJapanese(),
    document_filters.DocumentLengthFilter(min_doc_len=0,max_doc_len=1000),
    document_filters.ExampleHojiChar()
])
```

```
>>> cleaner('{"text": "こんにちは、"}')
'こんにちは、<hojichar>'
```

上記のフィルタでは 1. JSONから`'text'` キーの値を取得 2. 日本語文字列でなければ破棄, 3. 0字以上1000字以内の文章以外を破棄, 4. 文字列に `<hojichar>` を追加 の処理をしています.

定義済みのフィルタは、`hojichar.filters` の各フィルタです。

### ユーザー定義フィルタ

`Filter` クラスを継承し, `apply` 関数内にフィルタの挙動を記述します.

```Python
from hojichar.core.filter_interface import Filter

class YourFilter(Filter):
    def apply(self, document):
        document.text = your_process(document.text)
        return document
```

`apply` 関数は `hojichar.core.models.Document` 型を引数として受け取り,
返す関数です. `Document` は文字列をカプセル化したクラスです.

## 開発者向け

**Poetry によるローカルインストール**

`python >= 3.8, poetry >= 1.2`

```
https://github.com/HojiChar/HojiChar.git
cd HojiChar
poetry install
```

開発用のパッケージのインストールのために,

```
poetry install --with dev,lint,test
```

### テスト

テスト実行

```
pytest --doctest-modules .
```

で mypy と pytest のテストが実行されます.

Lint

```
poetry run task lint
```

Format

```
poetry run task format
```
