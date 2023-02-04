# HojiChar
[![PyPI version](https://badge.fury.io/py/hojichar.svg)](https://badge.fury.io/py/hojichar)
[![Python Versions](https://img.shields.io/pypi/pyversions/hojichar.svg)](https://pypi.org/project/hojichar/)
[![CI wowkflow](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml/badge.svg)](https://github.com/HojiChar/HojiChar/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/HojiChar/HojiChar/branch/main/graph/badge.svg?token=16928I9U9Y)](https://codecov.io/gh/HojiChar/HojiChar)

## 概要
HojiChar はテキストデータの前処理のためのPythonモジュールです. 言語モデル構築時などにコーパスを前処理する目的で開発されました。

`hojichar.filters` で定義された、あるいはユーザーが定義したテキスト前処理フィルタを束ね、ひとつの前処理パイプラインとして構成することができように作られています。

この前処理パイプラインは、`torchvision.transforms` に着想を得て開発されました。


## 使い方
### インストール
```
pip install hojichar
```

**Poetry によるローカルインストール**

`python >= 3.8, poetry >= 1.2`

```
https://github.com/HojiChar/HojiChar.git
cd HojiChar
poetry install
```


### Rocket start
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
