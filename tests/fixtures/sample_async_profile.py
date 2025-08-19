import json

from hojichar import AsyncCompose, Filter
from hojichar.filters.document_filters import ExampleHojiChar, JSONLoader


class JSONDumper(Filter):
    def apply(self, document):
        text = document.text
        document.text = json.dumps({"text": text}, ensure_ascii=False)
        return document


FILTER = AsyncCompose([JSONLoader(), ExampleHojiChar(), JSONDumper()])
