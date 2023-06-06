import json

from hojichar import Compose, Document, Filter
from hojichar.filters.document_filters import JSONLoader


class JSONDumper(Filter):
    def apply(self, document: Document) -> Document:
        text = document.text
        document.text = json.dumps({"text": text}, ensure_ascii=False)
        return document


class AddComment(Filter):
    def __init__(self, comment: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comment = comment

    def apply(self, document: Document) -> Document:
        text = document.text
        text += self.comment
        document.text = text
        return document


def callback(comment1, comment2):
    return Compose([JSONLoader(), AddComment(comment1), AddComment(comment2), JSONDumper()])


FACTORY = callback
