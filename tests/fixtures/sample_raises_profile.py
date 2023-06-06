from hojichar import Compose, Document, Filter


class RaiseKeywords(Filter):
    def apply(self, document: Document) -> Document:
        text = document.text
        if "<raise>" in text:
            raise
        return document


FILTER = Compose([RaiseKeywords()])
