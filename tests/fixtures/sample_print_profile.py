from hojichar import Compose, Filter


class DummyPrint(Filter):
    def apply(self, document):
        print("dummy print")
        return document


FILTER = Compose([DummyPrint()])
