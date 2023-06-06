import hojichar
from hojichar import Compose


class Success(hojichar.Filter):
    def apply(self, doc: hojichar.Document) -> hojichar.Document:
        """
        >>> Success()("")
        'success'
        """
        doc.text = "success"
        return doc


def factory():
    return Compose([Success()])


FACTORY = factory
