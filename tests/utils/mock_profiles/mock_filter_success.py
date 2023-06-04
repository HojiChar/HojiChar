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


FILTER = Compose([Success()])
