import hojichar
from hojichar.core.models import Document


class SetValue(hojichar.Filter):
    def __init__(self, value: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value = value

    def apply(self, document: Document) -> Document:
        """
        >>> SetValue("success")("")
        'success'
        """
        document.text = self.value
        return document


def factory(value):
    return hojichar.Compose([SetValue(value)])


FACTORY = factory
