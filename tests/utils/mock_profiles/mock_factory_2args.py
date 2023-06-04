import hojichar
from hojichar.core.models import Document


class SetValue(hojichar.Filter):
    def __init__(self, value1: str, value2: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.value1 = value1
        self.value2 = value2

    def apply(self, document: Document) -> Document:
        """
        >>> SetValue("arg1", "arg2")("")
        'arg1+arg2'
        """
        document.text = "+".join([self.value1, self.value2])
        return document


def factory(value1, value2):
    return hojichar.Compose([SetValue(value1, value2)])


FACTORY = factory
