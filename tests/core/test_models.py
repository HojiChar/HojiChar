from hojichar.core.models import Document


def test_repr() -> None:
    doc = Document("test", extras={"test": "test"})
    assert repr(doc) == "Document(text='test', is_rejected=False, extras={'test': 'test'})"
    assert eval(repr(doc)).text == doc.text
    assert eval(repr(doc)).extras == doc.extras
