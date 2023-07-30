import io
import json

import pytest

from hojichar.utils.io_iter import fileout_from_iter, stdin_iter, stdout_from_iter


class OpenBytesIO(io.BytesIO):
    def close(self):
        pass


class CaptureStdout(io.TextIOWrapper):
    def __init__(self):
        super().__init__(OpenBytesIO(), encoding="utf-8", newline="", write_through=True)

    def getvalue(self):
        return self.buffer.getvalue().decode("utf-8")

    @property
    def out(self):
        return self.getvalue()


@pytest.fixture
def mock_stdin(monkeypatch):  # Mock utf-8 strings into stdin.
    def _mock_stdin(test_data: str):
        mock_buffer = io.BytesIO(test_data.encode("utf-8"))
        mock_stdin = io.TextIOWrapper(buffer=mock_buffer, encoding="utf-8", errors="replace")
        monkeypatch.setattr("sys.stdin", mock_stdin)

    return _mock_stdin


@pytest.fixture
def mock_stdin_bytes(monkeypatch):  # Mock bytes into stdin.
    def _mock_stdin(test_data_bytes: bytes):
        mock_buffer = io.BytesIO(test_data_bytes)
        mock_stdin = io.TextIOWrapper(buffer=mock_buffer, encoding="utf-8", errors="replace")
        monkeypatch.setattr("sys.stdin", mock_stdin)

    return _mock_stdin


@pytest.fixture
def capture_stdout(monkeypatch):
    def _capture_stdout():
        capture = CaptureStdout()
        monkeypatch.setattr("sys.stdout", capture)
        return capture

    return _capture_stdout


@pytest.mark.parametrize(
    "test_data,expected_output",
    [
        ("Line1\nLine2\nLine3", ["Line1", "Line2", "Line3"]),
        ("Line1\r\nLine2\r\nLine3", ["Line1", "Line2", "Line3"]),
        ("Line1\rLine2\rLine3", ["Line1", "Line2", "Line3"]),
        ("Line1\nLine2\nLine3\n", ["Line1", "Line2", "Line3"]),
        ("Line1\n\nLine2\n\nLine3", ["Line1", "", "Line2", "", "Line3"]),
        ("", []),
        ("\n", [""]),
        ("\n\n", ["", ""]),
        ("SingleLine", ["SingleLine"]),
        ("SingleLine\n", ["SingleLine"]),
        ("SingleLine\r\n", ["SingleLine"]),
        ("SingleLine\r", ["SingleLine"]),
        (
            "\n".join([json.dumps({"text": "HojiChar"}) for _ in range(10)]),
            ['{"text": "HojiChar"}' for _ in range(10)],
        ),
    ],
)
def test_stdin_iter(mock_stdin, test_data, expected_output):
    mock_stdin(test_data)
    result = list(stdin_iter())
    assert result == expected_output


@pytest.mark.parametrize(
    "test_data, expected_outut",
    [
        (b"Line1\nLine2\nLine3", ["Line1", "Line2", "Line3"]),
        (b"\xff", ["�"]),
        ("HojiChar".encode("utf-8"), ["HojiChar"]),
        ("HojiChar".encode("Shift-JIS"), ["HojiChar"]),
        ("ほうじ茶".encode("Shift-JIS"), ["�ق�����"]),
    ],
)
def test_stdin_iter_invalid_unicode_bytes(mock_stdin_bytes, test_data, expected_outut):
    mock_stdin_bytes(test_data)
    result = list(stdin_iter())
    assert result == expected_outut


# pytest crush when use capfd fixture. This may be a pytest bug
# Since this function is different from the usual standard output that uses print, etc.,
# it is possible that pytest has run into a bug that it does not anticipate.
# HACK I implement fixture capture_stdout, and work it.
"""
@pytest.mark.parametrize(
    "test_data, expected_output",
    [
        (["Line1", "Line2", "Line3"], "Line1\nLine2\nLine3\n"),
        ([], "\n"),
        (["\n"], "\n\n"),
        (["SingleLine"], "SingleLine\n"),
        (["SingleLine\n"], "SingleLine\n\n"),
        (["hoge"], ""),
    ],
)
def test_stdout_from_iter(capfd, test_data, expected_output):
    stdout_from_iter(test_data)
    out, err = capfd.readouterr()
    assert out == expected_output
"""


@pytest.mark.parametrize(
    "test_data, expected_output",
    [
        (["Line1", "Line2", "Line3"], "Line1\nLine2\nLine3\n"),
        ([], ""),
        (["\n"], "\n\n"),
        (["SingleLine"], "SingleLine\n"),
        (["SingleLine\n"], "SingleLine\n\n"),
    ],
)
def test_stdout_from_iter(capture_stdout, test_data, expected_output):
    capture = capture_stdout()
    stdout_from_iter(test_data)
    assert capture.out == expected_output


@pytest.mark.parametrize(
    "test_data, expected_output",
    [
        (["Line1", "Line2", "Line3"], "Line1\nLine2\nLine3\n"),
        ([], ""),
        (["\n"], "\n\n"),
        (["SingleLine"], "SingleLine\n"),
        (["SingleLine\n"], "SingleLine\n\n"),
    ],
)
def test_fileout_from_iter(test_data, expected_output):
    fp = io.StringIO()
    fileout_from_iter(fp, test_data)
    assert fp.getvalue() == expected_output
