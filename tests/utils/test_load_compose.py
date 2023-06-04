import os
import shutil
from pathlib import Path

import pytest

from hojichar.utils.load_compose import (
    _check_args_num_mismatch,
    _load_module,
    load_compose,
    load_factory_from_file,
    load_filter_from_file,
    load_parametrized_filter_from_file,
)


@pytest.fixture
def mock_dir() -> Path:
    mock_dir = Path(__file__).parent / "mock_profiles"
    return mock_dir


def test_load_module(mock_dir):
    fpath = mock_dir / "mock_loading_verification.py"
    module = _load_module(fpath)
    assert module.IS_LOADED == "success"


def test_load_filter_from_file_success(mock_dir):
    fpath = mock_dir / "mock_filter_success.py"
    filter = load_filter_from_file(fpath)
    assert filter("") == "success"


def test_load_module_load_another(mock_dir):
    # HACK doctest loads *.py file and cause ModuleNotFoundError.
    original = mock_dir / "mock_filter_load_another_module"
    fpath = mock_dir / "mock_filter_load_another_module.py"
    shutil.copyfile(original, fpath)
    filter = load_filter_from_file(fpath)
    assert filter("") == "success"
    os.remove(fpath)


def test_load_filter_from_file_notimplemented(mock_dir):
    fpath = mock_dir / "mock_filter_notimplemented.py"
    with pytest.raises(NotImplementedError) as e:
        load_filter_from_file(fpath)
    assert str(e.value) == "FILTER is not defined in the profile."


def test_load_filter_from_file_typeerror(mock_dir):
    fpath = mock_dir / "mock_filter_typeerror.py"
    with pytest.raises(TypeError) as e:
        load_filter_from_file(fpath)
    assert str(e.value) == "FILTER must be hojichar.Compose object."


def test_load_factory_from_file_success(mock_dir):
    fpath = mock_dir / "mock_factory_success.py"
    factory = load_factory_from_file(fpath)
    filter = factory("success")
    assert filter("") == "success"


def test_load_factory_from_file_notimplemented(mock_dir):
    fpath = mock_dir / "mock_factory_notimplemented.py"
    with pytest.raises(NotImplementedError) as e:
        load_factory_from_file(fpath)
    assert str(e.value) == "FACTORY is not defined in the profile"


def test_load_parametrized_filter_0args(mock_dir):
    fpath = mock_dir / "mock_factory_0args.py"
    args = tuple([])
    filter = load_parametrized_filter_from_file(fpath, *args)
    assert filter("") == "success"


def test_load_parametrized_filter_1args(mock_dir):
    fpath = mock_dir / "mock_factory_success.py"
    args = tuple(["success"])
    filter = load_parametrized_filter_from_file(fpath, *args)
    assert filter("") == "success"


def test_load_parametrized_filter_2args(mock_dir):
    fpath = mock_dir / "mock_factory_2args.py"
    args = tuple(["arg1", "arg2"])
    filter = load_parametrized_filter_from_file(fpath, *args)
    assert filter("") == "arg1+arg2"


def test_load_compose_filter(mock_dir):
    fpath = mock_dir / "mock_filter_success.py"
    filter = load_compose(fpath)
    assert filter("") == "success"


def test_load_compose_filter_unnecessary_args(caplog, mock_dir):
    fpath = mock_dir / "mock_filter_success.py"
    args = tuple(["arg1", "arg2"])
    filter = load_compose(fpath, *args)
    assert filter("") == "success"
    assert "Warning: 2 arguments are ignored." in caplog.text


def test_load_compose_factory(mock_dir):
    fpath = mock_dir / "mock_factory_success.py"
    args = tuple(["success"])
    filter = load_compose(fpath, *args)
    assert filter("") == "success"


def test_check_args_num_mismatch(caplog):
    _check_args_num_mismatch(3)
    assert "Warning: 3 arguments are ignored." in caplog.text
