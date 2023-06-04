import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import hojichar

logger = logging.getLogger(__name__)


def _load_module(path: str) -> ModuleType:
    path_obj = Path(path)
    module_name = path_obj.stem
    spec = importlib.util.spec_from_file_location(module_name, path_obj)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        raise ModuleNotFoundError(f"Cannot loading {path}")
    return module


def load_compose_from_file(profile_path: str) -> hojichar.Compose:
    """_summary_

    Args:
        profile_path (str): Path to a Python file that implements your custom filter.
        hojichar.Compose must be defined as FILTER variable in the file.

    Raises:
        NotImplementedError:

    Returns:
        hojichar.Compose:
    """
    module = _load_module(profile_path)
    if hasattr(module, "FILTER"):
        filter = getattr(module, "FILTER")
        if not isinstance(filter, hojichar.Compose):
            raise TypeError("FILTER must be hojichar.Compose object.")
        return filter
    else:
        raise NotImplementedError("FILTER is not defined in the profile.")


def load_compose_factory(profile_path: str) -> Callable[[Any], hojichar.Compose]:
    module = _load_module(profile_path)
    if hasattr(module, "FACTORY"):
        factory = getattr(module, "FACTORY")
        return factory
    else:
        raise NotImplementedError("FACTORY is not defined in the profile")


def load_parametrized_compose_from_file(profile_path: str, *factroy_args: str) -> hojichar.Compose:
    factory = load_compose_factory(profile_path)
    filter = factory(*factroy_args)
    return filter


def get_compose(profile_path: str, *factroy_args: str) -> hojichar.Compose:
    try:
        filter = load_compose_from_file(profile_path)
        check_args_num_mismatch(0, len(factroy_args))
        return filter
    except NotImplementedError:
        return load_parametrized_compose_from_file(profile_path, *factroy_args)


def check_args_num_mismatch(expected_num_args: int, actual_num_args: int):
    if expected_num_args != actual_num_args:
        logger.warning(f"{actual_num_args} arguments are ignored.")
