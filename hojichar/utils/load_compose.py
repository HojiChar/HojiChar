import importlib.util
import logging
import sys
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Union

import hojichar

logger = logging.getLogger(__name__)


def _load_module(path: Union[str, PathLike]) -> ModuleType:
    # HACK type hint `os.PathLike[str]` is not allowed in Python 3.8 or older.
    # So I write Union[str, PathLike]
    path_obj = Path(path)
    module_name = path_obj.stem
    spec = importlib.util.spec_from_file_location(module_name, path_obj)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def load_filter_from_file(profile_path: Union[str, PathLike]) -> hojichar.Compose:
    """
    Loading a profile which has `FILTER` variable.
    Given path is added to sys.path automaticaly.

    Args:
        profile_path (str): Path to a Python file that implements custom filter
        hojichar.Compose must be defined as FILTER variable in the file.

    Raises:
        NotImplementedError:

    Returns:
        hojichar.Compose:
    """
    sys.path.append(str(Path(profile_path).parent))
    module = _load_module(profile_path)
    if hasattr(module, "FILTER"):
        filter = getattr(module, "FILTER")
        if not isinstance(filter, hojichar.Compose):
            raise TypeError("FILTER must be hojichar.Compose object.")
        return filter
    else:
        raise NotImplementedError("FILTER is not defined in the profile.")


def load_factory_from_file(
    profile_path: Union[str, PathLike]
) -> Callable[[Optional[Any]], hojichar.Compose]:
    """
    Loading a function by a profile which has `FACTORY` variable.
    Given path is added to sys.path automaticaly.

    Args:
        profile_path (PathLike): Path to a Python file that implements custom filter

    Raises:
        NotImplementedError:

    Returns:
        Callable[[Optional[Any]], hojichar.Compose]:
            An alias of the function which returns Compose.
    """
    sys.path.append(str(Path(profile_path).parent))
    module = _load_module(profile_path)
    if hasattr(module, "FACTORY"):
        factory: Callable[[Optional[Any]], hojichar.Compose] = getattr(module, "FACTORY")
        return factory
    else:
        raise NotImplementedError("FACTORY is not defined in the profile")


def load_parametrized_filter_from_file(
    profile_path: Union[str, PathLike], *factory_args: str
) -> hojichar.Compose:
    factory = load_factory_from_file(profile_path)
    filter = factory(*factory_args)
    return filter


def load_compose(profile_path: Union[str, PathLike], *factroy_args: str) -> hojichar.Compose:
    """
    Loading a Compose file from profile. Loading a Compose file from the profile.
    Both `FILTER` and `FACTORY` pattern of the profile is loaded.

    Args:
        profile_path (PathLike):

    Returns:
        hojichar.Compose:
    """
    try:
        filter = load_filter_from_file(profile_path)
        _check_args_num_mismatch(len(factroy_args))
        return filter
    except NotImplementedError:
        return load_parametrized_filter_from_file(profile_path, *factroy_args)


def _check_args_num_mismatch(num_args: int) -> None:
    if num_args > 0:
        logger.warning(f"Warning: {num_args} arguments are ignored.")
