import inspect
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def deprecated_since(version: str, alternative: Optional[str] = None) -> Callable[[Any], Any]:
    """
    Decorator to mark functions or classes as deprecated since a given version,
    emitting both a DeprecationWarning and a logging.warning().
    """

    def decorator(obj: Any) -> Any:
        name = obj.__name__
        msg = f"'{name}' is deprecated since version {version} and will be removed in a future release."
        if alternative:
            msg += f" Use '{alternative}' instead."

        def _emit_warning() -> None:
            warnings.warn(msg, category=DeprecationWarning, stacklevel=3)

        if inspect.isclass(obj):
            orig_init = obj.__init__

            @wraps(orig_init)
            def new_init(self, *args, **kwargs):  # type: ignore
                _emit_warning()
                return orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj
        else:

            @wraps(obj)
            def new_func(*args, **kwargs):  # type: ignore
                _emit_warning()
                return obj(*args, **kwargs)

            return new_func

    return decorator
