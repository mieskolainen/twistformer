# Timestamped console output with caller names
#
# mikael.mieskolainen@cern.ch (2026)

import builtins
from datetime import datetime
import sys

def print(*args, **kwargs):
    """
    Print a local timestamp and a magenta caller name in terminals

    Args:
        args:  Values passed to the built-in print
        kwargs:  Print options, with immediate flushing by default
    Returns:
        None: Output uses the active terminal and run log stream
    """
    kwargs.setdefault("flush", True)
    stream = kwargs.get("file") or sys.stdout
    terminal = getattr(stream, "terminal", stream)
    stamp = f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"
    frame = sys._getframe(1)
    try:
        spec = frame.f_globals.get("__spec__")
        module = spec.name if spec is not None else frame.f_globals.get("__name__", "__main__")
        module = module.removeprefix("twistformer.")
        caller = f"{module}.{frame.f_code.co_qualname}:"
    finally:
        del frame
    if getattr(terminal, "isatty", lambda: False)():
        stamp = f"\033[33m{stamp}\033[0m"
        caller = f"\033[35m{caller}\033[0m"
    builtins.print(stamp, caller, *args, **kwargs)
