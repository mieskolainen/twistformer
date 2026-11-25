# Console caller names and terminal colors
#
# mikael.mieskolainen@cern.ch (2026)

import io
import re
from types import SimpleNamespace

from twistboard.experiments import Tee
from twistformer.logging import print


def test_print_records_caller_and_options():
    """
    Retain the caller name and print options in plain log output
    """
    stream = io.StringIO()
    print("Active device:", "cpu", file=stream, sep=" | ", end="\n\n")
    output = stream.getvalue()
    assert re.match(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", output)
    assert output.endswith(
        f" | {__name__}.test_print_records_caller_and_options: | Active device: | cpu\n\n"
    )
    assert "\x1b" not in output


def test_print_colors_terminal_and_keeps_log_plain():
    """
    Color caller names magenta while preserving plain experiment logs
    """
    terminal, log = io.StringIO(), io.StringIO()
    terminal.isatty = lambda: True
    print("Active device: cuda", file=Tee(terminal, log))
    caller = f"{__name__}.test_print_colors_terminal_and_keeps_log_plain:"
    assert f"\x1b[35m{caller}\x1b[0m" in terminal.getvalue()
    assert caller in log.getvalue()
    assert "\x1b" not in log.getvalue()


def test_print_resolves_module_entry_point():
    """
    Report the importable module name for a Python module entry point
    """
    stream = io.StringIO()
    namespace = {
        "__name__": "__main__",
        "__spec__": SimpleNamespace(name="twistformer.cli.evaluate"),
        "print": print,
        "stream": stream,
    }
    exec("def report():\n    print('Active device: cpu', file=stream)\n", namespace)
    namespace["report"]()
    assert "cli.evaluate.report: Active device: cpu" in stream.getvalue()
    assert "twistformer." not in stream.getvalue()
