# Interruptible training subprocesses
#
# mikael.mieskolainen@cern.ch (2026)

import os
import signal
import subprocess


def run_command(command, *, env=None, check=True):
    """
    Forward Ctrl+C and allow the training child to close its workers

    Args:
        command: Child process argument list
        env: Optional child environment
        check: Raise for a nonzero exit status
    Returns:
        result: Completed child process record
    """
    with subprocess.Popen(command, env=env, start_new_session=os.name == "posix") as process:
        try:
            code = process.wait()
        except KeyboardInterrupt:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if code in {130, -signal.SIGINT}:
        raise KeyboardInterrupt
    if check and code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code)
