# Local dashboard process control
#
# mikael.mieskolainen@cern.ch (2026)

import os
from pathlib import Path
import signal
import socket
import sys
import time

DASHBOARD_NAMES = {"twistboard", "quisboard"}

def is_dashboard(command):
    """
    Identify current and legacy dashboard serve commands

    Args:
        command:  Process argument list
    Returns:
        matched: True for a Twistboard serve process
    """
    if not command:
        return False
    executable = Path(command[0]).name
    if executable in DASHBOARD_NAMES:
        return command[1:2] == ["serve"]
    if not executable.startswith("python"):
        return False
    for index, argument in enumerate(command[1:], 1):
        if argument in {"-c", "--"}:
            return False
        if argument == "-m":
            return len(command) > index + 2 and command[index + 1] in DASHBOARD_NAMES and command[index + 2] == "serve"
        if not argument.startswith("-"):
            return Path(argument).name in DASHBOARD_NAMES and command[index + 1 : index + 2] == ["serve"]
    return False

def process_identity(process):
    """
    Read the process start time and exclude exited processes

    Args:
        process:  Linux process directory
    Returns:
        identity: Start time in clock ticks, or None after exit
    """
    try:
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (FileNotFoundError, ProcessLookupError):
        return None

def terminate_dashboard(process, identity):
    """
    Stop a verified dashboard with a bounded termination wait

    Args:
        process:  Linux process directory
        identity: Process start time before listener inspection
    """
    handle = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            handle = os.pidfd_open(int(process.name))
        for termination, timeout in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 2.0)):
            if process_identity(process) != identity:
                return
            if handle is not None:
                signal.pidfd_send_signal(handle, termination)
            else:
                os.kill(int(process.name), termination)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process_identity(process) != identity:
                    return
                time.sleep(0.05)
        raise RuntimeError(f"Twistboard PID {process.name} did not exit")
    except ProcessLookupError:
        return
    finally:
        if handle is not None:
            os.close(handle)

def stop_dashboard(host, port):
    """
    Stop a same-user Linux dashboard that owns the requested listener

    Args:
        host:  Requested IPv4 host
        port:  Requested listening port
    Returns:
        stopped: True when a matching dashboard has exited
    """
    if not sys.platform.startswith("linux"):
        raise RuntimeError("--force requires Linux")
    address = socket.gethostbyname(host)
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        raw_address, raw_port = fields[1].split(":")
        bound_address = socket.inet_ntoa(int(raw_address, 16).to_bytes(4, sys.byteorder))
        if fields[3] == "0A" and int(raw_port, 16) == port:
            if address == "0.0.0.0" or bound_address in {address, "0.0.0.0"}:
                inodes.add(f"socket:[{fields[9]}]")
    if not inodes:
        return False
    stopped = False
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            identity = process_identity(process)
            if identity is None:
                continue
            command = (process / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            if not is_dashboard(command):
                continue
            owns_listener = False
            for descriptor in (process / "fd").iterdir():
                try:
                    if os.readlink(descriptor) in inodes:
                        owns_listener = True
                        break
                except FileNotFoundError:
                    continue
            if not owns_listener:
                continue
            print(f"Stopping Twistboard PID {process.name} on port {port}", flush=True)
            terminate_dashboard(process, identity)
            stopped = True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return stopped
