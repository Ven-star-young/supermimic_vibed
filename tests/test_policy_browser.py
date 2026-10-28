"""Exercise process ownership without loading a policy or allocating a GPU."""
import os
import socket
import sys
import time
from types import SimpleNamespace

import pytest

from tools.policy_browser import PolicyBrowser


def _browser(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        viewer_port = probe.getsockname()[1]
    args = SimpleNamespace(port=viewer_port - 1, index=0,
                           checkpoint=tmp_path / "model.pt")
    return PolicyBrowser(args, [tmp_path / f"chair_{i}.npz" for i in range(2)])


def _wait_for(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(.02)
    pytest.fail("Policy browser did not reach the expected state")


def _assert_exited(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_switch_reaps_previous_process_and_close_reaps_current(tmp_path, monkeypatch):
    browser = _browser(tmp_path)
    events = tmp_path / "events.txt"
    script = r"""
import os, signal, socket, sys, time
from pathlib import Path
index, port, root = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
def record(event):
    with (root / 'events.txt').open('a') as stream:
        stream.write(f'{event} {index}\n')
def stop(*_):
    record('stopping')
    time.sleep(.2)  # Make terminate-without-wait fail reliably.
    record('stopped')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
if index:
    previous = int((root / 'pid0').read_text())
    try:
        os.kill(previous, 0)
    except ProcessLookupError:
        pass
    else:
        print('ERROR: previous process still exists', flush=True)
        sys.exit(88)
(root / f'pid{index}').write_text(str(os.getpid()))
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', port))
server.listen()
server.settimeout(.1)
record('started')
while True:
    try:
        client, _ = server.accept()
        client.settimeout(1)
        client.recv(4096)
        client.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK')
        client.close()
    except socket.timeout:
        pass
"""
    monkeypatch.setattr(browser, "_command", lambda index: [
        sys.executable, "-u", "-c", script, str(index), str(browser.viewer_port), str(tmp_path),
    ])
    browser.start()
    try:
        _wait_for(lambda: browser.state()["status"] == "ready")
        first_pid = int((tmp_path / "pid0").read_text())
        browser.select(1)
        _wait_for(lambda: browser.state()["status"] == "ready")
        second_pid = int((tmp_path / "pid1").read_text())
        _assert_exited(first_pid)
        assert browser.state()["index"] == 1
        assert events.read_text().splitlines() == [
            "started 0", "stopping 0", "stopped 0", "started 1",
        ]
    finally:
        browser.close()
    _assert_exited(second_pid)
    assert events.read_text().splitlines()[-2:] == ["stopping 1", "stopped 1"]
    browser.close()  # Cleanup may be called by both an error handler and finally.


def test_child_failure_exposes_exit_code_and_log(tmp_path, monkeypatch):
    browser = _browser(tmp_path)
    monkeypatch.setattr(browser, "_command", lambda _: [
        sys.executable, "-u", "-c", "print('scene loading failed'); raise SystemExit(7)",
    ])
    browser.start()
    try:
        _wait_for(lambda: browser.state()["status"] == "error")
        state = browser.state()
        assert "code 7" in state["message"]
        assert "scene loading failed" in state["log_tail"]
    finally:
        browser.close()


def test_invalid_selection_does_not_change_current_scene(tmp_path):
    browser = _browser(tmp_path)
    original = browser.state()
    for invalid in (True, False, -1, 2, 1.0, "1", None):
        with pytest.raises(ValueError, match="Invalid chair index"):
            browser.select(invalid)
        assert browser.state() == original
    browser.close()
