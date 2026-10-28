"""Browser process ownership, request safety and safe reference presentation."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from supermimic.scripts import policy_browser


@pytest.fixture
def browser(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1] - 1
    args = SimpleNamespace(port=port, checkpoint=tmp_path / "model_200.pt", split=None, index=0,
                           cache_dir=tmp_path / "cache", device="cuda:0", random_start=False,
                           start_frame=0, data_root=None, max_steps=None, metrics_json=tmp_path / "metrics.json")
    value = policy_browser.PolicyBrowser(args, [tmp_path / "chair ; $(bad).npz", tmp_path / "other.npz"])
    yield value
    value.close()


def settings(index=0, **overrides):
    return dict(index=index, init_mode="perturbed", seed=7, noise_scale=.5) | overrides


def until(condition):
    deadline = time.monotonic() + 4
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("Browser worker did not reach expected state")
        time.sleep(.02)


def test_invalid_input_cannot_escape_reference_allowlist(browser):
    for bad in (settings(-1), settings(2), settings(True), settings(seed=-1), settings(seed=2**32),
                settings(seed=3.5), settings(noise_scale=float("nan")), settings(noise_scale=True),
                settings(init_mode="push"), settings(reference="/etc/passwd"), None, []):
        with pytest.raises(ValueError):
            browser.select(bad)
    assert browser.state()["status"] == "idle"
    command = browser._command(settings(), Path("/tmp/result.json"))
    assert command[command.index("--reference") + 1] == str(browser.paths[0])
    assert command[command.index("--metrics-json") + 1] == str(browser.args.metrics_json)
    assert command[command.index("--episodes") + 1] == "0"
    assert "supermimic.scripts.play_policy" in command
    assert "--browse" not in command


def test_replacing_and_stopping_reaps_child_before_new_scene(browser, monkeypatch):
    children, events = [], []

    class Child:
        def __init__(self, command, **kwargs):
            assert "shell" not in kwargs
            assert all(child.returncode is not None and child.waited for child in children)
            self.command, self.returncode, self.waited = command, None, False
            self.number = len(children)
            children.append(self)
            events.append((self.number, "spawn"))
            Path(command[command.index("--output") + 1]).write_text(json.dumps({"references": []}))

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            events.append((self.number, "terminate"))

        def wait(self, timeout=None):
            self.waited = True
            events.append((self.number, "wait"))
            return self.returncode

    monkeypatch.setattr(policy_browser.subprocess, "Popen", Child)
    monkeypatch.setattr(browser, "_viewer_ready", lambda: True)
    browser.start()
    assert not children  # Loading the page alone never allocates GPU state.
    browser.select(settings())
    until(lambda: browser.state()["status"] == "ready")
    browser.select(settings(1, seed=99))
    until(lambda: len(children) == 2)
    assert events[:4] == [(0, "spawn"), (0, "terminate"), (0, "wait"), (1, "spawn")]
    command = children[1].command
    assert command[command.index("--seed") + 1] == "99"
    browser.stop()
    until(lambda: browser.state()["status"] == "idle")
    assert all(child.waited and child.returncode == -15 for child in children)
    assert browser.state()["report"] == {"references": []}


def test_child_launch_failure_is_visible_and_can_retry(browser, monkeypatch):
    monkeypatch.setattr(policy_browser.subprocess, "Popen", lambda *_, **__: (_ for _ in ()).throw(OSError("missing executable")))
    browser.start()
    browser.select(settings())
    until(lambda: browser.state()["status"] == "error")
    assert "missing executable" in browser.state()["message"]
    browser.stop()
    until(lambda: browser.state()["status"] == "idle")


def test_latest_selection_wins_while_previous_process_is_exiting(browser, monkeypatch):
    children, waiting, release = [], threading.Event(), threading.Event()

    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.returncode = command, None
            children.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            if self is children[0]:
                waiting.set()
                assert release.wait(3)
            return self.returncode

    monkeypatch.setattr(policy_browser.subprocess, "Popen", Child)
    monkeypatch.setattr(browser, "_viewer_ready", lambda: True)
    browser.start()
    browser.select(settings())
    until(lambda: browser.state()["status"] == "ready")
    browser.select(settings(1))
    assert waiting.wait(2)
    browser.select(settings(0, seed=1234))
    release.set()
    until(lambda: len(children) == 2)
    command = children[1].command
    assert command[command.index("--reference") + 1] == str(browser.paths[0])
    assert command[command.index("--seed") + 1] == "1234"


def test_stuck_child_is_killed_and_reaped():
    calls = []

    class Child:
        poll = lambda self: None
        terminate = lambda self: calls.append("terminate")
        kill = lambda self: calls.append("kill")

        def wait(self, timeout=None):
            calls.append("wait")
            if timeout is not None:
                raise subprocess.TimeoutExpired("child", timeout)

    policy_browser.PolicyBrowser._close_process(Child())
    assert calls == ["terminate", "wait", "kill", "wait"]


def test_http_rejects_cross_site_and_dns_rebinding_and_keeps_names_as_data(browser):
    page = Path(policy_browser.__file__).with_suffix(".html").read_bytes()
    evil_name = '<img src=x onerror="window.BAD=true">.npz'
    catalog = {"references": [{"index": 0, "name": evil_name}]}
    server = ThreadingHTTPServer(("127.0.0.1", 0), policy_browser.make_handler(browser, catalog, page))
    # Handler origin allowlist is captured at construction, so rebuild with actual bound port.
    browser.args.port = server.server_address[1]
    server.RequestHandlerClass = policy_browser.make_handler(browser, catalog, page)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()

    def request(method, path, body=None, **headers):
        connection = HTTPConnection("127.0.0.1", browser.args.port, timeout=2)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None, headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    origin = f"http://127.0.0.1:{browser.args.port}"
    try:
        status, payload = request("GET", "/api/catalog")
        assert status == 200 and json.loads(payload)["references"][0]["name"] == evil_name
        assert evil_name.encode() not in page
        assert b"innerHTML" not in page and b"name.textContent=ref.name" in page
        assert request("GET", "/api/catalog", Host="attacker.example")[0] == 403
        assert request("POST", "/api/start", settings(), Origin="https://attacker.example", **{"Content-Type": "application/json"})[0] == 403
        assert request("POST", "/api/start", settings(), **{"Content-Type": "application/json"})[0] == 403
        assert request("POST", "/api/start", settings(), Origin=origin, **{"Content-Type": "text/plain"})[0] == 400
        assert request("POST", "/api/start", settings(index=99), Origin=origin, **{"Content-Type": "application/json"})[0] == 400
        assert request("POST", "/api/start", settings(), Origin=origin, **{"Content-Type": "application/json"})[0] == 202
        assert browser.state()["status"] == "loading"
        assert request("POST", "/api/stop", {}, Origin=origin, **{"Content-Type": "application/json"})[0] == 202
        assert browser.state()["status"] == "stopping"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
