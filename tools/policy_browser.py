"""Local chair picker supervising one isolated deploy_policy process at a time.

Each scene owns its CUDA and Viser lifetime. Switching waits for the previous
process to exit, so neither physics allocations nor viewer callbacks accumulate.
"""
from __future__ import annotations

import json
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


class PolicyBrowser:
    def __init__(self, args, paths):
        self.args, self.paths = args, paths
        self.viewer_port = args.port + 1
        # Refuse to reuse an unrelated viewer's port or show its scene.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.viewer_port))
        self.log_dir = args.checkpoint.resolve().parent / "policy_browser"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._generation = 0
        self._index = args.index
        self._state = dict(generation=0, index=args.index, status="loading",
                           message="正在加载策略和椅子…", log_tail="",
                           viewer_url=f"http://127.0.0.1:{self.viewer_port}")
        self._thread = threading.Thread(target=self._run, name="policy-browser", daemon=True)

    def start(self):
        self._thread.start()

    def select(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.paths):
            raise ValueError("Invalid chair index")
        with self._lock:
            self._generation += 1
            self._index = index
            self._state.update(generation=self._generation, index=index, status="loading",
                               message="正在切换椅子…", log_tail="")

    def state(self):
        with self._lock:
            return self._state.copy()

    def _update(self, generation, **changes):
        with self._lock:
            if generation == self._generation:
                self._state.update(changes)

    def _command(self, index):
        args = self.args
        command = [sys.executable, "-u", "-m", "tools.deploy_policy",
                   "--checkpoint", str(args.checkpoint.resolve()),
                   "--split", str((args.split or args.checkpoint.parent / "split.json").resolve()),
                   "--reference", str(self.paths[index].resolve()),
                   "--cache-dir", str(args.cache_dir.resolve()),
                   "--viewer", "viser", "--port", str(self.viewer_port), "--episodes", "0",
                   "--device", args.device, "--init", args.init_mode,
                   "--noise-scale", str(args.noise_scale), "--seed", str(args.seed)]
        command += ["--random-start"] if args.random_start else ["--start-frame", str(args.start_frame)]
        if args.data_root:
            command += ["--data-root", str(args.data_root.resolve())]
        if args.max_steps is not None:
            command += ["--max-steps", str(args.max_steps)]
        return command

    @staticmethod
    def _close_process(process):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _run(self):
        process, stream, log_path, active = None, None, None, -1
        ready = False
        try:
            while not self._stop.is_set():
                with self._lock:
                    desired, index = self._generation, self._index
                if desired != active:
                    self._close_process(process)
                    process = None
                    if stream is not None:
                        stream.close()
                        stream = None
                    # A user may change their choice while the old process exits.
                    with self._lock:
                        active, index = self._generation, self._index
                    if self._stop.is_set():
                        break
                    log_path = self.log_dir / f"visit-{time.time_ns()}-{index:03d}.log"
                    stream = log_path.open("w")
                    ready = False
                    try:
                        process = subprocess.Popen(self._command(index), cwd=ROOT,
                                                   stdout=stream, stderr=subprocess.STDOUT)
                    except OSError as exc:
                        self._update(active, status="error", message=f"无法启动回放：{exc}")
                    else:
                        print(f"[browse] index={index} pid={process.pid} reference={self.paths[index].name}", flush=True)
                if process is not None:
                    with log_path.open("rb") as reader:
                        reader.seek(max(0, log_path.stat().st_size - 12000))
                        tail = reader.read().decode("utf-8", errors="replace")
                    if process.poll() is not None:
                        self._update(active, status="error", log_tail=tail,
                                     message=f"回放进程退出（code {process.returncode}），可选择其它椅子或重新开始。")
                    elif "[ERROR] Exception during step:" in tail:
                        self._update(active, status="error", log_tail=tail,
                                     message="物理回放出现错误并已暂停，请查看日志。")
                    else:
                        if not ready:
                            connection = HTTPConnection("127.0.0.1", self.viewer_port, timeout=1)
                            try:
                                connection.request("GET", "/")
                                response = connection.getresponse()
                                ready = response.status == 200
                                response.read()
                            except (OSError, HTTPException):
                                pass
                            finally:
                                connection.close()
                        if ready:
                            self._update(active, status="ready", message="策略正在物理仿真中执行 · 50 Hz", log_tail=tail)
                self._stop.wait(.5)
        except Exception as exc:
            self._update(active, status="error", message=f"无法启动回放：{exc}")
        finally:
            self._close_process(process)
            if stream is not None:
                stream.close()

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()


def serve(args, manifest, paths, iteration):
    browser = PolicyBrowser(args, paths)
    partitions = {c["path"]: part for part in ("train", "test")
                  for group in manifest[part] for c in group["clips"]}
    catalog = dict(checkpoint=args.checkpoint.name, iteration=iteration,
                   references=[dict(index=i, name=p.name, partition=partitions.get(p.name, "external"))
                               for i, p in enumerate(paths)])
    page = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, mime="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.send(200, page, "text/html")
            elif self.path == "/api/catalog":
                self.send(200, catalog)
            elif self.path == "/api/state":
                self.send(200, browser.state())
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/select":
                return self.send(404, {"error": "Not found"})
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"):
                return self.send(403, {"error": "Local origin required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024:
                    raise ValueError("Invalid request size")
                browser.select(json.loads(self.rfile.read(size))["index"])
            except (ValueError, KeyError, TypeError) as exc:
                return self.send(400, {"error": str(exc)})
            self.send(202, browser.state())

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        browser.start()
        print(f"[browse] http://127.0.0.1:{args.port} — {len(paths)} chairs", flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        browser.close()
        server.server_close()
        signal.signal(signal.SIGTERM, previous)
