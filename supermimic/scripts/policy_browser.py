"""Local reference picker; one isolated policy/Viser process at a time."""
from __future__ import annotations

from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time


class PolicyBrowser:
    def __init__(self, args, paths):
        self.args, self.paths = args, [Path(path).resolve() for path in paths]
        self.viewer_port = args.port + 1
        if not 1 <= args.port < 65535:
            raise ValueError("Browser requires two consecutive valid ports")
        # Never display an unrelated Viser instance using this port.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.viewer_port))
        self.log_dir = args.checkpoint.resolve().parent / "policy_browser"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock, self._closed = threading.Lock(), threading.Event()
        self._generation, self._requested = 0, None
        self._state = dict(generation=0, index=args.index, status="idle", message="选择 reference 后开始回放。",
                           viewer_url=f"http://127.0.0.1:{self.viewer_port}", log_tail="", report=None)
        self._thread = threading.Thread(target=self._run, name="policy-browser", daemon=True)

    def start(self):
        self._thread.start()

    def select(self, settings):
        if not isinstance(settings, dict) or set(settings) != {"index", "init_mode", "seed", "noise_scale"}:
            raise ValueError("Expected index, init_mode, seed and noise_scale")
        index, mode, seed, scale = (settings[key] for key in ("index", "init_mode", "seed", "noise_scale"))
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.paths):
            raise ValueError("Reference index out of range")
        if mode not in ("clean", "perturbed"):
            raise ValueError("Invalid initialization mode")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2^32)")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError("Noise scale must be finite and nonnegative")
        with self._lock:
            self._generation += 1
            self._requested = dict(settings)
            self._state.update(generation=self._generation, index=index, status="loading", settings=dict(settings),
                               message="正在加载策略、碰撞网格与物理仿真…", report=None, log_tail="")

    def stop(self):
        with self._lock:
            self._generation += 1
            self._requested = None
            self._state.update(generation=self._generation, status="stopping", message="正在停止回放并释放显存…")

    def state(self):
        with self._lock:
            return self._state.copy()

    def _update(self, generation, **changes):
        with self._lock:
            if generation == self._generation:
                self._state.update(changes)

    def _command(self, settings, output):
        args = self.args
        command = [sys.executable, "-u", "-m", "supermimic.scripts.play_policy",
                   "--checkpoint", str(args.checkpoint.resolve()),
                   "--split", str((args.split or args.checkpoint.parent / "split.json").resolve()),
                   "--reference", str(self.paths[settings["index"]]),
                   "--cache-dir", str(args.cache_dir.resolve()),
                   "--viewer", "viser", "--port", str(self.viewer_port), "--episodes", "0",
                   "--output", str(output), "--device", args.device,
                   "--init", settings["init_mode"], "--noise-scale", str(settings["noise_scale"]),
                   "--seed", str(settings["seed"])]
        command += ["--random-start"] if args.random_start else ["--start-frame", str(args.start_frame)]
        if args.data_root:
            command += ["--data-root", str(args.data_root.resolve())]
        if args.max_steps is not None:
            command += ["--max-steps", str(args.max_steps)]
        if getattr(args, "metrics_json", None):
            command += ["--metrics-json", str(args.metrics_json.resolve())]
        return command

    @staticmethod
    def _close_process(process):
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    def _tail(path):
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16000))
            return stream.read().decode("utf-8", errors="replace")

    @staticmethod
    def _read_report(path):
        try:
            return json.loads(path.read_text()) if path and path.is_file() else None
        except (OSError, ValueError):
            return None

    def _viewer_ready(self):
        connection = HTTPConnection("127.0.0.1", self.viewer_port, timeout=.5)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read()
            return response.status == 200
        except (OSError, HTTPException):
            return False
        finally:
            connection.close()

    def _run(self):
        process, stream, active, output, log_path = None, None, 0, None, None
        ready = False
        try:
            while not self._closed.is_set():
                with self._lock:
                    generation, requested = self._generation, self._requested
                if generation != active:
                    self._close_process(process)
                    process = None
                    if stream is not None:
                        stream.close()
                        stream = None
                    # Changes made while shutdown waits must supersede stale choices.
                    with self._lock:
                        active, requested = self._generation, self._requested
                    if self._closed.is_set():
                        break
                    if requested is None:
                        self._update(active, status="idle", message="回放已停止，显存已释放。",
                                     report=self._read_report(output))
                    else:
                        name = f"visit-{time.time_ns()}-{requested['index']:03d}"
                        log_path, output = self.log_dir / f"{name}.log", self.log_dir / f"{name}.json"
                        stream, ready = log_path.open("w"), False
                        try:
                            process = subprocess.Popen(self._command(requested, output),
                                                       stdout=stream, stderr=subprocess.STDOUT)
                        except OSError as exc:
                            self._update(active, status="error", message=f"无法启动回放：{exc}")
                        else:
                            self._update(active, output=str(output), log_path=str(log_path))
                if process is not None:
                    tail, report = self._tail(log_path), self._read_report(output)
                    if process.poll() is not None:
                        process.wait()
                        self._update(active, status="error", log_tail=tail, report=report,
                                     message=f"回放进程退出（code {process.returncode}）。查看日志后可重新开始。")
                    elif "[ERROR] Exception during step:" in tail:
                        self._update(active, status="error", log_tail=tail, report=report,
                                     message="物理回放异常，请查看日志并停止或重新开始。")
                    else:
                        ready = ready or self._viewer_ready()
                        self._update(active, status="ready" if ready else "loading", log_tail=tail, report=report,
                                     message="策略正在执行 · 50 Hz · 每回合记录 seed 与指标" if ready else "正在加载策略与物理仿真…")
                self._closed.wait(.35)
        except Exception as exc:
            self._update(active, status="error", message=f"回放管理异常：{exc}")
        finally:
            self._close_process(process)
            if stream is not None:
                stream.close()

    def close(self):
        self._closed.set()
        if self._thread.is_alive():
            self._thread.join()


def make_handler(browser, catalog, page):
    """Only local same-origin requests can select or terminate a rollout."""
    origins = {f"http://{host}:{browser.args.port}" for host in ("127.0.0.1", "localhost")}
    hosts = {origin.removeprefix("http://") for origin in origins}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, mime="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _local(self):
            if self.headers.get("Host") not in hosts:
                self.send(403, {"error": "Local host required"})
                return False
            return True

        def do_GET(self):
            if not self._local():
                return
            if self.path == "/":
                self.send(200, page, "text/html")
            elif self.path == "/api/catalog":
                self.send(200, catalog)
            elif self.path == "/api/state":
                self.send(200, browser.state())
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if not self._local():
                return
            if self.headers.get("Origin") not in origins or self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.send(403, {"error": "Same local origin required"})
            if self.path not in ("/api/start", "/api/stop"):
                return self.send(404, {"error": "Not found"})
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("JSON request required")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2048:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(size))
                if self.path == "/api/start":
                    browser.select(value)
                elif value != {}:
                    raise ValueError("Stop accepts an empty object")
                else:
                    browser.stop()
            except (ValueError, KeyError, TypeError) as exc:
                return self.send(400, {"error": str(exc)})
            self.send(202, browser.state())

    return Handler


def run_browser(args, manifest, paths, iteration):
    browser = PolicyBrowser(args, paths)
    metadata = {clip["path"]: dict(partition=part, sha256=clip.get("sha256")) for part in ("train", "test")
                for group in manifest[part] for clip in group["clips"]}
    catalog = dict(checkpoint=args.checkpoint.name, iteration=iteration,
                   defaults=dict(index=args.index, init_mode=args.init_mode, seed=args.seed, noise_scale=args.noise_scale),
                   references=[dict(index=i, name=p.name, path=str(p.resolve()),
                                    **metadata.get(p.name, dict(partition="external", sha256=None)))
                               for i, p in enumerate(paths)], report=None)
    if getattr(args, "report", None):
        catalog["report"] = json.loads(args.report.read_text())
        catalog["report_name"] = args.report.name
    page = Path(__file__).with_suffix(".html").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(browser, catalog, page))
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        browser.start()
        print(f"[browse] http://127.0.0.1:{args.port} — {len(paths)} references; select one to start", flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        browser.close()
        server.server_close()
        signal.signal(signal.SIGTERM, previous)
