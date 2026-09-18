"""Read-only, loopback training charts. This service never controls a learner."""
import argparse
import json
import math
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .lifecycle import lifecycle


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


class History:
    """Incremental complete-line reader; keep all records, including old sessions."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.key = None
        self.rows = []

    def read(self):
        try:
            with self.path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                key = stat.st_dev, stat.st_ino
                if key != self.key or stat.st_size < self.offset:
                    self.rows, self.offset, self.key = [], 0, key
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            self.rows.append(clean(row))
                    except (ValueError, UnicodeError):
                        continue
        except FileNotFoundError:
            self.rows, self.offset, self.key = [], 0, None
        return self.rows


class Dashboard:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.updates = History(self.run / "updates.jsonl")
        self.episodes = History(self.run / "episodes.jsonl")
        self.lock = threading.Lock()
        self.started = time.time()

    def snapshot(self):
        with self.lock:
            try:
                state = json.loads((self.run / "status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = {}
            live, phase, age = lifecycle(state)
            # Serialize under the lock: concurrent refreshes cannot mutate the lists.
            return json.dumps(clean(dict(run=self.run.name, run_path=str(self.run), state=state,
                lifecycle=live, phase=phase, heartbeat_age=age,
                server=dict(pid=os.getpid(), started=self.started, status="alive"),
                updates=self.updates.read(), episodes=self.episodes.read(),
                generated_at=time.time())), allow_nan=False).encode()


def make_server(run, port=8765):
    dashboard = Dashboard(run)
    assets = Path(__file__).with_name("dashboard_assets")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Reject DNS-rebinding hosts; no CORS and no arbitrary filesystem routes.
            if self.headers.get("Host") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                self.send_error(403)
                return
            route = self.path.split("?", 1)[0]
            if route == "/api/history":
                body, mime = dashboard.snapshot(), "application/json"
            elif route in {"/", "/charts.js"}:
                body = (assets / ("index.html" if route == "/" else "charts.js")).read_bytes()
                mime = "text/html; charset=utf-8" if route == "/" else "text/javascript; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error("Run directory does not exist")
    server = make_server(args.run, args.port)
    print(f"Local: http://127.0.0.1:{server.server_port}/", flush=True)
    print("Read-only history service; remains available after learner exit. Ctrl+C exits only this dashboard.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
