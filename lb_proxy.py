"""Lightweight streaming-aware HTTP load balancer for TTS backends."""

import argparse
import itertools
import logging
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

import urllib3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LB] %(message)s")
logger = logging.getLogger(__name__)

urllib3.disable_warnings()

backends = []
backend_cycle = None
cycle_lock = threading.Lock()


class LBHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _pick_backend(self):
        with cycle_lock:
            return next(backend_cycle)

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def _proxy(self):
        backend = self._pick_backend()
        url = f"http://{backend}{self.path}"

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        try:
            http = urllib3.PoolManager()
            resp = http.urlopen(
                self.command,
                url,
                body=body,
                headers={k: v for k, v in self.headers.items()},
                preload_content=False,
                redirect=False,
            )

            self.send_response(resp.status)
            for key, val in resp.headers.items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            for chunk in resp.stream(4096):
                if chunk:
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            resp.release_conn()

        except Exception as e:
            logger.error(f"Backend {backend} error: {e}")
            self.send_error(502, f"Backend error: {e}")

    def log_message(self, fmt, *args):
        logger.info(f"{self.client_address[0]} -> {self._pick_backend.__name__}: {fmt % args}")


def main():
    global backends, backend_cycle

    parser = argparse.ArgumentParser(description="TTS Load Balancer")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument("--backends", type=str, required=True,
                        help="Comma-separated backend addresses (host:port)")
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")]
    backend_cycle = itertools.cycle(backends)

    logger.info(f"Backends: {backends}")
    logger.info(f"Listening on :{args.listen_port}")

    server = ThreadingHTTPServer(("0.0.0.0", args.listen_port), LBHandler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
