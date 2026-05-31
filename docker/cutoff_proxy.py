#!/usr/bin/env python3
"""Build/runtime package-date-cutoff proxy for conda AND pip.

One stdlib HTTP server that serves date-filtered views of two package indexes,
so no package published after the cutoff can be selected (defense vs
Shai-Hulud-style supply-chain attacks on popular packages):

  * conda - GET /<channel>/<subdir>/repodata.json: fetch upstream, drop packages
            whose `timestamp` is newer than the cutoff. The compressed /
            current_repodata.json variants are 404'd to force the plain full
            repodata; package files are 302-redirected to the upstream
            (immutable, sha256-checked).
  * pip   - GET /simple/<pkg>/: fetch the upstream PEP 691 JSON, drop files whose
            `upload-time` is newer than the cutoff. Kept files keep their
            absolute files.pythonhosted.org URLs, so pip downloads them directly.

Wire it up with config files (no per-tool daemon): conda via .condarc
`channel_alias: http://127.0.0.1:<port>`, pip via /etc/pip.conf
`index-url = http://127.0.0.1:<port>/simple`. Both are read on every invocation,
so even a naive `conda install foo` / `%pip install foo` is bounded.

stdlib only, so it runs on the base python at build and runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"


# --------------------------------------------------------------------------- #
# Pure filters (unit-testable)
# --------------------------------------------------------------------------- #
def filter_repodata(data: dict, cutoff_ms: int) -> tuple[dict, int]:
    """Drop conda packages whose `timestamp` (ms) is newer than the cutoff.

    Entries with no timestamp (pre-timestamp-era) are kept. Returns the filtered
    repodata and the number of dropped entries.
    """
    dropped = 0
    for key in ("packages", "packages.conda"):
        pkgs = data.get(key)
        if not isinstance(pkgs, dict):
            continue
        kept = {}
        for fn, rec in pkgs.items():
            if rec.get("timestamp", 0) <= cutoff_ms:
                kept[fn] = rec
            else:
                dropped += 1
        data[key] = kept
    return data, dropped


def _parse_upload_time(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_simple(data: dict, cutoff: dt.datetime) -> tuple[dict, int]:
    """Drop PyPI Simple files whose `upload-time` is newer than the cutoff.

    Files with no `upload-time` are kept. Returns the filtered project page and
    the number of dropped files.
    """
    files = data.get("files")
    if not isinstance(files, list):
        return data, 0
    kept, dropped = [], 0
    for f in files:
        ut = f.get("upload-time")
        t = _parse_upload_time(ut) if isinstance(ut, str) else None
        if t is None or t <= cutoff:
            kept.append(f)
        else:
            dropped += 1
    data["files"] = kept
    return data, dropped


# --------------------------------------------------------------------------- #
# Upstream fetchers
# --------------------------------------------------------------------------- #
def _conda_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as r:  # noqa: S310 (fixed upstream)
        return r.read()


def _pypi_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": SIMPLE_JSON})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed upstream)
        return r.read()


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
def build_handler(
    cutoff_ms: int,
    cutoff_dt: dt.datetime,
    conda_upstream: str,
    pypi_upstream: str,
    conda_fetch: Callable[[str], bytes] = _conda_fetch,
    pypi_fetch: Callable[[str], bytes] = _pypi_fetch,
):
    cache: dict[str, bytes] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes = b"", ctype: str | None = None) -> None:
            self.send_response(code)
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command == "GET" and body:
                self.wfile.write(body)

        def _serve_pip(self, base: str) -> None:
            if base in ("/simple", "/simple/"):
                self._send(200, b'{"meta":{"api-version":"1.0"},"projects":[]}', SIMPLE_JSON)
                return
            try:
                data = json.loads(pypi_fetch(pypi_upstream + base))
                filtered, dropped = filter_simple(data, cutoff_dt)
                body = json.dumps(filtered).encode()
            except urllib.error.HTTPError as e:
                self._send(e.code)
                return
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[cutoff:pypi] ERROR {base}: {e!r}\n")
                sys.stderr.flush()
                self._send(502)
                return
            sys.stderr.write(f"[cutoff:pypi] {base}: kept {len(filtered.get('files', []))}, dropped {dropped}\n")
            sys.stderr.flush()
            self._send(200, body, SIMPLE_JSON)

        def _serve_conda(self, base: str) -> None:
            name = base.rsplit("/", 1)[-1]
            # Force the client onto the full, plain repodata.json we filter
            # (404 the compressed / current variants, NOT package .tar.bz2 files).
            if name.startswith("current_repodata.json") or (name.startswith("repodata.json") and name != "repodata.json"):
                self._send(404)
                return
            if name == "repodata.json":
                with lock:
                    body = cache.get(base)
                if body is None:
                    try:
                        data = json.loads(conda_fetch(conda_upstream + base))
                        filtered, dropped = filter_repodata(data, cutoff_ms)
                        body = json.dumps(filtered).encode()
                    except Exception as e:  # noqa: BLE001
                        sys.stderr.write(f"[cutoff:conda] ERROR {base}: {e!r}\n")
                        sys.stderr.flush()
                        self._send(502)
                        return
                    kept = sum(len(filtered.get(k, {})) for k in ("packages", "packages.conda"))
                    sys.stderr.write(f"[cutoff:conda] {base}: kept {kept}, dropped {dropped}\n")
                    sys.stderr.flush()
                    with lock:
                        cache[base] = body
                self._send(200, body, "application/json")
                return
            # Package file: redirect to the immutable upstream artifact.
            self.send_response(302)
            self.send_header("Location", conda_upstream + base)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            base = self.path.split("?", 1)[0]
            if base == "/healthz":
                self._send(200, b"ok", "text/plain")
            elif base == "/simple" or base.startswith("/simple/"):
                self._serve_pip(base)
            else:
                self._serve_conda(base)

        do_HEAD = do_GET  # noqa: N815

    return Handler


def build_server(
    port: int,
    cutoff_ms: int,
    cutoff_dt: dt.datetime,
    conda_upstream: str,
    pypi_upstream: str,
    conda_fetch: Callable[[str], bytes] = _conda_fetch,
    pypi_fetch: Callable[[str], bytes] = _pypi_fetch,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        ("127.0.0.1", port),
        build_handler(cutoff_ms, cutoff_dt, conda_upstream.rstrip("/"), pypi_upstream.rstrip("/"), conda_fetch, pypi_fetch),
    )


def cutoff_to_dt(cutoff: str) -> dt.datetime:
    return dt.datetime.strptime(cutoff, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8723)
    ap.add_argument("--cutoff", required=True, help="YYYY-MM-DD (UTC); packages newer than this are dropped")
    ap.add_argument("--conda-upstream", default="https://conda.anaconda.org")
    ap.add_argument("--pypi-upstream", default="https://pypi.org")
    args = ap.parse_args()

    cutoff_dt = cutoff_to_dt(args.cutoff)
    cutoff_ms = int(cutoff_dt.timestamp() * 1000)
    server = build_server(args.port, cutoff_ms, cutoff_dt, args.conda_upstream, args.pypi_upstream)
    sys.stderr.write(
        f"[cutoff] serving :{args.port} cutoff={args.cutoff} conda={args.conda_upstream} pypi={args.pypi_upstream}\n"
    )
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
