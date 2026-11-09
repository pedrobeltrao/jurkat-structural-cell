"""Local static server for the viewer with HTTP Range support and CORS (python -m http.server has neither).

Whole-cell builds (jenc_full.py) read each tile as a byte range of a block shard; the viewer falls back to fetching
whole shards from servers without Range support, so plain http.server also works, only with more traffic.

usage: python serve.py [DIR] [--port 8000] [--bind 127.0.0.1]
"""
import argparse, os, re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204); self.end_headers()

    def send_head(self):
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", "").strip())
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        a = int(m.group(1)); b = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
        if a >= size or b < a:
            self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return None
        f = open(path, "rb"); f.seek(a)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self._remaining = b - a + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            buf = source.read(min(n, 1 << 20))
            if not buf: break
            outputfile.write(buf); n -= len(buf)
        self._remaining = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"))
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--bind", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.bind, a.port), partial(RangeHandler, directory=a.dir))
    print(f"serving {a.dir} at http://{a.bind}:{a.port}/ (Range + CORS)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
