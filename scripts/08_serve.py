"""全唐诗语义图谱 API + 静态服务（仅用标准库）。

数据在 data/poems.db（由 07_build_db.py 生成）。用法:
    .venv/bin/python scripts/08_serve.py [端口]   # 默认 8899

接口:
    GET /              -> outputs/index.html
    GET /api/points    -> 全部点位（轻量字段），页面初始化用
    GET /api/poem?id=N -> 单首诗完整信息（正文/小传/标签等）
    GET /api/random?m=N -> 主题 N 内随机一首诗 id
    GET /api/routes    -> 漫游路线（outputs/routes.json）
    GET /api/search?q= -> 按诗名/诗人/诗句搜索（LIKE，最多 20 条）
"""
import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_DIR, "data", "poems.db")
INDEX_PATH = os.path.join(BASE_DIR, "outputs", "index.html")

_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row


def get_points():
    rows = _db.execute(
        "SELECT id, title, author, tx, ty, fine, macro FROM poems").fetchall()
    macros = _db.execute(
        "SELECT macro, macro_name, COUNT(*) AS n FROM poems "
        "GROUP BY macro ORDER BY macro").fetchall()
    fines = _db.execute(
        "SELECT macro, fine, fine_name, COUNT(*) AS n FROM poems "
        "GROUP BY fine ORDER BY fine").fetchall()
    fines_by_macro = {}
    for r in fines:
        fines_by_macro.setdefault(r["macro"], []).append(
            {"id": r["fine"], "name": r["fine_name"], "count": r["n"]})
    return {
        "macros": [{"id": r["macro"], "name": r["macro_name"], "count": r["n"],
                    "fines": fines_by_macro.get(r["macro"], [])}
                   for r in macros],
        "points": [[r["id"], r["title"], r["author"], r["tx"], r["ty"],
                    r["macro"], r["fine"]] for r in rows],
    }


def get_poem(pid: int):
    r = _db.execute("SELECT * FROM poems WHERE id = ?", (pid,)).fetchone()
    if r is None:
        return None
    return {
        "id": r["id"], "title": r["title"], "author": r["author"],
        "paragraphs": json.loads(r["paragraphs"]),
        "era": r["era"], "form": r["form"],
        "tags": json.loads(r["tags"]), "bio": r["bio"],
        "analysis": r["analysis"],
        "macro": r["macro"], "macro_name": r["macro_name"], "fine": r["fine"],
        "fine_name": r["fine_name"],
    }


def random_poem(macro: int):
    r = _db.execute(
        "SELECT id FROM poems WHERE macro = ? ORDER BY RANDOM() LIMIT 1",
        (macro,)).fetchone()
    return {"id": r["id"]} if r else None


def search_poems(q: str):
    like = f"%{q}%"
    rows = _db.execute(
        "SELECT id, title, author FROM poems "
        "WHERE title LIKE ? OR author LIKE ? OR paragraphs LIKE ? "
        "ORDER BY CASE "
        "  WHEN title LIKE ? THEN 0 "
        "  WHEN author LIKE ? THEN 1 "
        "  ELSE 2 END, id "
        "LIMIT 20", (like, like, like, like, like)).fetchall()
    return [{"id": r["id"], "title": r["title"], "author": r["author"]}
            for r in rows]


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", code)

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                with open(INDEX_PATH, "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
            elif url.path == "/api/points":
                self._json(POINTS)
            elif url.path == "/api/poem":
                poem = get_poem(int(qs["id"][0]))
                self._json(poem) if poem else self._json({"error": "not found"}, 404)
            elif url.path == "/api/random":
                hit = random_poem(int(qs["m"][0]))
                self._json(hit) if hit else self._json({"error": "not found"}, 404)
            elif url.path == "/api/routes":
                self._json(ROUTES)
            elif url.path == "/api/search":
                q = qs["q"][0].strip()
                self._json(search_poems(q) if q else [])
            else:
                self._json({"error": "not found"}, 404)
        except (KeyError, ValueError, IndexError):
            self._json({"error": "bad request"}, 400)

    def log_message(self, *args):
        pass


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
POINTS = get_points()  # 启动时加载一次，之后只读
with open(os.path.join(BASE_DIR, "outputs", "routes.json"), encoding="utf-8") as _f:
    ROUTES = json.load(_f)
print(f"点位 {len(POINTS['points'])} 首, 主题 {len(POINTS['macros'])} 个, 路线 {len(ROUTES)} 条")
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
