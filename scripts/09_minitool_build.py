"""把交互图谱打包成小红书小工具 zip（纯离线，数据内置）。

做法：读 outputs/index.html，拆出内联 JS 为 assets/app.js，并按小工具规范改造
（禁 fetch -> 内置数据、外置经典脚本、viewport/安全区/触摸适配）；
从 data/poems.db 生成 assets/data.js（window.TANGSHI_DATA）。
产物：minitool/ 目录 + tangshi-minitool.zip（zip 根即 index.html）。

用法: python3 scripts/09_minitool_build.py
"""
import json
import os
import re
import sqlite3
import zipfile

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_DIR, "data", "poems.db")
SRC_HTML = os.path.join(BASE_DIR, "outputs", "index.html")
BUILD_DIR = os.path.join(BASE_DIR, "minitool")
ZIP_PATH = os.path.join(BASE_DIR, "tangshi-minitool.zip")
ZIP_LIMIT = 10 * 1024 * 1024  # 容器上限 10MB


def build_data_js():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id,title,author,paragraphs,era,form,tags,bio,"
        "tx,ty,fine,macro,macro_name,fine_name "
        "FROM poems ORDER BY id").fetchall()
    macros = conn.execute(
        "SELECT macro, macro_name, COUNT(*) n FROM poems "
        "GROUP BY macro ORDER BY macro").fetchall()
    fines = conn.execute(
        "SELECT macro, fine, fine_name, COUNT(*) n FROM poems "
        "GROUP BY fine ORDER BY fine").fetchall()
    conn.close()

    fines_by_macro = {}
    for f in fines:
        fines_by_macro.setdefault(f["macro"], []).append(
            {"id": f["fine"], "name": f["fine_name"], "count": f["n"]})
    bios, fine_names, points = {}, {}, []
    for r in rows:
        if r["bio"]:
            bios[r["author"]] = r["bio"]
        fine_names[str(r["fine"])] = r["fine_name"]
        points.append([
            r["title"], r["author"], r["era"], r["form"],
            json.loads(r["tags"]), json.loads(r["paragraphs"]),
            r["tx"], r["ty"], r["macro"], r["fine"],
        ])
    data = {
        "macros": [{"id": m["macro"], "name": m["macro_name"], "count": m["n"],
                    "fines": fines_by_macro.get(m["macro"], [])}
                   for m in macros],
        "bios": bios,
        "fineNames": fine_names,
        "points": points,
    }
    return ("window.TANGSHI_DATA = "
            + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")


def replace_once(text, old, new):
    assert text.count(old) == 1, f"替换目标不唯一或缺失: {old[:60]!r}"
    return text.replace(old, new)


def build_app_js(js):
    """内联 JS -> 离线版 app.js"""
    # 1) 禁 fetch：详情改为读内置数据
    js = replace_once(js, """// ---- 详情按需加载（缓存 + 防抖） ----
const detailCache = new Map();   // id -> detail
const pendingFetch = new Map();  // id -> Promise
function fetchDetail(id) {
  if (detailCache.has(id)) return Promise.resolve(detailCache.get(id));
  if (pendingFetch.has(id)) return pendingFetch.get(id);
  const p = fetch(`/api/poem?id=${id}`)
    .then(r => r.json())
    .then(d => { detailCache.set(id, d); pendingFetch.delete(id); return d; })
    .catch(() => { pendingFetch.delete(id); return null; });
  pendingFetch.set(id, p);
  return p;
}""", """// ---- 详情（数据内置于 data.js，零网络） ----
const detailCache = new Map();   // id -> detail
function fetchDetail(id) {
  if (detailCache.has(id)) return Promise.resolve(detailCache.get(id));
  const r = window.TANGSHI_DATA.points[id];
  const d = {
    id: id, title: r[0], author: r[1], era: r[2], form: r[3],
    tags: r[4], paragraphs: r[5],
    bio: window.TANGSHI_DATA.bios[r[1]] || "",
    background: "", analysis: "",
    macro: r[8], macro_name: MACRO_NAMES[r[8]], fine: r[9],
    fine_name: window.TANGSHI_DATA.fineNames[String(r[9])] || "",
  };
  detailCache.set(id, d);
  return Promise.resolve(d);
}""")
    # 2) 初始化：fetch /api/points -> 直接读内置数据
    js = replace_once(js,
        'fetch("/api/points").then(r => r.json()).then(data => {',
        "(function init(data) {")
    js = replace_once(js,
        "  POEMS = data.points.map(([i, t, a, tx, ty, m, f]) => ({ i, t, a, tx, ty, m, f }));",
        "  POEMS = data.points.map((r, i) => ({ i, t: r[0], a: r[1], tx: r[6], ty: r[7], m: r[8], f: r[9] }));")
    js = replace_once(js, "  resize();\n  syncFromURL();  // 支持刷新/分享链接直达某首诗\n});",
        "  resize();\n  syncFromURL();  // 支持刷新/分享链接直达某首诗\n})(window.TANGSHI_DATA);")
    # 3) 副标题
    js = replace_once(js, " · PCA→t-SNE · SQLite`;", " · PCA→t-SNE · 离线版`;")
    # 4) 触屏放大命中半径
    js = replace_once(js, "  let best = -1, bestD = 64;",
        "  let best = -1, bestD = (window.matchMedia && matchMedia(\"(pointer: coarse)\").matches) ? 400 : 64; // 触屏放大命中半径")
    return js


def build_html(html):
    # viewport：viewport-fit=cover + 禁缩放（容器规范模板）
    html = replace_once(html,
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">')
    # 跨端适配：触摸
    html = replace_once(html,
        "  * { margin: 0; padding: 0; box-sizing: border-box; }",
        "  * { margin: 0; padding: 0; box-sizing: border-box; }\n"
        "  html { touch-action: manipulation; }")
    html = replace_once(html,
        "    height: 100vh; display: flex; flex-direction: column; overflow: hidden;\n  }",
        "    height: 100vh; display: flex; flex-direction: column; overflow: hidden;\n"
        "    -webkit-tap-highlight-color: transparent; -webkit-touch-callout: none;\n  }")
    # 安全区：面板底部
    html = replace_once(html,
        "    padding: 20px; overflow-y: auto;",
        "    padding: 20px 20px calc(20px + var(--safe-area-inset-bottom, env(safe-area-inset-bottom, 0px))); overflow-y: auto;")
    # 拆出内联脚本 -> 外置经典脚本（CSP 禁内联）
    head, rest = html.split("<script>", 1)
    js, tail = rest.split("</script>", 1)
    html = (head
            + '<script src="./assets/data.js"></script>\n'
            + '<script src="./assets/app.js"></script>'
            + tail)
    return html, js


def self_check(app_js, html):
    banned = ["fetch(", "XMLHttpRequest", "eval(", "new Function(",
              "WebSocket", "EventSource", "type=\"module\"",
              "import ", "export ", "http://", "https://",
              "window.open(", "onclick=", "<iframe", "<base "]
    for pat in banned:
        assert pat not in app_js, f"app.js 命中禁用模式: {pat}"
        assert pat not in html, f"index.html 命中禁用模式: {pat}"


def main():
    with open(SRC_HTML, encoding="utf-8") as f:
        html = f.read()
    html, js = build_html(html)
    app_js = build_app_js(js)
    data_js = build_data_js()
    self_check(app_js, html)

    assets = os.path.join(BUILD_DIR, "assets")
    os.makedirs(assets, exist_ok=True)
    with open(os.path.join(BUILD_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(assets, "app.js"), "w", encoding="utf-8") as f:
        f.write(app_js)
    with open(os.path.join(assets, "data.js"), "w", encoding="utf-8") as f:
        f.write(data_js)

    # 压缩的是 minitool/ 的内容（index.html 在 zip 根），不是目录本身
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(BUILD_DIR):
            for name in files:
                if name == ".DS_Store":
                    continue
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, BUILD_DIR))

    data_kb = len(data_js.encode("utf-8")) // 1024
    zip_kb = os.path.getsize(ZIP_PATH) // 1024
    print(f"data.js {data_kb} KB (内置 {len(json.loads(data_js[len('window.TANGSHI_DATA = '):-2])['points'])} 首)")
    print(f"tangshi-minitool.zip {zip_kb} KB (上限 {ZIP_LIMIT // 1024 // 1024} MB)")
    assert os.path.getsize(ZIP_PATH) <= ZIP_LIMIT, "超出容器 10MB 上限"


if __name__ == "__main__":
    main()
