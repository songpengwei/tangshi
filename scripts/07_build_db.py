"""把全部诗作元信息写入 SQLite: data/poems.db

字段来源与 04_interactive.py 一致:
- 正文/坐标/两层簇: poems.jsonl + outputs/poems_with_cluster.csv（行序一致）
- 诗人生平: 全唐诗/authors.tang.json（存全文，前端负责截断展开）
- 时代: 名家名单分组; 体裁: 句式推定; 标签: 唐诗三百首选篇
- analysis 列预留给 09_generate_analysis.py 增量填充
"""
import json
import os
import re
import sqlite3

import pandas as pd
from opencc import OpenCC

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
TANG_DIR = os.path.join(DATA_DIR, "chinese-poetry", "全唐诗")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(DATA_DIR, "poems.db")

POET_ERA = {}
for era, names in {
    "初唐": "王勃 杨炯 卢照邻 骆宾王 陈子昂 张九龄 贺知章 张若虚 沈佺期 宋之问",
    "盛唐": "李白 杜甫 王维 孟浩然 王昌龄 王之涣 岑参 高适 崔颢 常建 刘长卿 韦应物 王翰 李颀 储光羲",
    "中唐": "白居易 元稹 刘禹锡 柳宗元 韩愈 孟郊 贾岛 李贺 张继 卢纶 李益 司空曙 戴叔伦 顾况 张籍 王建 薛涛 寒山 拾得 皎然",
    "晚唐": "李商隐 杜牧 温庭筠 许浑 韦庄 杜荀鹤 张祜 马戴 郑谷 罗隐 皮日休 陆龟蒙 鱼玄机 贯休 齐己",
}.items():
    for n in names.split():
        POET_ERA[n] = era


def normalize(text: str) -> str:
    text = re.sub(r"[（(].*?[)）]", "", text)
    return re.sub(r"\s+", "", text)


def guess_form(paragraphs):
    clauses = re.split(r"[。！？；，、]", "".join(paragraphs))
    clauses = [c for c in clauses if c.strip()]
    if not clauses:
        return "未知"
    lens = [len(c) for c in clauses]
    uniform = max(set(lens), key=lens.count)
    ratio = lens.count(uniform) / len(lens)
    if uniform not in (5, 7) or ratio < 0.9:
        return "杂言古体"
    yan = "五言" if uniform == 5 else "七言"
    n = len(clauses)
    if n == 4:
        return yan + "绝句"
    if n == 8:
        return yan + "律诗"
    return yan + ("排律" if n % 2 == 0 else "古诗")


def load_bios(cc):
    bios = {}
    with open(os.path.join(TANG_DIR, "authors.tang.json"), encoding="utf-8") as f:
        for a in json.load(f):
            bios[cc.convert(a["name"])] = cc.convert(a.get("desc", ""))
    return bios


def load_tags(cc):
    tags = {}
    with open(os.path.join(TANG_DIR, "唐诗三百首.json"), encoding="utf-8") as f:
        for p in json.load(f):
            key = (cc.convert(p["author"]),
                   cc.convert(normalize("".join(p["paragraphs"]))))
            tags[key] = [cc.convert(t) for t in p.get("tags", []) if t != "唐诗三百首"]
    return tags


def main():
    with open(os.path.join(DATA_DIR, "poems.jsonl"), encoding="utf-8") as f:
        poems = [json.loads(line) for line in f]
    df = pd.read_csv(os.path.join(OUT_DIR, "poems_with_cluster.csv"))
    assert len(poems) == len(df), "poems.jsonl 与 csv 行数不一致"
    cc = OpenCC("t2s")
    bios = load_bios(cc)
    tags300 = load_tags(cc)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE poems (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            paragraphs TEXT NOT NULL,
            era TEXT, form TEXT, tags TEXT, bio TEXT,
            analysis TEXT NOT NULL DEFAULT '',
            tx REAL, ty REAL,
            fine INTEGER, macro INTEGER, macro_name TEXT
        )""")
    conn.execute("CREATE INDEX idx_macro ON poems(macro)")

    n_tags = 0
    rows = []
    for i, (poem, r) in enumerate(zip(poems, df.itertuples())):
        key = (poem["author"], poem["content"])
        tags = tags300.get(key, [])
        if tags:
            n_tags += 1
        rows.append((
            i, poem["title"], poem["author"],
            json.dumps(poem["paragraphs"], ensure_ascii=False),
            POET_ERA.get(poem["author"], ""),
            guess_form(poem["paragraphs"]),
            json.dumps(tags, ensure_ascii=False),
            bios.get(poem["author"], ""),
            round(float(r.tsne_x), 3), round(float(r.tsne_y), 3),
            int(r.fine), int(r.macro), str(r.macro_name),
        ))
    conn.executemany(
        "INSERT INTO poems (id,title,author,paragraphs,era,form,tags,bio,"
        "tx,ty,fine,macro,macro_name) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM poems").fetchone()[0]
    size_kb = os.path.getsize(DB_PATH) // 1024
    conn.close()
    print(f"poems.db: {n} 首, {size_kb} KB, 含三百首标签 {n_tags} 首")


if __name__ == "__main__":
    main()
