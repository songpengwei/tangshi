"""生成 outputs/data.js，供交互页面使用。

元信息来源:
- 坐标/簇: outputs/poems_with_cluster.csv（与 poems.jsonl 行序一致）
- 诗人生平: 全唐诗/authors.tang.json
- 时代: 按名家名单分组（初唐/盛唐/中唐/晚唐）
- 体裁: 由句式程序推定（五言/七言 × 绝句/律诗/排律，余为古体杂言）
- 题材标签: 匹配《唐诗三百首》选篇（366 首有 tags）
"""
import json
import os
import re

import pandas as pd
from opencc import OpenCC

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
TANG_DIR = os.path.join(DATA_DIR, "chinese-poetry", "全唐诗")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

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
    """由句式推定体裁：返回 五言绝句/七言律诗/杂言古体 等"""
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
    """唐诗三百首 tags，按 (作者, 规范化正文) 匹配；源数据为繁体，统一转简体"""
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
    bios = load_bios(OpenCC("t2s"))
    tags300 = load_tags(OpenCC("t2s"))

    records = []
    n_tags = 0
    for poem, r in zip(poems, df.itertuples()):
        key = (poem["author"], poem["content"])
        tags = tags300.get(key, [])
        if tags:
            n_tags += 1
        records.append({
            "t": poem["title"],
            "a": poem["author"],
            "pp": poem["paragraphs"],
            "m": int(r.macro),
            "mn": str(r.macro_name),
            "f": int(r.fine),
            "era": POET_ERA.get(poem["author"], ""),
            "form": guess_form(poem["paragraphs"]),
            "tags": tags,
            "bio": bios.get(poem["author"], "")[:300],
            "tx": round(float(r.tsne_x), 3), "ty": round(float(r.tsne_y), 3),
        })
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    out_path = os.path.join(OUT_DIR, "data.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("const POEMS = " + payload + ";\n")
    print(f"data.js: {len(records)} 首, {os.path.getsize(out_path) // 1024} KB, "
          f"含三百首标签 {n_tags} 首")


if __name__ == "__main__":
    main()
