"""从全唐诗中筛选名家诗作，生成 poems.jsonl。

每条记录: {"author": ..., "title": ..., "content": ...}
"""
import glob
import json
import os
import re

from opencc import OpenCC

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TANG_DIR = os.path.join(DATA_DIR, "chinese-poetry", "全唐诗")
OUT_PATH = os.path.join(DATA_DIR, "poems.jsonl")

# 名家名单（初唐、盛唐、中唐、晚唐代表诗人）
FAMOUS_POETS = {
    # 初唐
    "王勃", "杨炯", "卢照邻", "骆宾王", "陈子昂", "张九龄", "贺知章", "张若虚",
    "沈佺期", "宋之问",
    # 盛唐
    "李白", "杜甫", "王维", "孟浩然", "王昌龄", "王之涣", "岑参", "高适",
    "崔颢", "常建", "刘长卿", "韦应物", "王翰", "李颀", "储光羲",
    # 中唐
    "白居易", "元稹", "刘禹锡", "柳宗元", "韩愈", "孟郊", "贾岛", "李贺",
    "张继", "卢纶", "李益", "司空曙", "戴叔伦", "顾况", "张籍", "王建",
    # 晚唐
    "李商隐", "杜牧", "温庭筠", "许浑", "韦庄", "杜荀鹤", "张祜", "马戴",
    "郑谷", "罗隐", "皮日休", "陆龟蒙",
    # 女诗人、僧诗人等
    "薛涛", "鱼玄机", "寒山", "拾得", "皎然", "贯休", "齐己",
}

MIN_LEN, MAX_LEN = 8, 500  # 诗句总字数过滤：去掉残句和超长篇（如长恨歌）


def normalize(text: str) -> str:
    # 去掉括号注文与空白
    text = re.sub(r"[（(].*?[)）]", "", text)
    return re.sub(r"\s+", "", text)


def main():
    files = sorted(glob.glob(os.path.join(TANG_DIR, "poet.tang.*.json")))
    assert files, f"未找到全唐诗数据文件: {TANG_DIR}"

    seen = set()
    n_kept = 0
    cc = OpenCC("t2s")  # 繁体 -> 简体；去重用繁体原值，保证与已缓存 embedding 行序一致
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                poems = json.load(f)
            for p in poems:
                author = p.get("author", "").strip()
                # 源数据为繁体，统一转简体后再匹配名家名单
                if cc.convert(author) not in FAMOUS_POETS:
                    continue
                content = normalize("".join(p.get("paragraphs", [])))
                if not (MIN_LEN <= len(content) <= MAX_LEN):
                    continue
                key = (author, content)
                if key in seen:
                    continue
                seen.add(key)
                out.write(json.dumps(
                    {"author": cc.convert(author),
                     "title": cc.convert(p.get("title", "").strip()),
                     "content": cc.convert(content),
                     "paragraphs": [cc.convert(x.strip())
                                    for x in p.get("paragraphs", []) if x.strip()]},
                    ensure_ascii=False) + "\n")
                n_kept += 1

    print(f"共保留 {n_kept} 首名家诗作 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
