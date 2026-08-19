"""为首屏入口词预计算候选诗（Qwen3-Embedding-0.6B，与 02_embed.py 同一模型）。

每个生活化入口词（思乡/爱情/山水…）encode 成 query，与全库诗向量算余弦，
作者去重 + 名家优先，取 top 8，输出 outputs/entries.json。

用法: /path/to/venv/bin/python scripts/11_entries.py
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
POEMS_PATH = os.path.join(BASE_DIR, "data", "poems.jsonl")
EMB_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "entries.json")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_PROMPT = "Instruct: 根据描述检索语义最贴近的唐诗\nQuery: "

FAMOUS = {
    "李白", "杜甫", "王维", "白居易", "李商隐", "杜牧", "孟浩然", "王昌龄",
    "高适", "岑参", "刘禹锡", "韩愈", "柳宗元", "韦应物", "李贺", "温庭筠",
    "王之涣", "王勃", "骆宾王", "陈子昂", "张九龄", "贾岛", "孟郊", "元稹",
    "张继", "刘长卿", "韦庄", "皮日休", "陆龟蒙", "常建", "王翰", "王湾",
}

# 入口词 -> 检索描述
ENTRIES = [
    ("思乡", "思念故乡，羁旅途中浓浓的乡愁"),
    ("爱情", "男女相思之情，闺中思念心上人"),
    ("山水", "山水自然风光，风景优美"),
    ("孤独", "孤独寂寞，独自一人独坐独酌"),
    ("春天", "春天的景色，春日花开鸟鸣"),
    ("月亮", "夜晚明月高悬，望月抒怀"),
    ("饮酒", "饮酒作乐，把酒言欢"),
    ("边塞", "边塞风光，塞外征战"),
    ("秋天", "秋天萧瑟的景色，秋日感怀"),
    ("离别", "送别友人，离愁别绪"),
]

TOP_N = 8


def main():
    with open(POEMS_PATH, encoding="utf-8") as f:
        poems = [json.loads(line) for line in f]
    emb = np.load(EMB_PATH)
    assert len(poems) == emb.shape[0], "诗与向量行数不一致"

    model = SentenceTransformer(
        MODEL_NAME, device="mps", model_kwargs={"dtype": "float32"})
    model.max_seq_length = 512

    queries = [q for _, q in ENTRIES]
    q_emb = model.encode(queries, prompt=QUERY_PROMPT,
                         normalize_embeddings=True, show_progress_bar=False)
    sims = q_emb @ emb.T

    out = []
    for (word, _), sim in zip(ENTRIES, sims):
        order = np.argsort(-sim)
        used_authors = set()
        picked = []
        for j in order:
            p = poems[j]
            if p["author"] in used_authors:
                continue
            used_authors.add(p["author"])
            bonus = 0.02 if p["author"] in FAMOUS else 0.0
            picked.append((int(j), float(sim[j]) + bonus, p))
            if len(picked) >= TOP_N * 2:
                break
        picked.sort(key=lambda x: -x[1])
        picks = picked[:TOP_N]
        out.append({"word": word, "poemIds": [j for j, _, _ in picks]})
        print(f"{word}: " + " / ".join(f"{p['author']}《{p['title']}》" for _, _, p in picks))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
