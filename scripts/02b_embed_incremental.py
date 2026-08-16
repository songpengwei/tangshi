"""增量编码：名家名单 bug 修复后数据集从 10637 扩到 23986 首。

老数据（/tmp/poems_old.jsonl + /tmp/embeddings_old.npy，行序对齐）的 embedding 按
(作者, 正文) 键复用；只对新增诗作跑模型，结果按新 poems.jsonl 行序写回
data/embeddings.npy，并落盘 data/embeddings_keys.json 供以后增量复用。
"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OLD_POEMS = "/tmp/poems_old.jsonl"
OLD_EMB = "/tmp/embeddings_old.npy"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def key_of(p):
    return p["author"] + "\x00" + p["content"]


def main():
    poems = load_jsonl(os.path.join(DATA_DIR, "poems.jsonl"))
    old_poems = load_jsonl(OLD_POEMS)
    old_emb = np.load(OLD_EMB)
    assert len(old_poems) == len(old_emb)

    old_map = {}
    for i, p in enumerate(old_poems):
        old_map.setdefault(key_of(p), old_emb[i])

    todo = [i for i, p in enumerate(poems) if key_of(p) not in old_map]
    print(f"总数 {len(poems)}，复用 {len(poems) - len(todo)}，新增待编码 {len(todo)}")

    new_emb = np.empty((len(poems), old_emb.shape[1]), dtype=np.float32)
    for i, p in enumerate(poems):
        e = old_map.get(key_of(p))
        if e is not None:
            new_emb[i] = e

    if todo:
        # 断点续跑：新增诗作的 embedding 分批编码，每批落盘缓存，中断后可续
        cache_path = os.path.join(DATA_DIR, "embeddings_todo_cache.npz")
        done = {}
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=False)
            done = {k: v for k, v in zip(z["keys"], z["embs"])}
            print(f"从缓存续跑，已完成 {len(done)}")

        model = SentenceTransformer(
            MODEL_NAME, device="mps", model_kwargs={"dtype": "float32"})
        model.max_seq_length = 512

        remaining = [i for i in todo if key_of(poems[i]) not in done]
        # 按文本长度从长到短排序：同 batch 内 padding 相近，长短混排会慢数倍
        remaining.sort(key=lambda i: -len(poems[i]["content"]))
        texts = [f"{poems[i]['title']}。{poems[i]['content']}" for i in remaining]
        chunk = 640  # 每 10 个 batch 落一次盘
        for off in range(0, len(texts), chunk):
            emb = model.encode(texts[off:off + chunk], batch_size=64,
                               show_progress_bar=True, normalize_embeddings=True)
            for j, i in enumerate(remaining[off:off + chunk]):
                done[key_of(poems[i])] = emb[j]
            np.savez(cache_path,
                     keys=np.array(list(done.keys())),
                     embs=np.stack(list(done.values())))
        for i in todo:
            new_emb[i] = done[key_of(poems[i])]
        os.remove(cache_path)

    np.save(os.path.join(DATA_DIR, "embeddings.npy"), new_emb)
    with open(os.path.join(DATA_DIR, "embeddings_keys.json"), "w", encoding="utf-8") as f:
        json.dump([key_of(p) for p in poems], f, ensure_ascii=False)
    print(f"embeddings 形状: {new_emb.shape}")


if __name__ == "__main__":
    main()
