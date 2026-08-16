"""用 Qwen3-Embedding-0.6B 为每首诗抽取 embedding，缓存到 embeddings.npy。"""
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
POEMS_PATH = os.path.join(DATA_DIR, "poems.jsonl")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def main():
    with open(POEMS_PATH, encoding="utf-8") as f:
        poems = [json.loads(line) for line in f]
    texts = [f"{p['title']}。{p['content']}" for p in poems]
    print(f"共 {len(texts)} 首诗待编码")

    # macOS 13 的 MPS 不支持 bf16，强制 float32
    model = SentenceTransformer(
        MODEL_NAME, device="mps", model_kwargs={"dtype": "float32"})
    model.max_seq_length = 512  # 诗作已过滤到 500 字以内
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(EMB_PATH, embeddings)
    print(f"embeddings 形状: {embeddings.shape} -> {EMB_PATH}")


if __name__ == "__main__":
    main()
