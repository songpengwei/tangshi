"""k 值扫描：对 KMeans 用惯性（手肘法）和轮廓系数评估不同 k，输出 k_sweep.png。"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
RANDOM_STATE = 42

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

K_VALUES = [10, 20, 40, 60, 80, 100, 120, 155]


def main():
    emb = np.load(os.path.join(BASE_DIR, "data", "embeddings.npy"))
    n = len(emb)
    print(f"n={n}, sqrt(n/2)={np.sqrt(n/2):.0f}, sqrt(n)={np.sqrt(n):.0f}")

    # 轮廓系数全量太慢，抽 5000 个样本估算
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(n, 5000, replace=False)

    inertias, silhouettes = [], []
    for k in K_VALUES:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             batch_size=2048, n_init=3)
        labels = km.fit_predict(emb)
        inertias.append(km.inertia_)
        sil = silhouette_score(emb[sample_idx], labels[sample_idx],
                               metric="cosine", random_state=RANDOM_STATE)
        silhouettes.append(sil)
        print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(K_VALUES, inertias, "o-", color="#77AAC2", label="惯性（手肘法）")
    ax2.plot(K_VALUES, silhouettes, "s-", color="#F19A97", label="轮廓系数（样本估算）")
    for k, s in zip(K_VALUES, silhouettes):
        ax2.annotate(f"{s:.3f}", (k, s), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#F19A97")
    ax1.set_xlabel("k（簇数）")
    ax1.set_ylabel("惯性 inertia", color="#77AAC2")
    ax2.set_ylabel("轮廓系数 silhouette", color="#F19A97")
    ax1.set_title(f"KMeans k 值扫描（n={n}）")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.85))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "k_sweep.png"), dpi=150)
    print("->", os.path.join(OUT_DIR, "k_sweep.png"))


if __name__ == "__main__":
    main()
