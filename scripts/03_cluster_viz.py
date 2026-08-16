"""KMeans 聚类 + PCA / t-SNE 降维可视化。

输出:
  outputs/clusters.json        每首诗的簇标签
  outputs/cluster_summary.txt  每个簇的代表诗作
  outputs/pca.png / tsne.png   静态散点图
  outputs/pca.html / tsne.html 交互式散点图（hover 显示诗名/作者/诗句）
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
N_CLUSTERS = 20
RANDOM_STATE = 42

# 由模型阅读各簇代表诗作后拟定的类名（23986 首版本）
CLUSTER_NAMES = {
    0: "咏鸟寄兴", 1: "秋日感怀", 2: "离情别绪", 3: "舟行旅泊", 4: "山寺访僧",
    5: "春日感兴", 6: "宫词怀古", 7: "池亭游赏", 8: "寒山禅诗", 9: "咏物花木",
    10: "闲居田园", 11: "送别赠答", 12: "夜月怀思", 13: "乐府古题", 14: "客路离怀",
    15: "山水寻隐", 16: "应制投赠", 17: "乐府杂曲", 18: "老病咏怀", 19: "边塞军旅",
}

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "poems.jsonl"), encoding="utf-8") as f:
        poems = [json.loads(line) for line in f]
    emb = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    print(f"诗数: {len(poems)}, embedding 维度: {emb.shape[1]}")

    # ---------- KMeans 聚类 ----------
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(emb)

    # ---------- 降维：先 PCA 到 50 维降噪，再 t-SNE 到 2 维 ----------
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(emb)
    emb_pca50 = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(emb)
    tsne_2d = TSNE(
        n_components=2, random_state=RANDOM_STATE, perplexity=30,
        init="pca", learning_rate="auto",
    ).fit_transform(emb_pca50)

    df = pd.DataFrame({
        "author": [p["author"] for p in poems],
        "title": [p["title"] for p in poems],
        "content": [p["content"] for p in poems],
        "cluster": labels,
    })
    df["cluster_name"] = df["cluster"].map(
        lambda c: f"{c} {CLUSTER_NAMES.get(c, '')}")

    # ---------- 保存聚类结果 ----------
    df.to_json(os.path.join(OUT_DIR, "clusters.json"),
               orient="records", force_ascii=False, indent=1)

    # 每个簇：按离簇中心距离排序，取前 10 首作为代表
    with open(os.path.join(OUT_DIR, "cluster_summary.txt"), "w", encoding="utf-8") as f:
        for c in range(N_CLUSTERS):
            idx = np.where(labels == c)[0]
            center = km.cluster_centers_[c]
            dists = np.linalg.norm(emb[idx] - center, axis=1)
            top = idx[np.argsort(dists)[:10]]
            poets = pd.Series([poems[i]["author"] for i in idx]).value_counts()
            top_poets = "、".join(f"{a}({n})" for a, n in poets.head(5).items())
            f.write(f"簇 {c} 【{CLUSTER_NAMES.get(c, '')}】| {len(idx)} 首 | 主要诗人: {top_poets}\n")
            for i in top:
                f.write(f"  《{poems[i]['title']}》 {poems[i]['author']}: "
                        f"{poems[i]['content'][:40]}…\n")
            f.write("\n")

    # ---------- 可视化 ----------
    for name, coords in [("pca", pca_2d), ("tsne", tsne_2d)]:
        df[f"{name}_x"], df[f"{name}_y"] = coords[:, 0], coords[:, 1]

        # 静态图（图例显示类名）
        fig, ax = plt.subplots(figsize=(14, 10))
        cmap = plt.get_cmap("tab20")
        for c in range(N_CLUSTERS):
            mask = labels == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=cmap(c / N_CLUSTERS), s=6, alpha=0.7,
                       label=f"{c} {CLUSTER_NAMES.get(c, '')}")
        ax.set_title(f"全唐诗名家诗作 embedding - {name.upper()} (KMeans k={N_CLUSTERS})")
        ax.legend(title="簇", loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, markerscale=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 交互图（按类名着色，图例可点击筛选）
        df["hover"] = ("《" + df["title"] + "》 " + df["author"]
                       + "<br>" + df["content"].str.slice(0, 50) + "…")
        fig = px.scatter(
            df, x=f"{name}_x", y=f"{name}_y",
            color="cluster_name",
            hover_name="hover",
            hover_data={f"{name}_x": False, f"{name}_y": False, "cluster_name": False},
            title=f"全唐诗名家诗作 embedding - {name.upper()} (KMeans k={N_CLUSTERS})",
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(legend_title_text="簇")
        fig.write_html(os.path.join(OUT_DIR, f"{name}.html"))
        print(f"{name} 图已保存")

    df.to_csv(os.path.join(OUT_DIR, "poems_with_cluster.csv"),
              index=False, encoding="utf-8-sig")
    print("全部输出完成 ->", OUT_DIR)


if __name__ == "__main__":
    main()
