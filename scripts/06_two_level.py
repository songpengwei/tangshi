"""两层聚类：KMeans k=100 细簇 -> 簇心 Ward 层次聚类归并为 20 个大主题。

复用 poems_with_cluster.csv 里的 t-SNE 坐标（t-SNE 与 k 无关，无需重算）。
输出:
  outputs/poems_with_cluster.csv  覆盖，新增 fine / macro 列
  outputs/cluster_summary.txt     大主题 -> 细簇 -> 代表诗作
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
K_FINE = 100
K_MACRO = 20
RANDOM_STATE = 42

# 大主题类名：由模型阅读各主题代表诗作后拟定
MACRO_NAMES = {
    0: "赠别咏怀", 1: "边塞军旅", 2: "春日感兴", 3: "皮陆唱和", 4: "舟行旅泊",
    5: "秋夜感怀", 6: "咏物花木", 7: "山寺访僧", 8: "寒山禅诗", 9: "应制颂赠",
    10: "山水寻隐", 11: "乐府相和", 12: "送别赠答", 13: "闲居园田", 14: "乐府杂曲",
    15: "宫词", 16: "咏鸟寄兴", 17: "元白唱和", 18: "饮酒宴集", 19: "虫豸讽喻",
}


def main():
    emb = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    df = pd.read_csv(os.path.join(OUT_DIR, "poems_with_cluster.csv"))
    with open(os.path.join(DATA_DIR, "poems.jsonl"), encoding="utf-8") as f:
        poems = [json.loads(line) for line in f]
    assert len(df) == len(emb) == len(poems)

    # ---- 细簇 ----
    km = KMeans(n_clusters=K_FINE, random_state=RANDOM_STATE, n_init=10)
    fine = km.fit_predict(emb)

    # ---- 归并：簇心 Ward 层次聚类 ----
    agg = AgglomerativeClustering(n_clusters=K_MACRO, linkage="ward")
    fine_to_macro = agg.fit_predict(km.cluster_centers_)
    macro = fine_to_macro[fine]

    df["fine"] = fine
    df["macro"] = macro
    df["macro_name"] = df["macro"].map(
        lambda c: f"{c} {MACRO_NAMES.get(c, '')}".strip())
    df["fine_name"] = df.apply(
        lambda r: f"{r['macro_name']}·细类{r['fine']}", axis=1)
    df.to_csv(os.path.join(OUT_DIR, "poems_with_cluster.csv"),
              index=False, encoding="utf-8-sig")

    # ---- 摘要：按大主题组织 ----
    with open(os.path.join(OUT_DIR, "cluster_summary.txt"), "w", encoding="utf-8") as f:
        for m in range(K_MACRO):
            midx = np.where(macro == m)[0]
            center = emb[midx].mean(axis=0)
            dists = np.linalg.norm(emb[midx] - center, axis=1)
            top = midx[np.argsort(dists)[:8]]
            poets = pd.Series([poems[i]["author"] for i in midx]).value_counts()
            top_poets = "、".join(f"{a}({n})" for a, n in poets.head(5).items())
            fines = df[df.macro == m].fine.value_counts()
            fine_list = "、".join(str(x) for x in fines.index.tolist())
            f.write(f"主题 {m} 【{MACRO_NAMES.get(m, '')}】| {len(midx)} 首 | "
                    f"主要诗人: {top_poets} | 细簇: {fine_list}\n")
            for i in top:
                f.write(f"  《{poems[i]['title']}》 {poems[i]['author']}: "
                        f"{poems[i]['content'][:40]}…\n")
            f.write("\n")
    print(f"细簇 {K_FINE} -> 主题 {K_MACRO} 完成 ->",
          os.path.join(OUT_DIR, "cluster_summary.txt"))


if __name__ == "__main__":
    main()
