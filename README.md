# 全唐诗名家诗作语义图谱

23986 首唐诗名家诗作，用 Qwen3-Embedding-0.6B 抽取向量，KMeans k=100 细簇
经 Ward 层次聚类归并为 20 个主题，PCA→t-SNE 降维后以交互散点图呈现。

## 快速开始

```sh
make build   # 创建 .venv 并安装依赖
make run     # 启动服务，访问 http://127.0.0.1:8899/
```

仓库自带 `data/poems.db`（SQLite，含全部诗作元信息、两层聚类标签、
t-SNE 坐标），`make run` 无需任何额外数据即可直接跑。

## 交互

- 页眉 20 个主题标签（带诗作数量），点击高亮该主题并随机展示一首
- 锁定主题后面板顶部列出其下细类（共 100 个，名称+数量），点击跳转该类随机一首
- hover 点位查看诗作，点击锁定，再次点击取消
- 诗人小传超 140 字可展开全文

## 从零重建流水线

```sh
make data      # 克隆 chinese-poetry 数据源并构建 data/poems.jsonl
make embed     # 抽取 embedding（Qwen3-Embedding-0.6B，支持断点续跑）
make cluster   # PCA→t-SNE + 两层聚类 k=100→20
make db        # 汇总元信息写入 data/poems.db
```

## 目录

- `scripts/` — 流水线脚本（编号即顺序）
- `data/poems.jsonl` — 诗作原始数据（23986 首，简体）
- `data/poems.db` — SQLite 汇总库（含 fine/macro 标签、t-SNE 坐标、诗人小传）
- `outputs/index.html` — 交互页面（由 08_serve.py 提供服务）
- `outputs/poems_with_cluster.csv` — 聚类结果明细

## 数据来源

诗作文本与诗人小传来自
[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
（MIT License）全唐诗部分，经 OpenCC 转简体。embedding 模型为
[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
