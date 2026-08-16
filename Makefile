# 全唐诗名家诗作语义图谱
# 快速开始: make run -> http://127.0.0.1:8899/（零依赖，仅需 Python 3.9+）

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
PORT := 8899

.PHONY: run build-pipeline data embed cluster db clean

# 启动交互页面服务（仅需 data/poems.db，纯标准库，直接用系统 python3）
run:
	python3 scripts/08_serve.py $(PORT)

# ---- 以下为从零重建流水线（一般不需要，仓库已带 data/poems.db） ----

# 创建虚拟环境并安装流水线依赖（torch/sklearn 等，体积较大）
build-pipeline: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements-pipeline.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements-pipeline.txt
	@touch $(VENV)/bin/activate

# 克隆数据源并构建诗作集
data: build-pipeline
	git clone --depth 1 --filter=blob:none --sparse \
		https://github.com/chinese-poetry/chinese-poetry data/chinese-poetry
	cd data/chinese-poetry && git sparse-checkout set 全唐诗
	$(PY) scripts/01_build_dataset.py

# 抽取 embedding（Qwen3-Embedding-0.6B，断点续跑）
embed: build-pipeline
	$(PY) scripts/02_embed.py
	$(PY) scripts/02b_embed_incremental.py

# PCA→t-SNE 降维 + 两层聚类（k=100 细簇 -> Ward 归并 20 主题）
cluster: build-pipeline
	$(PY) scripts/03_cluster_viz.py
	$(PY) scripts/06_two_level.py

# 汇总元信息写入 SQLite
db: build-pipeline
	$(PY) scripts/07_build_db.py

clean:
	rm -rf $(VENV)
