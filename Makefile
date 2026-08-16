# 全唐诗名家诗作语义图谱
# 快速开始: make build && make run -> http://127.0.0.1:8899/

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
PORT := 8899

.PHONY: build run data embed cluster db clean

# 创建虚拟环境并安装依赖（幂等）
build: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

# 启动交互页面服务（仅需 data/poems.db，纯标准库，无重依赖）
run:
	$(PY) scripts/08_serve.py $(PORT)

# ---- 从零重建流水线（一般不需要，仓库已带 data/poems.db） ----

# 克隆数据源并构建诗作集
data:
	git clone --depth 1 --filter=blob:none --sparse \
		https://github.com/chinese-poetry/chinese-poetry data/chinese-poetry
	cd data/chinese-poetry && git sparse-checkout set 全唐诗
	$(PY) scripts/01_build_dataset.py

# 抽取 embedding（Qwen3-Embedding-0.6B，断点续跑）
embed:
	$(PY) scripts/02_embed.py
	$(PY) scripts/02b_embed_incremental.py

# PCA→t-SNE 降维 + 两层聚类（k=100 细簇 -> Ward 归并 20 主题）
cluster:
	$(PY) scripts/03_cluster_viz.py
	$(PY) scripts/06_two_level.py

# 汇总元信息写入 SQLite
db:
	$(PY) scripts/07_build_db.py

clean:
	rm -rf $(VENV)
