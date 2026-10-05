# Particle evidence-chain retrieval

通过随机游走收集候选证据链，再由 LLM 联合检查、保留有用前缀并安排续走，最后读取来源文章回答多跳问题的研究实现。

当前实现按职责组织在 `ppr_graphrag/`，对应已完成 2Wiki 实验的第三版构图与 r11 检索 QA。代码路径不带实验迭代后缀；历史对应关系见[来源清单](docs/particle/provenance.json)。本次整理保留实际实验的方法与提示词。

## 阅读入口

| 内容 | 位置 |
|---|---|
| 方法与数据流 | [docs/particle/method.md](docs/particle/method.md) |
| 抽取、逆向关系、身份归并和建图 | [graph/](ppr_graphrag/graph/) 与 [joint_extraction.py](ppr_graphrag/llm/joint_extraction.py) |
| 起点匹配、粒子调度、检查和 QA | [retrieval/](ppr_graphrag/retrieval/) 与 [pipelines/](ppr_graphrag/pipelines/) |
| 全部模型提示词 | [prompts/](prompts/) |
| NV-Embed-v2 与多 GPU 编码 | [embedding/](ppr_graphrag/embedding/) |
| 配置 | [configs/particle/](configs/particle/) |
| 启动与断点续跑 | [docs/particle/running.md](docs/particle/running.md) |

```text
ppr_graphrag/
  graph/              实体归并、事实图构造与有向图访问
  retrieval/          粒子游走、起点调度、证据归档、链与来源组织
  llm/                联合抽取、检查器、输出约束、模型调用与 token 计数
  embedding/          NV-Embed-v2 编码进程与可选多卡调度
  pipelines/          构图调度、并行检索 QA、Reader 与评测流程
  core/               原有缓存与公共工具、模型服务连接
  evaluation/         原有 QA 与检索指标
  data/               原有数据转换工具
prompts/              当前八个提示词及早期通用提示词
scripts/
  build_particle_graph.py
  run_particle_qa.py
  serve_particle_embeddings.py
configs/particle/     2Wiki、HotpotQA 和编码调度配置
```

当前方法复用已有 SQLite 缓存、QA 评分和数据转换工具。粒子机制位于 [retrieval/particle.py](ppr_graphrag/retrieval/particle.py)，不另建平行的算法总目录。早期 Fact–Entity/PPR 实现仍可通过原脚本访问；本页列出的入口运行当前方法。历史候选、数据集、图、向量、缓存和完整调用日志不提交到 Git。

## 使用

```bash
pip install -e .
python scripts/build_particle_graph.py --help
python scripts/run_particle_qa.py --help
python scripts/serve_particle_embeddings.py --help
```

先按[运行说明](docs/particle/running.md)准备数据、NV 模型路径和已加载的 Ollama 服务，再运行：

```bash
python scripts/build_particle_graph.py --config configs/particle/twowiki.yaml
python scripts/run_particle_qa.py --config configs/particle/twowiki.yaml \
  --output experiments/particle_twowiki/qa
```

构图按文章并行，检索 QA 按问题并行。`--ports` 指定多个已经加载模型的端口；运行器不负责启动或卸载这些 LLM 服务。编码使用独立 NV-Embed-v2 服务。

## 已完成的 2Wiki 结果

相同的 1,000 道问题、同一 r11 检索 QA、最多五篇来源文章：

| 指标 | 第二版图 | 第三版图（当前方法） |
|---|---:|---:|
| 文章 Recall@5 | 83.90% | 88.80% |
| 全部标注来源覆盖 | 73.10% | 81.00% |
| QA EM | 54.30% | 57.00% |
| QA F1 | 62.69% | 65.84% |

原始汇总：[第二版](reports/particle/second_graph_metrics.json)、[第三版](reports/particle/third_graph_metrics.json)。这些是目录整理前冻结实验的结果，不是重组后重新跑出的分数。文章覆盖不等于证据链语义正确率，Reader 的 `valid` 也不是人工标注。

历史第三版图复用了部分已有逆向结果，包括历史 medium 调用；来源记录未改写。发布配置从零使用 low，默认不依赖这些本地历史缓存，因此不承诺从零重跑得到完全相同的图或分数。HotpotQA 使用相同方法，但这里没有发布其最终结果。
