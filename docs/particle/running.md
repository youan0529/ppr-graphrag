# 运行说明

从仓库根目录运行。构图、检索和编码服务均以前台进程执行，可由终端复用器或任务系统管理。

## 数据和环境

安装 `pip install -e .`。使用 `scripts/convert_dataset.py --help` 查看原有数据转换工具。语料格式：

```json
{"doc_id":"doc-1","title":"Article title","text":"Article body"}
```

问题格式：

```json
{"query_id":"q-1","question":"Question?","answers":["answer"],"supporting_doc_ids":["doc-1"]}
```

默认数据路径位于 `data/processed/twowiki_validation/` 或 `data/processed/hotpotqa_distractor_validation/`。问题文件支持 JSONL 或 JSON 数组。数据集和模型权重不随 Git 提交。

编辑 `configs/particle/twowiki.yaml` 或 `hotpotqa.yaml`。`embedding.python` 指向加载 NV-Embed-v2 的 Python 环境，`embedding.model_path` 指向完整的固定 revision 本地快照。代码使用离线加载和模型自带代码；`HF_HOME` 可用环境变量指定。

已运行的编码环境包含 torch 2.5.1、transformers 4.45.2、accelerate 1.13.0、einops 0.8.2、numpy 1.26.4，运行器使用 tiktoken 0.7.0。这是已用环境记录，不代表所有最新版依赖组合均已验证。

## 模型服务

构图和 QA 只借用已启动并加载模型的 Ollama 服务，不启动或卸载 LLM。每个端口一次处理一个文章或问题任务。模型需完整驻留 GPU，digest 和上下文长度必须匹配配置。

在分配给本任务的 GPU 和端口上启动服务，使用 `OLLAMA_NUM_PARALLEL=1`、`OLLAMA_CONTEXT_LENGTH=16384`，提前加载配置中的模型。GPU 位置由服务进程的 `CUDA_VISIBLE_DEVICES` 控制，`--ports` 只选择服务。已有实验使用 Ollama 0.18.2 的模板和结构化输出行为；其他版本需核对兼容性。

`execution.lock_directory` 应由共享同一组模型服务的运行器共同使用。跨工作目录共享服务时，将它设为同一个绝对路径。新入口不会自动接管旧进程、旧锁或旧输出目录。

启动独立编码服务：

```bash
python scripts/serve_particle_embeddings.py \
  --config configs/particle/twowiki.yaml --port 11700
```

单实例使用 `embedding.cuda_visible_devices`。可加 `--parallel` 启用 `configs/particle/embedding_pool.json` 中的多卡策略，在大卡放独立实例，在有余量的小卡之间分片。该策略来自实验机器的 GPU 0/3 与 1/2/4 拓扑，换机器需检查策略。

编码服务仅绑定本机。构图和 QA 检查其模型 revision、维度及最大长度；任务退出后保留模型服务。编码服务自身退出时关闭它拥有的编码子进程。

## 构图与续跑

```bash
python scripts/build_particle_graph.py \
  --config configs/particle/twowiki.yaml \
  --ports 11434,11435 \
  --output experiments/particle_twowiki/graph
```

`--pilot` 使用固定种子的均匀文章样本，仅用于诊断。发布入口不依赖历史回归题清单或本地 pilot_review.json；全量方法参数没有改变。

文章独立缓存，逆向关系按批次保存。任务数据库维护已完成状态。同一配置和输出路径下重启会复用完成任务；改变方法、语料或配置应使用新输出目录。历史候选 manifest 的代码路径不同，不要用发布入口覆盖旧任务来续跑。

输出包括 manifest.json、status.json、jobs.sqlite、items/、extractions.jsonl、graph.json 和向量索引。graph_finished_with_errors 应结合错误清单查看，不等于每篇都成功。

## 检索与 QA

构图完成后运行：

```bash
python scripts/run_particle_qa.py \
  --config configs/particle/twowiki.yaml \
  --graph experiments/particle_twowiki/graph \
  --ports 11434,11435 \
  --output experiments/particle_twowiki/qa
```

HotpotQA 换用对应配置和独立输出目录。`--queries` 指定固定问题集，`--limit` 用于小规模诊断。检索面对整个语料图，不使用单道问题的支持文章标签定位证据。

按问题保存初始化、逐轮采样、检查和 Reader 输入输出及最终结果；全量汇总为 metrics.json、results.jsonl 和 status.json。重启跳过已完成问题。SIGTERM/SIGINT 请求停止派发并等待当前文章或问题完成，再保存状态。

已有 HotpotQA 冻结任务继续使用其原候选路径和启动脚本；发布正式目录不会重命名、迁移或重启它们。
