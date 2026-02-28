# Steerling 项目工作概览（不含 `linux_solution/`）

> 本文档基于当前仓库代码再次核对整理，范围明确排除 `linux_solution/` 目录。

## 1) 核对结论

你的原始梳理**整体准确**，主干结构与关键数据流都与代码一致。  
可补充的点主要有 3 个：

1. `models/interpretable/outputs.py` 定义了 `InterpretableOutput` 数据结构（承载 hidden/composed/known/unknown 等完整输出），在“可解释输出”层面很关键。
2. `models/layers/primitives.py` 包含 `RMSNorm`、`RotaryEmbedding`、`MLP`，是 `causal_diffusion_layers.py` 的核心底层依赖。
3. 推理层还包含 `GenerationOutput`（在 `inference/causal_diffusion.py` 中定义，并由 `inference/__init__.py` 导出），是 `generate_full()` 的结构化返回类型。

---

## 2) 项目概览

Steerling 是一个可解释的因果扩散语言模型（inference-only release），核心能力包括：

- 文本生成：基于置信度的非自回归 unmasking
- 概念分解：将 hidden 分解为 known / unknown / epsilon 相关分量
- 概念调控：通过 known / unknown 概念干预改变生成分布
- 嵌入提取：支持 hidden / composed / known / unknown 多视角表示

---

## 3) 模块结构（ASCII）

```text
+----------------------------------------------------------------------------------+
|                        STEERLING 项目架构（非 linux_solution）                    |
+----------------------------------------------------------------------------------+

  +--------------------------------------------------------------------------+
  | 入口 / 用户 API                                                          |
  | steerling/__init__.py                                                    |
  | -> SteerlingGenerator, CausalDiffusionConfig, ConceptConfig,             |
  |    GenerationConfig                                                       |
  +--------------------------------------------------------------------------+
                                      |
                                      v
  +--------------------------------------------------------------------------+
  | 推理层 (steerling/inference/)                                            |
  | - causal_diffusion.py                                                    |
  |   -> SteerlingGenerator                                                  |
  |   -> GenerationOutput                                                    |
  |   -> from_pretrained / generate / generate_full / get_embeddings         |
  | - checkpoint_utils.py                                                    |
  |   -> load_config / load_state_dict                                       |
  +--------------------------------------------------------------------------+
                      |                        |                      |
                      v                        v                      v
  +------------------------+    +--------------------------+    +------------------------------+
  | configs/               |    | data/                    |    | models/                      |
  | - causal_diffusion.py  |    | - tokenizer.py           |    | - causal_diffusion.py        |
  | - concept.py           |    |   (cl100k_base + 4特种)  |    | - interpretable/             |
  | - generation.py        |    |                          |    |   - interpretable_causal...  |
  +------------------------+    +--------------------------+    |   - concept_head.py          |
                                                                 |   - outputs.py               |
                                                                 | - layers/                    |
                                                                 |   - causal_diffusion_layers.py|
                                                                 |   - primitives.py            |
                                                                 +------------------------------+
                                      |
                                      v
  +--------------------------------------------------------------------------+
  | 工具与示例                                                                |
  | - scripts/convert_weights.py  (ScaleX -> Steerling safetensors/config)   |
  | - notebooks/simple_walk_through.ipynb                                     |
  +--------------------------------------------------------------------------+
                                      |
                                      v
  +--------------------------------------------------------------------------+
  | 测试与 CI                                                                  |
  | - tests/: conftest.py, test_config.py, test_tokenizer.py,                |
  |          test_model.py, test_generate.py                                  |
  | - .github/workflows/ci.yml: uv + ruff + pytest                            |
  +--------------------------------------------------------------------------+
```

---

## 4) 数据流（核对版）

### [1] 模型加载

1. `SteerlingGenerator.from_pretrained(model_name_or_path)`
2. `load_config()` 读取 `config.json`（本地目录或 HuggingFace Hub）
3. 解析：
   - `CausalDiffusionConfig`
   - 若 `interpretable=true` 且有 `concept`，则解析 `ConceptConfig`
   - `vocab_size` 与 tokenizer 相关字段
4. 构建模型：
   - `CausalDiffusionLM` 或 `InterpretableCausalDiffusionLM`
5. `load_state_dict()` 加载 safetensors（单文件或分片）
6. 恢复/处理权重绑定（weight tying）并返回 `SteerlingGenerator`

### [2] 文本生成（confidence-based unmasking）

1. `tokenizer.encode(prompt, add_special_tokens=False)` 得到 `prompt_ids`
2. 初始化序列：`[prompt_ids | mask | mask | ...]`
3. 每轮：
   - 前向得到 `logits`
   - 仅在 masked 位置计算置信度（softmax 后最大概率）
   - 选取最高置信度位置（`tokens_per_step`）
   - 对每个位置做 top-p 采样并回填 token
4. 直到达到 `max_new_tokens` 或触发 EOS 终止条件
5. `tokenizer.decode()` 输出文本  
6. `generate_full()` 额外返回 `GenerationOutput`（text/tokens/prompt_tokens/generated_tokens）

### [3] 可解释概念分解（Interpretable）

对 `input_ids` 前向时：

1. `transformer(..., return_hidden=True)` 得到 `hidden`
2. `known_head(hidden)` 得到 `known_features`
3. `unk = hidden - known_features.detach()`
4. `unknown_head(hidden.detach())`（可选）得到 `unk_hat`
5. `composed = known_features + unk_for_lm`
6. `lm_head(composed)` 得到最终 `logits`
7. 封装 `InterpretableOutput`（hidden/known/unk/composed/epsilon/top-k等）

关系可概括为：

```text
hidden ~= known_features + unk_hat + epsilon
composed = known_features + unk_for_lm
logits = lm_head(composed)
```

### [4] 概念调控（Steering）

1. 用户通过 `GenerationConfig` 传入：
   - `steer_known={concept_id: weight}`
   - `steer_unknown={concept_id: weight}`
2. 推理层构造 intervention tensors
3. 传入 interpretable 模型前向
4. 在 `ConceptHead` 中对指定概念激活做干预
5. 干预 `composed`，进而改变 `logits` 与生成文本

### [5] 嵌入提取

1. `get_embeddings(text, pooling, embedding_type)`
2. interpretable 模型下可取：
   - `hidden`
   - `composed`
   - `known_features`
   - `unk_hat`（或回退 `unk`）
3. 支持 `mean | last | first | none` pooling
4. 返回 `[D]` 或 `[T, D]`

---

## 5) 模块职责小结

| 模块 | 文件 | 职责 |
|---|---|---|
| 包入口 | `steerling/__init__.py` | 暴露公共 API（Generator + Configs） |
| 推理 | `steerling/inference/causal_diffusion.py` | 加载模型、生成、嵌入提取、概念调控、`GenerationOutput` |
| Checkpoint | `steerling/inference/checkpoint_utils.py` | 读取 `config.json`，加载单文件/分片 safetensors |
| 配置 | `steerling/configs/*.py` | `CausalDiffusionConfig` / `ConceptConfig` / `GenerationConfig`（Pydantic 校验） |
| 分词 | `steerling/data/tokenizer.py` | `cl100k_base` + `pad/bos/endofchunk/mask` |
| 基础模型 | `steerling/models/causal_diffusion.py` | block-causal 扩散 Transformer 主干 |
| 可解释模型 | `steerling/models/interpretable/interpretable_causal_diffusion.py` | known/unknown 概念分解与组合 |
| 概念头 | `steerling/models/interpretable/concept_head.py` | top-k 概念特征、干预入口 |
| 输出结构 | `steerling/models/interpretable/outputs.py` | `InterpretableOutput` 数据容器 |
| 层实现 | `steerling/models/layers/causal_diffusion_layers.py` | `BlockCausalAttention`、`CausalDiffusionBlock`、FlexAttention/SDPA |
| 层原语 | `steerling/models/layers/primitives.py` | `RMSNorm`、`RotaryEmbedding`、`MLP` |
| 权重转换 | `scripts/convert_weights.py` | ScaleX checkpoint -> Steerling safetensors + config |
| 示例 | `notebooks/simple_walk_through.ipynb` | 加载、生成、归因、嵌入提取示例 |
| 测试 | `tests/*.py` | config/tokenizer/model/generator 核心单测 |
| CI | `.github/workflows/ci.yml` | `uv` 环境下执行 `ruff check/format --check` 与 `pytest` |

---

## 6) 工具链与依赖（代码层核对）

- 包管理与构建：`uv` + `pyproject.toml`（`hatchling`）
- 核心依赖：PyTorch 2.8、Triton、safetensors、tiktoken、huggingface-hub、Pydantic、NumPy
- 质量保障：Ruff + Pytest（GitHub Actions CI）

---

## 7) 范围说明

本梳理明确**不包含** `linux_solution/` 下的脚本、notebook、结果文件与实验内容。
