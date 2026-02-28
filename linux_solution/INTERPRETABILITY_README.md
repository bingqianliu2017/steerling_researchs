# Steerling-8B 可解释性实现梳理

## 一、当前实现 vs 模型能力

### 你现在的实现（test_steering_json.py）
- ✅ **文本生成**：根据 prompt 生成模型回复
- ✅ **Concept Steering**：通过 `steer_known` / `steer_unknown` 激活/抑制概念
- ❌ **Token 级概念归因**：`extract_token_info` 期望的 `known_features` / `unk_hat` / `epsilon` 全为 null

**原因**：`GenerationOutput.tokens` 只是 **token ID 的 Tensor**，不是带归因信息的对象。`generate_full` 内部使用 `minimal_output=True`，不保存概念分解结果。

### Steerling 宣称的三类可解释性

| 类型 | 描述 | 当前 PyPI 包支持情况 |
|------|------|----------------------|
| **1. Input 归因** | 哪些 prompt token 影响了输出 | ❌ 未提供 |
| **2. Concept 归因** | 人类可理解的概念（tone/content）对输出的贡献 | ⚠️ 可通过模型 forward 获取 |
| **3. Training Data 归因** | 输出源于哪些训练数据源（ArXiv/Wikipedia/FLAN） | ❌ **明确不支持**（lightweight 版本） |

> 官方 README: *"This release is a light-weight version of the pipeline, so it doesn't directly support training data attribution. ... If you're interested in supporting training data attribution, please reach out to Guide Labs."*

---

## 二、如何实现 Concept 归因（概念归因）

模型内部在 forward 时已计算 `InterpretableOutput`，包含：
- `known_topk_indices`：Top-k 已知概念 ID
- `known_topk_logits`：对应 logits
- `unknown_topk_indices` / `unknown_topk_logits`：未知概念
- `known_features` / `unk_hat` / `epsilon`：特征分解

**方案**：生成完成后，对 **完整序列**（prompt + 生成文本）再跑一次 forward，并设置 `minimal_output=False`，即可拿到每个位置的 concept 归因。

---

## 三、已实现功能示例（interpretability_demo.py）

`interpretability_demo.py` 演示：
1. **Embedding 提取**：`get_embeddings()` 获取 hidden / composed / known / unknown 表示
2. **Concept 归因**：对生成后的序列做一次 forward，提取 known_topk_indices / known_topk_logits
3. **当前无法实现**：Input attribution、Training data attribution（需等官方或联系 Guide Labs）
