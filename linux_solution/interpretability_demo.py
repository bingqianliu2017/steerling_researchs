# steerling 可解释性功能演示
# 实现：Concept 归因、Embedding 提取
# 参考：https://github.com/guidelabs/steerling

import torch
import json
from steerling import SteerlingGenerator, GenerationConfig

# -------------------------------
# 1. 模型加载
# -------------------------------
model_path = "/root/models/steerling-8b/guidelabs/steerling-8b"
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = SteerlingGenerator.from_pretrained(model_path, device=device)

if not generator.is_interpretable:
    raise RuntimeError("需要 interpretable 模型才能做概念归因，请确认加载的是 steerling-8b")

# -------------------------------
# 2. Embedding 提取（get_embeddings 官方 API）
# -------------------------------
def demo_embeddings():
    """演示：提取 hidden / composed / known / unknown 表示"""
    text = "Machine learning is a subset of artificial intelligence."
    for emb_type in ["hidden", "composed", "known", "unknown"]:
        emb = generator.get_embeddings(text, pooling="mean", embedding_type=emb_type)
        print(f"  {emb_type}: shape={emb.shape}, dtype={emb.dtype}")
    return generator.get_embeddings(text, pooling="none", embedding_type="known")

# -------------------------------
# 3. Concept 归因（对生成后的序列做 forward，minimal_output=False）
# -------------------------------
def get_concept_attribution(generator, full_text: str, top_k: int = 5):
    """
    对完整文本（prompt + 生成）做一次 forward，提取每个位置的 concept 归因。
    
    Returns:
        list of dict: 每个 token 位置的 known_topk 概念 ID 和 logits
    """
    token_ids = generator.tokenizer.encode(full_text, add_special_tokens=False)
    x = torch.tensor([token_ids], dtype=torch.long, device=generator.device)

    with torch.inference_mode():
        _, outputs = generator.model(
            x,
            use_teacher_forcing=False,
            minimal_output=False,  # 关键：获取 full InterpretableOutput
        )

    # outputs: InterpretableOutput
    # known_topk_indices: (1, T, k), known_topk_logits: (1, T, k)
    known_idx = outputs.known_topk_indices  # (1, T, k)
    known_logits = outputs.known_topk_logits

    if known_idx is None or known_logits is None:
        return []

    k = min(top_k, known_idx.shape[-1])
    results = []
    for pos in range(known_idx.shape[1]):
        idx = known_idx[0, pos, :k].cpu().tolist()
        logits = known_logits[0, pos, :k].cpu().tolist()
        results.append({
            "position": pos,
            "concept_ids": idx,
            "concept_logits": [round(l, 4) for l in logits],
        })
    return results

# -------------------------------
# 4. 主流程：生成 + 概念归因
# -------------------------------
def main():
    print("=" * 50)
    print("1. Embedding 提取")
    print("=" * 50)
    demo_embeddings()

    print("\n" + "=" * 50)
    print("2. 文本生成 + Concept 归因")
    print("=" * 50)

    prompt = "what is machine learning?"
    config = GenerationConfig(max_new_tokens=30, seed=42)
    output = generator.generate_full(prompt, config)

    full_text = prompt + " " + output.text
    print(f"Prompt: {prompt}")
    print(f"Generated: {output.text[:200]}...")
    print(f"Total tokens: {output.prompt_tokens + output.generated_tokens}")

    attributions = get_concept_attribution(generator, full_text, top_k=5)

    # 只展示生成部分的归因（prompt 之后）
    gen_attributions = attributions[output.prompt_tokens:output.prompt_tokens + min(10, output.generated_tokens)]
    print(f"\n前 10 个生成 token 的 top-5 概念 ID 与 logits:")
    for a in gen_attributions:
        print(f"  pos {a['position']}: concept_ids={a['concept_ids']}, logits={a['concept_logits']}")

    # 保存到 JSON
    result = {
        "prompt": prompt,
        "generated_text": output.text,
        "prompt_tokens": output.prompt_tokens,
        "generated_tokens": output.generated_tokens,
        "concept_attributions": attributions[:50],  # 限制长度
    }
    with open("interpretability_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到 interpretability_results.json")

if __name__ == "__main__":
    main()
