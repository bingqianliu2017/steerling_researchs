# steerling_full_export_complete.py
import torch
import json
from steerling import SteerlingGenerator, GenerationConfig

# -------------------------------
# 1️⃣ 模型加载
# -------------------------------
model_path = "/root/models/steerling-8b/guidelabs/steerling-8b"
device = "cuda" if torch.cuda.is_available() else "cpu"

generator = SteerlingGenerator.from_pretrained(model_path, device=device)
print(f"Loaded Steerling-8B on device: {device}")

# 用于保存全部生成结果
results = {}

# -------------------------------
# 2️⃣ 辅助函数：安全获取 token 属性
# -------------------------------
def extract_token_info(token_obj):
    """
    将 Steerling token 对象转换为 dict，支持 text / known_features / unk_hat / epsilon
    如果某些属性不存在，则返回 None
    """
    if isinstance(token_obj, dict):
        # 有些版本可能直接返回 dict
        t = token_obj
        return {
            "token": t.get("text"),
            "known_features": t.get("known_features"),
            "unk_hat": t.get("unk_hat"),
            "epsilon": t.get("epsilon")
        }
    else:
        # tensor 或自定义对象
        return {
            "token": getattr(token_obj, "text", None),
            "known_features": getattr(token_obj, "known_features", None).tolist() if getattr(token_obj, "known_features", None) is not None else None,
            "unk_hat": getattr(token_obj, "unk_hat", None).tolist() if getattr(token_obj, "unk_hat", None) is not None else None,
            "epsilon": getattr(token_obj, "epsilon", None).tolist() if getattr(token_obj, "epsilon", None) is not None else None
        }

# -------------------------------
# 3️⃣ 基础文本生成
# -------------------------------
prompt_1 = "what is machine learning?" 
config_1 = GenerationConfig(max_new_tokens=50, seed=42)
output_1 = generator.generate_full(prompt_1, config_1)

results['basic_generation'] = {
    "prompt": prompt_1,
    "text": getattr(output_1, "text", None),
    "total_tokens": len(getattr(output_1, "tokens", [])),
    "tokens": [extract_token_info(t) for t in getattr(output_1, "tokens", [])]
}

# -------------------------------
# 4️⃣ 高级参数生成
# -------------------------------
prompt_2 = "what is machine learning?"
config_2 = GenerationConfig(
    max_new_tokens=100,
    seed=123,
    top_p=0.9,
    repetition_penalty=1.2,
    use_entropy_sampling=True
)
output_2 = generator.generate_full(prompt_2, config_2)

results['advanced_generation'] = {
    "prompt": prompt_2,
    "text": getattr(output_2, "text", None),
    "config": {
        "max_new_tokens": config_2.max_new_tokens,
        "seed": config_2.seed,
        "top_p": config_2.top_p,
        "repetition_penalty": config_2.repetition_penalty,
        "use_entropy_sampling": config_2.use_entropy_sampling
    },
    "tokens": [extract_token_info(t) for t in getattr(output_2, "tokens", [])]
}

# -------------------------------
# 5️⃣ 概念引导示例
# -------------------------------
concept_known = {42: 1.0}       # 激活已知概念 42
concept_unknown = {314: -0.5}   # 抑制未知概念 314

config_steer = GenerationConfig(
    max_new_tokens=50,
    seed=2026,
    steer_known=concept_known,
    steer_unknown=concept_unknown
)
prompt_3 = "what is machine learning?"
output_steer = generator.generate_full(prompt_3, config_steer)

results['concept_steering'] = {
    "prompt": prompt_3,
    "text": getattr(output_steer, "text", None),
    "steer_known": concept_known,
    "steer_unknown": concept_unknown,
    "tokens": [extract_token_info(t) for t in getattr(output_steer, "tokens", [])]
}

# -------------------------------
# 6️⃣ 保存结果到 JSON
# -------------------------------
json_path = "steerling_full_results_complete.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"All generation results saved to {json_path}")