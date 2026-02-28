"""


python sentence_concept_steering_demo.py \
  --model /root/models/steerling-8b/guidelabs/steerling-8b \
  --prompt "请解释机器学习，并给出生活中的例子。" \
  --max-new-tokens 120 \
  --seed 42 \
  --output-json sentence_concept_steering_results.json



Sentence-level concept attribution + steering demo for Steerling-8B.

What this script does:
1. Generate baseline text from a prompt.
2. Run one extra forward pass with minimal_output=False for concept attribution.
3. Aggregate token-level concept attribution into sentence-level summaries.
4. Pick representative concepts and run promote/suppress steering experiments.
5. Save everything to JSON for side-by-side analysis.

Notes:
- Concept IDs are numeric. This repo does not provide concept-name mapping.
- Training data attribution is not supported in this release.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any

import torch
from steerling import GenerationConfig, SteerlingGenerator


@dataclass
class SentenceSpan:
    sentence_index: int
    text: str
    token_start: int
    token_end_exclusive: int


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for both English and Chinese punctuation."""
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


def align_sentences_to_tokens(token_ids: list[int], tokenizer) -> list[SentenceSpan]:
    """
    Build sentence token spans by incremental decode.

    This approach avoids relying on char-level token offsets not exposed by tokenizer API.
    """
    full_text = tokenizer.decode(token_ids)
    sentences = split_sentences(full_text)
    if not sentences:
        return []

    spans: list[SentenceSpan] = []
    cursor = 0
    for idx, sentence in enumerate(sentences):
        sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
        if not sentence_ids:
            continue
        start = cursor
        end = min(len(token_ids), start + len(sentence_ids))
        spans.append(
            SentenceSpan(
                sentence_index=idx,
                text=sentence,
                token_start=start,
                token_end_exclusive=end,
            )
        )
        cursor = end
    return spans


def to_float_list(tensor: torch.Tensor, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in tensor.detach().cpu().tolist()]


def token_level_concept_attribution(
    generator: SteerlingGenerator,
    full_text: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Return token-level known/unknown concept attribution arrays.
    """
    token_ids = generator.tokenizer.encode(full_text, add_special_tokens=False)
    x = torch.tensor([token_ids], dtype=torch.long, device=generator.device)

    with torch.inference_mode():
        _, outputs = generator.model(
            x,
            use_teacher_forcing=False,
            minimal_output=False,
        )

    known_idx = outputs.known_topk_indices
    known_logits = outputs.known_topk_logits
    unknown_idx = outputs.unknown_topk_indices
    unknown_logits = outputs.unknown_topk_logits

    # Fallback for checkpoints/paths that return dense known_logits but no known_topk_*.
    if known_idx is None or known_logits is None:
        if outputs.known_logits is not None:
            dense_known_logits = outputs.known_logits
            k_dense = min(top_k, dense_known_logits.shape[-1])
            topk_vals, topk_idx = torch.topk(dense_known_logits, k=k_dense, dim=-1)
            known_idx = topk_idx
            known_logits = topk_vals
            print(
                "[Info] known_topk_* is None; fallback to top-k from outputs.known_logits "
                f"(k={k_dense})."
            )
        else:
            raise RuntimeError(
                "known_topk_indices/known_topk_logits is None and outputs.known_logits is also None; "
                "cannot build concept attribution."
            )

    k = min(top_k, known_idx.shape[-1])
    per_token: list[dict[str, Any]] = []

    for pos in range(known_idx.shape[1]):
        token_id = int(token_ids[pos])
        token_text = generator.tokenizer.decode([token_id])
        row: dict[str, Any] = {
            "position": pos,
            "token_id": token_id,
            "token_text": token_text,
            "known_concept_ids": known_idx[0, pos, :k].detach().cpu().tolist(),
            "known_concept_logits": to_float_list(known_logits[0, pos, :k]),
        }
        if unknown_idx is not None and unknown_logits is not None:
            uk = min(k, unknown_idx.shape[-1])
            row["unknown_concept_ids"] = unknown_idx[0, pos, :uk].detach().cpu().tolist()
            row["unknown_concept_logits"] = to_float_list(unknown_logits[0, pos, :uk])
        per_token.append(row)

    return {
        "token_ids": token_ids,
        "token_level_attribution": per_token,
    }


def aggregate_sentence_concepts(
    tokenizer,
    token_ids: list[int],
    token_level_attribution: list[dict[str, Any]],
    sentence_top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Aggregate token-level concept logits to sentence-level ranking.
    Score = mean(logit) across sentence tokens where concept appears in top-k.
    """
    sentence_spans = align_sentences_to_tokens(token_ids, tokenizer)
    results: list[dict[str, Any]] = []

    for span in sentence_spans:
        concept_scores: dict[int, list[float]] = {}
        token_slice = token_level_attribution[span.token_start : span.token_end_exclusive]

        for item in token_slice:
            ids = item["known_concept_ids"]
            logits = item["known_concept_logits"]
            for cid, logit in zip(ids, logits, strict=True):
                concept_scores.setdefault(int(cid), []).append(float(logit))

        ranked = sorted(
            (
                {
                    "concept_id": cid,
                    "mean_logit": round(sum(vals) / len(vals), 4),
                    "support_tokens": len(vals),
                }
                for cid, vals in concept_scores.items()
            ),
            key=lambda x: x["mean_logit"],
            reverse=True,
        )

        results.append(
            {
                "sentence_index": span.sentence_index,
                "sentence_text": span.text,
                "token_range": [span.token_start, span.token_end_exclusive],
                "top_concepts": ranked[:sentence_top_k],
            }
        )
    return results


def choose_demo_concepts(sentence_summary: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """
    Pick two concept IDs for steering demo:
    - promote_concept: highest mean logit in sentence summaries
    - suppress_concept: second-best candidate (or same if not enough)
    """
    pool: list[tuple[int, float]] = []
    for sent in sentence_summary:
        for c in sent["top_concepts"]:
            pool.append((int(c["concept_id"]), float(c["mean_logit"])))

    if not pool:
        return None, None

    pool_sorted = sorted(pool, key=lambda x: x[1], reverse=True)
    promote = pool_sorted[0][0]
    suppress = pool_sorted[1][0] if len(pool_sorted) > 1 else pool_sorted[0][0]
    return promote, suppress


def run_generation(generator: SteerlingGenerator, prompt: str, config: GenerationConfig) -> dict[str, Any]:
    output = generator.generate_full(prompt, config)
    return {
        "prompt": prompt,
        "generated_text": output.text,
        "prompt_tokens": output.prompt_tokens,
        "generated_tokens": output.generated_tokens,
    }


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentence-level concept attribution + steering demo")
    parser.add_argument(
        "--model",
        default="/root/models/steerling-8b/guidelabs/steerling-8b",
        help="Local model path or HF repo id",
    )
    parser.add_argument("--prompt", default="What is machine learning? Explain with examples.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-token", type=int, default=5, help="Top-k concepts per token")
    parser.add_argument("--top-k-sentence", type=int, default=8, help="Top-k concepts per sentence")
    parser.add_argument(
        "--steer-weight",
        type=float,
        default=1.2,
        help="Absolute weight used in steering: +w for promote, -w for suppress",
    )
    parser.add_argument(
        "--promote-concept-id",
        type=int,
        default=None,
        help="Optional manual known concept id to promote. If omitted, auto-picked from attribution.",
    )
    parser.add_argument(
        "--suppress-concept-id",
        type=int,
        default=None,
        help="Optional manual known concept id to suppress. If omitted, auto-picked from attribution.",
    )
    parser.add_argument(
        "--output-json",
        default="sentence_concept_steering_results.json",
        help="Output JSON file path",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {args.model}")
    generator = SteerlingGenerator.from_pretrained(args.model, device=device)
    print(f"Loaded on device: {device}, interpretable={generator.is_interpretable}")

    if not generator.is_interpretable:
        raise RuntimeError("Model is not interpretable. Please load steerling-8b interpretable checkpoint.")

    # 1) Baseline generation
    base_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        top_p=0.9,
        repetition_penalty=1.2,
        use_entropy_sampling=True,
    )
    baseline = run_generation(generator, args.prompt, base_cfg)
    full_text = args.prompt + " " + baseline["generated_text"]

    # 2) Token-level concept attribution
    attr = token_level_concept_attribution(generator, full_text, top_k=args.top_k_token)
    sentence_summary = aggregate_sentence_concepts(
        tokenizer=generator.tokenizer,
        token_ids=attr["token_ids"],
        token_level_attribution=attr["token_level_attribution"],
        sentence_top_k=args.top_k_sentence,
    )

    # 3) Choose two concepts and run steering comparison
    auto_promote, auto_suppress = choose_demo_concepts(sentence_summary)
    promote_concept = args.promote_concept_id if args.promote_concept_id is not None else auto_promote
    suppress_concept = args.suppress_concept_id if args.suppress_concept_id is not None else auto_suppress

    steering_runs: dict[str, Any] = {}
    if promote_concept is not None:
        promote_cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            top_p=0.9,
            repetition_penalty=1.2,
            use_entropy_sampling=True,
            steer_known={promote_concept: abs(args.steer_weight)},
        )
        steering_runs["promote_known"] = {
            "steer_known": {promote_concept: abs(args.steer_weight)},
            **run_generation(generator, args.prompt, promote_cfg),
        }

    if suppress_concept is not None:
        suppress_cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            top_p=0.9,
            repetition_penalty=1.2,
            use_entropy_sampling=True,
            steer_known={suppress_concept: -abs(args.steer_weight)},
        )
        steering_runs["suppress_known"] = {
            "steer_known": {suppress_concept: -abs(args.steer_weight)},
            **run_generation(generator, args.prompt, suppress_cfg),
        }

    # 4) Save report
    result = {
        "metadata": {
            "model": args.model,
            "device": device,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "top_k_token": args.top_k_token,
            "top_k_sentence": args.top_k_sentence,
            "steer_weight_abs": abs(args.steer_weight),
        },
        "baseline": baseline,
        "sentence_concept_attribution": sentence_summary,
        "steering_demo": steering_runs,
        "notes": {
            "concept_name_mapping": "Not provided in this release. Concept IDs are numeric only.",
            "training_data_attribution": "Not supported in lightweight inference release.",
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"- Baseline generated tokens: {baseline['generated_tokens']}")
    print(f"- Sentence count: {len(sentence_summary)}")
    print(f"- Output JSON: {args.output_json}")
    if promote_concept is not None:
        print(f"- Promote concept id: {promote_concept}")
    if suppress_concept is not None:
        print(f"- Suppress concept id: {suppress_concept}")


if __name__ == "__main__":
    main()
