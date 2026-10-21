from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.train import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adapter perplexity and normalized exact-match rate")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--dataset", required=True, help="Held-out JSONL, separate from training data")
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Use a CUDA GPU for representative model evaluation")
    rows = load_jsonl(Path(args.dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    losses: list[float] = []
    exact = 0
    example_stats = []
    for row in rows:
        prompt = f"### Instruction:\n{row['instruction']}\n### Input:\n{row['input']}\n### Response:\n"
        full = prompt + row["output"] + tokenizer.eos_token
        encoded = tokenizer(full, return_tensors="pt", truncation=True, max_length=args.max_seq_length)
        prefix_len = len(tokenizer(prompt, truncation=True, max_length=args.max_seq_length)["input_ids"])
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        labels = encoded["input_ids"].clone()
        labels[:, :min(prefix_len, labels.shape[1])] = -100
        prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
        with torch.inference_mode():
            out = model(**encoded, labels=labels)
            generated = model.generate(**prompt_tokens, max_new_tokens=min(256, args.max_seq_length), do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(generated[0][prompt_tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        expected = row["output"].strip()
        losses.append(float(out.loss.detach().cpu()))
        match = prediction.casefold() == expected.casefold()
        exact += int(match)
        example_stats.append({"exact_normalized_match": match, "prediction_chars": len(prediction), "reference_chars": len(expected)})
    mean_loss = sum(losses) / len(losses)
    result = {"base_model": args.base_model, "adapter": args.adapter or None, "dataset": args.dataset,
              "examples": len(rows), "mean_token_loss": mean_loss,
              "perplexity": math.exp(min(mean_loss, 80)), "exact_normalized_match_rate": exact / len(rows),
              "example_summaries": example_stats,
              "note": "Perplexity is response-token loss; exact match is strict and should not be the sole task metric.",
              "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(0)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
