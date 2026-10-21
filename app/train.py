from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, str]]:
    if path.stat().st_size > 1_000_000_000:
        raise ValueError("dataset exceeds 1 GB safety limit")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON: {exc.msg}") from None
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("instruction", "input", "output")):
            raise ValueError(f"line {line_no}: expected string instruction, input, and output")
        if not row["instruction"].strip() or not row["output"].strip():
            raise ValueError(f"line {line_no}: instruction and output cannot be empty")
        rows.append(row)
    if not rows:
        raise ValueError("dataset has no records")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA instruction fine-tuning")
    parser.add_argument("--config", default="configs/qlora.yaml")
    args = parser.parse_args()
    try:
        import yaml
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    except ImportError as exc:
        raise SystemExit(f"Install requirements and use a supported CUDA/Linux environment: {exc}") from None

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_name = config["model_name_or_path"]
    if model_name.startswith("REPLACE_"):
        raise SystemExit("Set model_name_or_path in the config to a model you are licensed to use")
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA configuration requires a CUDA GPU; CPU training is not supported by this example")
    rows = load_jsonl(Path(config["dataset_path"]))
    from sklearn.model_selection import train_test_split
    validation_fraction = float(config.get("validation_fraction", 0.1))
    if len(rows) < 2 or not 0 < validation_fraction < 0.5:
        raise SystemExit("Provide at least two examples and validation_fraction in (0, 0.5)")
    train_rows, eval_rows = train_test_split(rows, test_size=validation_fraction, random_state=int(config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
                                      bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
                      lora_dropout=float(config["lora_dropout"]), bias="none", task_type="CAUSAL_LM",
                      target_modules="all-linear")
    model = get_peft_model(model, lora)

    def tokenize(row: dict) -> dict:
        prompt = f"### Instruction:\n{row['instruction']}\n### Input:\n{row['input']}\n### Response:\n"
        full = prompt + row["output"] + tokenizer.eos_token
        encoded = tokenizer(full, truncation=True, max_length=int(config["max_seq_length"]))
        prefix = tokenizer(prompt, truncation=True, max_length=int(config["max_seq_length"]))["input_ids"]
        labels = list(encoded["input_ids"])
        labels[:min(len(prefix), len(labels))] = [-100] * min(len(prefix), len(labels))
        encoded["labels"] = labels
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(rows[0].keys()))
    eval_dataset = Dataset.from_list(eval_rows).map(tokenize, remove_columns=list(rows[0].keys()))
    training_args = TrainingArguments(
        output_dir=config["output_dir"], learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]), per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]), logging_steps=10,
        save_strategy="epoch", evaluation_strategy="epoch", load_best_model_at_end=True,
        bf16=compute_dtype == torch.bfloat16, fp16=compute_dtype == torch.float16,
        report_to="none", seed=int(config["seed"]),
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      eval_dataset=eval_dataset, data_collator=collator)
    trainer.train()
    output = Path(config["output_dir"])
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    metrics = trainer.evaluate()
    metadata = {"base_model": model_name, "dataset": config["dataset_path"], "records": len(rows),
                "train_records": len(train_rows), "validation_records": len(eval_rows),
                "validation_metrics": metrics,
                "seed": config["seed"], "training_args": {k: config[k] for k in
                ("learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "max_seq_length")}}
    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Adapter saved to {output}; evaluate it before any export or deployment.")


if __name__ == "__main__":
    main()
