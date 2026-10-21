# Open-Source LLM Fine-Tuning & Quantization Pipeline

A reproducible QLoRA instruction-tuning starter for CUDA/Linux environments. It validates JSONL records, loads a licensed base model in 4-bit, trains a LoRA adapter, and saves the adapter plus run metadata.

## Workflow

```mermaid
flowchart LR
  D[Reviewed instruction JSONL] --> V[Schema and data validation]
  V --> P[Format and tokenize examples]
  P --> M[Licensed base model loaded in 4-bit]
  M --> T[QLoRA / PEFT adapter training]
  T --> A[Adapter and run metadata]
  A --> E[Held-out evaluation and human review]
  E -->|approved| X[Optional merge / conversion]
  X --> Q[Quantization toolchain selected by target]
  Q --> GG[GGUF or other runtime artifact]
  E -->|failed| RET[Revise data/config; retrain]
```

## Current implementation

The pipeline includes JSONL validation, deterministic train/validation splitting, 4-bit model loading, response-only adapter training, validation loss/perplexity, held-out perplexity and exact-match evaluation, and GGUF/AWQ export entry points. Export requires separately installed compatible toolchains and validation on the intended runtime. See the deployment guide for exact commands.

## Requirements

Use a supported Linux/CUDA/PyTorch/Transformers/bitsandbytes stack and a model you are licensed to use. Dependency installation may need adjustment for the installed CUDA and platform versions. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

Do not commit datasets, model weights, adapters, access tokens, or private training examples. Review [`docs/DATA_AND_MODEL.md`](docs/DATA_AND_MODEL.md) before using real data.
