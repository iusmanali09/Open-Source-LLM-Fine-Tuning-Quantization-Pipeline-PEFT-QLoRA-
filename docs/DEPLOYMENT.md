# Training Setup

## Prepare environment

Use a Linux host with an NVIDIA GPU, a compatible driver/CUDA stack, and enough VRAM for the selected model and sequence length. This repository does not pin a single CUDA wheel because PyTorch and bitsandbytes installation depends on the host. Install the appropriate PyTorch build first, then:

```sh
cd "Open-Source-LLM-Fine-Tuning-Quantization-Pipeline-PEFT-QLoRA-"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp data/train.example.jsonl data/train.jsonl
```

Set a licensed model ID or local model path in `configs/qlora.yaml`, review all hyperparameters, then run:

```sh
python -m app.train --config configs/qlora.yaml
```

The configured JSONL must contain enough distinct examples for a train/validation split; use `validation_fraction` to adjust it. Training reports validation metrics and stores run metadata. For held-out evaluation, provide a separate dataset file and run:

```sh
python -m app.evaluate --base-model 'MODEL_ID' --adapter outputs/adapter \
  --dataset data/heldout.jsonl --output outputs/heldout-evaluation.json
```

## Export

Install and build a compatible llama.cpp checkout separately, then merge and quantize to GGUF:

```sh
python -m app.export gguf --base-model 'MODEL_ID' --adapter outputs/adapter \
  --llama-cpp-dir /path/to/llama.cpp --quantization Q4_K_M
```

The optional AWQ path requires the compatible `autoawq` package and a merged model:

```sh
python -m app.export awq --merged-model /path/to/merged-model --output-dir outputs/awq
```

The default config intentionally contains a placeholder model path. The script refuses to proceed until it is replaced and exits when CUDA is unavailable. Model downloads may require provider authentication; use the provider's supported CLI and keep tokens outside the repository.

## Output

The adapter, tokenizer files, and `run-metadata.json` are saved to `outputs/adapter` by default. The `.gitignore` excludes outputs, weights, and checkpoints. Back up artifacts in an approved model registry, not Git.

## Export and evaluation

No export command is included because GGUF/AWQ conversion depends on the exact base model, adapter merge compatibility, quantizer, and runtime. Follow the selected tool's current documentation, record versions, validate output loading, and benchmark on target hardware. Measure perplexity or task accuracy on a held-out dataset before release. The training loss alone is not a release metric.
