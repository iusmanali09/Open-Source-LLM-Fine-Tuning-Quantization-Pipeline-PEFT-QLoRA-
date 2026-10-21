from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PEFT adapters and export through selected toolchains")
    sub = parser.add_subparsers(dest="format", required=True)
    gguf = sub.add_parser("gguf")
    gguf.add_argument("--base-model", required=True)
    gguf.add_argument("--adapter", required=True)
    gguf.add_argument("--output-dir", default="outputs/merged")
    gguf.add_argument("--llama-cpp-dir", required=True)
    gguf.add_argument("--quantization", default="Q4_K_M")
    awq = sub.add_parser("awq")
    awq.add_argument("--merged-model", required=True)
    awq.add_argument("--output-dir", default="outputs/awq")
    args = parser.parse_args()
    if args.format == "gguf":
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="cpu")
        merged = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(output_dir, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(output_dir)
        llama = Path(args.llama_cpp_dir).resolve()
        converter, quantizer = llama / "convert_hf_to_gguf.py", llama / "build" / "bin" / "llama-quantize"
        if not converter.is_file() or not quantizer.is_file():
            raise SystemExit("The selected llama.cpp checkout must contain convert_hf_to_gguf.py and build/bin/llama-quantize")
        f16 = output_dir / "model-f16.gguf"
        quantized = output_dir / f"model-{args.quantization}.gguf"
        run(["python", str(converter), str(output_dir), "--outfile", str(f16), "--outtype", "f16"])
        run([str(quantizer), str(f16), str(quantized), args.quantization])
        print(f"Exported {quantized}")
    else:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        model = AutoAWQForCausalLM.from_pretrained(args.merged_model)
        tokenizer = AutoTokenizer.from_pretrained(args.merged_model)
        model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_quantized(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Exported AWQ model to {args.output_dir}")


if __name__ == "__main__":
    main()
