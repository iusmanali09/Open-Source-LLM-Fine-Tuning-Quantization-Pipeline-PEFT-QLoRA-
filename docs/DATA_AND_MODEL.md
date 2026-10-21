# Dataset and Model Governance

## Dataset format

Each JSONL row contains string fields `instruction`, `input`, and `output`. `input` may be empty. The example dataset is synthetic and only demonstrates the format.

Before training, record source, license/permission, collection date, intended purpose, personal-data review, deduplication, filtering, and train/validation/test split methodology. Keep private data outside the repository and define deletion/retention requirements.

## Model selection

Record the exact model identifier and immutable revision, tokenizer revision, model license, access terms, and any attribution requirements. A model being downloadable does not establish that a use or redistribution is permitted.

## Evaluation and export

Use held-out examples that do not overlap with training. Choose task metrics appropriate to the intended use and inspect representative failures. Quantization changes quality and must be evaluated again on the target runtime and device. Publish artifact hashes, toolchain versions, benchmark conditions, and limitations with any release.
