# Development Protocol

- Version the base model revision, tokenizer, dataset revision, config, seed, and library versions with every run.
- Split data before training and prevent near-duplicate or source-level leakage into evaluation data.
- Validate JSONL schemas and reject empty instructions/outputs.
- Review licenses and training-data permissions before ingesting examples.
- Add tests for data validation, formatting, truncation, masking, checkpoint resume, and metadata output.
- Report held-out task metrics, perplexity methodology, hardware, and runtime; do not infer performance from training loss alone.
- Require evaluation and human review before model conversion, release, or deployment.

## Pull request checklist

- [ ] Model and dataset licenses/use are documented.
- [ ] Dataset splits and preprocessing are reproducible.
- [ ] No private dataset rows or weights are committed.
- [ ] Training parameters and model revision are recorded.
- [ ] Evaluation is separated from training and reports failure cases.
- [ ] Export format and runtime compatibility are validated on the target device.
