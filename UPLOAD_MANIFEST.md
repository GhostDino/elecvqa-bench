# ElecVQA-Bench Upload Manifest

## Purpose

This folder is a GitHub-ready release package for the ElecVQA-Bench study. It focuses on reviewer reproducibility and does not redistribute InsPLAD source imagery.

## Included

### Code and configuration

- `scripts/`: dataset construction, audit, training, evaluation, statistical analysis, and release-metadata scripts.
- `configs/`: model configuration, prompt templates, asset metadata, and label crosswalk.
- `requirements.txt`: pinned Python dependencies used by the released experiments.
- `tasks.yaml`: experiment task dependency definitions.
- `LICENSE`: MIT license for code and documentation.
- `DATA_LICENSE.md`: source-data notice and no-redistribution policy.

### Benchmark data

- `data/benchmark/tasks/T1_grounding.jsonl`: 10,605 grounding items.
- `data/benchmark/tasks/T2_binary.jsonl`: 9,997 binary-assessment items.
- `data/benchmark/tasks/T3_multiclass.jsonl`: 36,370 fine-grained-typing items.
- `data/benchmark/splits/`: parent-ID-grouped train/val/test records and SHA-256 manifests.
- `data/benchmark/split_comparison/`: crop, image, tower, tower14, and CNN comparison splits.
- `data/benchmark/prompt_ablation/`: T3 P1 prompt-ablation train/val/test files.
- `data/crops/expand_2.0_manifest.jsonl`: candidate crop manifest.
- `data/provenance/`: construction-stage state records.

### Metadata and audit

- `configs/prompts.json`: P1–P4 templates.
- `configs/asset_metadata.json`: per-asset names, options, labels, and answer mappings.
- `configs/label_crosswalk.csv`: benchmark label mapping.
- `data/metadata/per_class_support.csv`: task, split, asset, and label support.
- `data/metadata/construction_audit.csv`: candidate-to-final T3 construction audit.
- `data/metadata/construction_audit_summary.json`: audit counts and discrepancy note.
- `data/metadata/task_summary.json`: task and split summaries.
- `data/metadata/split_manifest_verification.json`: base split SHA-256 verification.
- `data/audit/`: source audit, parent-ID map, perceptual-hash collisions, and data card source audit.

### Predictions, metrics, and study notes

- `results/predictions/`: per-item predictions for main and multi-seed experiments.
- `results/metrics/`: main metrics and resolution/pixel-budget summaries.
- `results/statistics/`: confidence intervals, sign-flip tests, split comparisons, and cluster statistics.
- `results/audits/`: provenance and input-change audits.
- `docs/experiment-notes/`: key experiment notes and diagnostics.
- `checksums/SHA256SUMS.txt`: checksums for release files.

## Excluded

- InsPLAD source images and original archives.
- `.env`, credentials, API keys, and server-specific secrets.
- Model checkpoints, LoRA adapters, and base-model weights.
- Temporary caches, logs, and internal progress files.
- Historical analysis versions and review notes.

## Remaining issue

The released construction audit reports 39,145 T3 candidates and 2,775 exclusions, while the current study reports 38,350 candidates and 1,980 exclusions. The discrepancy is documented in `DATA_CARD.md` and `data/metadata/construction_audit_summary.json` and should be reconciled before submission.

## Regeneration

```powershell
python scripts/build_release_metadata.py
python scripts/build_splits.py --verify-only --out data/benchmark/splits
```
