# ElecVQA-Bench

This repository contains the reproducibility package for **ElecVQA-Bench**, including construction scripts, split manifests, task metadata, prompts, per-item predictions, statistical results, and the study notes.

## Repository map

```text
configs/                 Model, prompt, asset, and label metadata
scripts/                 Dataset construction, training, evaluation, and analysis scripts
data/benchmark/tasks/    T1, T2, and T3 JSONL task files
data/benchmark/splits/   Parent-ID-grouped base split and manifests
data/benchmark/split_comparison/  Crop/image/tower/tower14 comparison splits
data/audit/              Source-dataset audit outputs
data/crops/              Candidate crop manifest
data/metadata/           Support tables, construction audit, and verification reports
data/provenance/         Construction-stage provenance records
results/                 Predictions, metrics, statistics, and audits
docs/                    Experiment notes and diagnostics
```

## Quick start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Regenerate support tables and the construction audit:

```powershell
python scripts/build_release_metadata.py
```

Verify the base split manifest:

```powershell
python scripts/build_splits.py --verify-only --out data/benchmark/splits
```

## Source imagery

InsPLAD source images are not redistributed. Download InsPLAD separately and place it under:

```text
data/raw/
  InsPLAD-det/
  supervised_fault_classification/
  unsupervised_anomaly_detection/
```

See `DATA_LICENSE.md` and `DATA_CARD.md` for source terms and usage limitations.

## Important note

`data/metadata/construction_audit_summary.json` records a discrepancy between the current study's stated T3 candidate/exclusion counts and the counts produced by the released files. Please reconcile that discrepancy before submission.
