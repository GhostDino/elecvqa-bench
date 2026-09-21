# ElecVQA-Bench Data Card

## Dataset summary

ElecVQA-Bench is a benchmark-audit dataset derived from public InsPLAD UAV power-line inspection imagery. It contains three task families:

| Task | Description | Released items |
|---|---|---:|
| T1 | Grounding | 10,605 |
| T2 | Binary condition assessment | 9,997 |
| T3 | Fine-grained defect typing | 36,370 |

The task files are in:

```text
data/benchmark/tasks/
  T1_grounding.jsonl
  T2_binary.jsonl
  T3_multiclass.jsonl
```

Each record contains a split label, image path, question or message, answer, and task metadata. Source images are not included.

## Splits

The parent-ID-grouped base split is in:

```text
data/benchmark/splits/
  train.jsonl
  val.jsonl
  test.jsonl
  MANIFEST.json
  MANIFEST_L2.json
```

The manifest records SHA-256 hashes and record counts. Verification results are in:

```text
data/metadata/split_manifest_verification.json
```

Split-comparison variants used in the study are in:

```text
data/benchmark/split_comparison/
```

## Labels

T2 uses normal/defective/cannot-judge options. T3 uses the following source-derived labels:

- `good`
- `rust`
- `corrosão`
- `missing-cap`
- `nest`
- `torned-up`
- `peeling-paint`

The full mapping is in:

```text
configs/label_crosswalk.csv
```

Per-class and per-asset support is in:

```text
data/metadata/per_class_support.csv
```

## Prompts and side information

P1–P4 prompt templates are in:

```text
configs/prompts.json
```

Per-asset Chinese names, candidate option lists, labels, and answer mappings are in:

```text
configs/asset_metadata.json
```

P3 is the default prompt level. P4 requires additional in-context reference images and is not used for training in the released pipeline.

## Construction audit

The construction audit is in:

```text
data/metadata/construction_audit.csv
data/metadata/construction_audit_summary.json
```

It records candidate crop IDs from the supervised and unsupervised subsets and whether each crop was included in T3.

### Important discrepancy

The current study states 38,350 T3 candidates and 1,980 exclusions. The released files and build script produce 39,145 candidates and 2,775 exclusions. The dominant actual exclusion reason is that an asset has no observed defect class, so a per-asset defect-typing question cannot be constructed. This discrepancy is recorded in `construction_audit_summary.json` and should be reconciled before submission.

## Predictions and metrics

Per-item prediction files are in:

```text
results/predictions/
```

Metric, statistical, and audit outputs are in:

```text
results/metrics/
results/statistics/
results/audits/
```

## Reproduction

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Regenerate metadata:

```powershell
python scripts/build_release_metadata.py
```

Verify base split hashes:

```powershell
python scripts/build_splits.py --verify-only --out data/benchmark/splits
```

## Intended use

The dataset is intended for reproducible research on UAV power-line asset inspection, vision-language model evaluation, vision-only baselines, prompt-level side information, split sensitivity, and defect-type classification.

## Limitations

- Source imagery is not included and must be obtained separately.
- Class support is highly imbalanced; rare classes can strongly affect macro-averaged recall.
- The current released construction audit differs from the study's candidate/exclusion counts.
- The benchmark is derived from InsPLAD and should not be treated as a general electrical-infrastructure inspection dataset.
