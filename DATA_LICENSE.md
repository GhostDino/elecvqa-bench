# Data License and Source Notice

## Source dataset

ElecVQA-Bench is derived from the public InsPLAD dataset:

Vieira e Silva, A.L.B.; de Castro Felix, H.; Simões, F.P.M.; Teichrieb, V.; dos Santos, M.M.; Santiago, H.; Sgotti, V.; Lott Neto, H.B.D.T. *InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images.* *International Journal of Remote Sensing* 44 (2023): 7294–7320. https://doi.org/10.1080/01431161.2023.2283900

## No source-image redistribution

This repository does **not** redistribute InsPLAD source photographs or the original InsPLAD archive. Users must obtain InsPLAD from its public release and comply with its terms.

## Released derived metadata

The JSON/JSONL/CSV metadata, split manifests, prompts, predictions, and audit records in this repository are released for reproducibility of the ElecVQA-Bench study. They reference, but do not embed, source imagery.

If InsPLAD's release terms impose additional restrictions on derived metadata, those terms take precedence for the derived-data files. The MIT license in this repository applies to the software and documentation authored for this release.

## Expected source layout

Place the downloaded InsPLAD subsets under:

```text
data/raw/
  InsPLAD-det/
  supervised_fault_classification/
  unsupervised_anomaly_detection/
```

The released construction scripts expect this layout.
