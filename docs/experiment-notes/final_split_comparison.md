# Final Split Comparison — Round 2 Final

Primary metric: common-six-class macro recall. All crop/image rows use models retrained on their respective splits; tower uses the tower-disjoint retrained models.

| Split | Qwen | ResNet-50 448px | Difference | 95% CI | Raw p | Holm p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| crop | 0.9413 | 0.9464 | -0.51 pt | [-5.58, 15.12] pt | 0.6901 | 0.8035 | Qwen not ahead; not significant |
| image | 0.9227 | 0.9667 | -4.40 pt | [-16.23, 1.09] pt | 0.4018 | 0.8035 | Qwen not ahead; not significant |
| tower | 0.7647 | 0.6845 | +8.02 pt | [4.04, 10.72] pt | 0.0239 | 0.0717 | Qwen ahead; not significant after Holm |

Crop-level still allows same-image and same-tower overlap. Image-level removes image/parent overlap but allows tower overlap. Tower-level removes tower overlap.

No split regime shows a Holm-significant Qwen–ResNet advantage after retraining.

## V9 Tower Multi-Seed Sensitivity

Exploratory follow-up on the same 13,574-item tower-disjoint test set. Bootstrap and permutation are tower-clustered; Holm correction is applied within this three-comparison family.

| Comparison | Qwen | ResNet-50 448px | Difference | 95% CI | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| seed 42 vs. seed 42 | 0.7647 | 0.6845 | +8.02 pt | [+4.04, +10.72] pt | 0.0239 | 0.0478 |
| seed 43 vs. seed 43 | 0.7560 | 0.7231 | +3.29 pt | [-4.13, +6.24] pt | 0.4474 | 0.4474 |
| seed 43 vs. seed 44 (sensitivity) | 0.7560 | 0.7057 | +5.03 pt | [+1.14, +7.66] pt | 0.0090 | 0.0270 |

Interpretation: Qwen is ahead in every tested tower run, but the effect size and inferential status are seed-sensitive. This is exploratory sensitivity evidence, not confirmation of the primary tower advantage.

## V9.1 Tower14 Replication

Exploratory 14-test-tower, three-seed replication on 5,536 T3-only items. Bootstrap and permutation are tower-clustered; Holm correction is applied within the three seed contrasts.

| Seed | Qwen | ResNet-50 448px | Difference | 95% CI | Raw p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.9369 | 0.9561 | -1.91 pt | [-2.85, -1.29] pt | 0.0904 | 0.2712 |
| 43 | 0.9517 | 0.9455 | +0.62 pt | [-0.68, +2.33] pt | 0.9157 | 1.0000 |
| 44 | 0.8370 | 0.9512 | -11.42 pt | [-12.60, -10.50] pt | 0.6546 | 1.0000 |

Model means over the three seeds:

- Qwen: 0.9085 (sample SD 0.0624)
- ResNet-50: 0.9509 (sample SD 0.0053)
- Mean difference: -4.24 pt

Interpretation: the larger tower replication does not support a stable Qwen advantage. ResNet-50 leads in two of three seeds and is substantially more stable across seeds.
