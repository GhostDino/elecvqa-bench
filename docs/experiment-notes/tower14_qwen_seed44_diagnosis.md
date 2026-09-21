# Qwen seed 44 / tower14 性能下降诊断

## 结论

Qwen seed 44 在 tower14 中的下降不是训练失败，也不是当时 GPU / NVML 异常造成的评估假象，而是一次**真实的稀有类别 seed 敏感性**，主要由 multiclass 任务中的 `torned-up` 类触发。

## 关键证据

机器可读复测摘要：`tower14_qwen_seed44_rerun_analysis.json`。

### 1. 训练本身没有异常

| Seed | 最终 train loss | token accuracy |
|---|---:|---:|
| 42 | 0.01616 | ~0.998 |
| 43 | 0.01647 | ~0.998 |
| 44 | 0.01711 | ~0.998 |

三个 seed 的训练指标相近，说明 seed 44 不是整体训练失败。

### 2. 健康环境复测复现了同样的错误

原始 seed 44 推理曾出现异常耗时：

- seed 42 / 43：约 1,734–1,746 秒
- seed 44 原始推理：约 151,793 秒，并伴随 `Can't initialize NVML` 和 CPU fallback 警告

因此对 seed 44 做了一次健康 GPU 复测：

- 输出：`runs/tower14_qwen3vl_8b_seed44/eval_T3_rerun.jsonl`
- 样本数：7,057
- 推理耗时：1,795.54 秒
- 速度：3.93 samples/s
- 复测仍把两个关键 `torned-up` 样本判为 `good`

复测与原始结果在 5,536 条 multiclass 样本中有 14 条预测不同，但整体指标几乎不变：

| 指标 | 原始 seed 44 | 健康复测 |
|---|---:|---:|
| full-seven macro recall | 0.78882 | 0.78953 |
| common-six macro recall | 0.83696 | 0.83778 |
| `torned-up` recall | 0.33333 | 0.33333 |

因此，原始结果中的性能下降不是因为 GPU 异常导致的整体评估失真。

### 3. 下降几乎完全来自 `torned-up`

`torned-up` 在 tower14 T3-only 测试集中只有 3 个样本：

| Seed | `torned-up` recall |
|---|---:|
| 42 | 1.0000 |
| 43 | 1.0000 |
| 44 | 0.3333 |

seed 44 只答对 1/3，导致：

- `torned-up` 单类 recall 下降 66.7 个百分点
- 对 common-six macro recall 的影响约 11.11 个百分点

去掉 `torned-up` 后，其余五类 macro recall 为：

| Seed | 5-class macro recall |
|---|---:|
| 42 | 0.9243 |
| 43 | 0.9420 |
| 44 | 0.9377 |

因此 seed 44 并不是整体能力退化，而是被一个只有 3 个测试样本的稀有类放大。

### 4. 失败集中在 multiclass，而不是 binary 缺陷检测

两个误判样本为：

1. `01-06-2021_DJI_0457_142.jpg`
2. `01-06-2021_DJI_0456_141.jpg`

在 binary 任务中，seed 42 / 43 / 44 都能把这两张图判为“存在缺陷”。

但在 multiclass 任务中：

- seed 42 / 43 判为 `torned-up`
- seed 44 判为 `good`

这说明 seed 44 并非完全看不见损伤，而是在“状态正常 / 撕裂”这个更细粒度的类别选择上发生了 seed 敏感错误。

## 对论文解释的影响

1. tower14 的 seed 44 结果应解释为**稀有类放大下的 seed 敏感性**，不应解释为模型整体退化。
2. 该结果进一步支持把 tower14 定位为 exploratory replication，而不是 confirmatory replication。
3. 在讨论 `torned-up` 时应明确其测试支持只有 3 个样本，macro recall 对单个样本错误极其敏感。
4. 如果后续继续做 tower 复现实验，应预先扩大 `torned-up` 等稀有类的测试支持，或报告 class-support 加权的敏感性分析。
