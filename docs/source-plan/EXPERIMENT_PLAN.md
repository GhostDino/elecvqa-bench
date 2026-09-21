# ElecVQA-Bench 实验方案（智能体自动化执行版）

> 版本：v1.0　生成日期：2026-08-18
> 目标硬件：单卡 NVIDIA RTX PRO 6000 Blackwell 96GB（sm_120）
> 目标产出：1 篇开源 SCI/EI 期刊论文（首选 *Drones* / *Remote Sensing* / *IEEE Access*）
> 执行方式：本地 AI 智能体经 SSH 连接远程服务器，按 `tasks.yaml` 中的 DAG 顺序执行

---

## 0. 阅读顺序与本文档的使用方式

本文档面向两类读者：

1. **人类（你）**——第 1、2、9、10、12 节是决策相关的，需要你确认后智能体才能继续。
2. **智能体**——第 3–8 节是可执行规范，配套 `tasks.yaml`（DAG）、`scripts/`（脚本）、`configs/`（配置）。

**智能体的三条铁律，写进 system prompt：**

> 1. 每个 task 完成后写 `state/<task_id>.done` 哨兵文件与 `state/<task_id>.json` 摘要；重启时跳过已完成 task。
> 2. 任何标记为 `gate: true` 的 task，产出后必须**停止并向人类汇报**，收到明确指令才继续。
> 3. 禁止修改 `data/raw/` 下任何文件；所有派生数据写入 `work/`，所有实验产物写入 `runs/<run_id>/`。

---

## 1. 前置结论：这份方案与前几轮讨论的三处修正

在开始之前，有三件事必须先纠正，否则会做出跑不完的实验。

### 1.1 235B NVFP4 在 96GB 上不可行——量化研究必须降级到 8B/27B

`Qwen3-VL-235B-A22B-Instruct-NVFP4` 权重约 130–140 GB，单卡 96GB 装不下，CPU offload 后延迟不可用。**量化对比实验（G 组）的对象改为 8B 与 27B 级模型**：

| 精度 | 8B 模型显存 | 27B 模型显存 | 单卡可行性 |
|---|---|---|---|
| BF16 | ~16 GB | ~56 GB | ✅ |
| FP8 (W8A8) | ~9 GB | ~28 GB | ✅ |
| NVFP4 (W4A4) | ~5.5 GB | ~15 GB | ✅ 但见 §1.2 |
| AWQ/GPTQ INT4 (W4A16) | ~6 GB | ~16 GB | ✅ |

这不影响论文成立——**"量化对工业缺陷研判可靠性的影响"这个问题在 8B 尺度上同样没人做过**，而且 8B 才是真正会被部署到变电站边缘侧的尺度，论证反而更贴合场景。

### 1.2 sm_120 上的 NVFP4 支持是本方案最大的技术风险

RTX PRO 6000 Blackwell 是 sm_120，与 B200（sm_100）**不是同一套 kernel 路径**。vLLM / TensorRT-LLM 的 NVFP4 kernel 对 sm_100 的支持成熟度明显高于 sm_120，且各版本行为差异大。

**处置方式（写进 `tasks.yaml` 的 `T80_probe_fp4`）：**
- 在做任何 FP4 实验前，先跑一个 5 分钟的探针任务：加载一个小 NVFP4 模型跑 10 条推理，记录是否走原生 FP4 kernel（看 vLLM 日志里的 kernel 选择 / `nvidia-smi dmon` 的功耗特征）。
- **探针失败 → 自动降级**：G 组改为 `BF16 / FP8 / AWQ-INT4 / GPTQ-INT4` 四档，论文里把 NVFP4 那一格标注为"当前推理框架在 sm_120 上不支持原生 FP4 kernel"，**这本身也是一个可发表的工程发现**，不要视为失败。
- 探针脚本：`scripts/probe_fp4.py`。

### 1.3 InsPLAD 上不能做缺陷分级

InsPLAD 只有 `good / 缺陷名` 两级标签，**没有 DL/T 741 的危急/严重/一般**。任何在 InsPLAD 上声称做了规程分级的表述都是伪标签，审稿人一戳就破。

- **分级任务（T5）只在你的私有国网数据上做**，作为论文的第二部分。
- InsPLAD 承担：可复现基准、缺陷判别、长尾曲线、量化评估、跨域源域。
- 如果私有数据本轮不可用，**删掉分级章节，论文照样成立**（选题 A+B+D 组合）。

---

## 2. 论文骨架（决定了实验必须产出什么）

> **题目（草案）**
> *ElecVQA-Bench: A Reproducible Multimodal Benchmark and Detector-Guided Baseline for UAV Power Line Inspection under Extreme Class Imbalance*

**四条贡献 → 对应的实验组：**

| # | 贡献 | 对应实验组 | 关键产出 |
|---|---|---|---|
| C1 | 首个完全可复现的电力巡检多模态 VQA 基准 | Phase 2 | `ElecVQA-Bench` 数据集 + 构造脚本 + 评测 harness |
| C2 | 检测引导 ROI 研判流水线 + 误差传播定量分解 | Phase 3/5 | 端到端 vs 条件化准确率分解表 |
| C3 | 极端长尾下 VLM 与 CNN 的交叉点实证 | Phase 6 | 交叉点曲线图（论文的 Figure 1 候选） |
| C4 | 量化部署对研判可靠性的影响（含 ECE 与漏检率） | Phase 8 | 四精度 × 三维度（精度/延迟/校准）表 |

**每一条贡献都不依赖"打败 SOTA"**——这是这个选题命中率高的根本原因。C3 的结论无论交叉点落在哪里都是有效发现；C4 无论量化掉不掉分都是有效发现。

---

## 3. 服务器环境与目录规范

### 3.1 目录布局（智能体必须严格遵守）

```
/workspace/elecvqa/
├── data/
│   └── raw/                        # 只读，InsPLAD 原始三个子集
│       ├── InsPLAD-det/
│       ├── supervised_fault_classification/
│       └── unsupervised_anomaly_detection/
├── work/                           # 派生数据（可重建）
│   ├── audit/                      # Phase 0 审计产物
│   ├── splits/                     # Phase 1 划分文件
│   ├── crops/                      # ROI 裁剪缓存（按外扩比例分目录）
│   └── vqa/                        # Phase 2 生成的 QA jsonl
├── runs/
│   └── <run_id>/                   # run_id = <phase>_<exp>_<yyyymmdd_HHMMSS>
│       ├── config.resolved.yaml    # 冻结的完整配置
│       ├── train/                  # checkpoint、adapter
│       ├── preds/                  # 逐样本预测 jsonl
│       ├── metrics.json            # 结构化指标
│       └── log.jsonl               # 结构化日志
├── models/                         # HF 权重缓存（HF_HOME 指到这里）
├── configs/
├── scripts/
├── state/                          # 哨兵文件，幂等控制
├── logs/                           # 智能体自身的执行日志
└── results.sqlite                  # 所有 metrics.json 的聚合库
```

### 3.2 环境安装（`T00_setup`）

```bash
# 基础
conda create -n elecvqa python=3.11 -y && conda activate elecvqa

# PyTorch：sm_120 需要 CUDA 12.8+ 构建，不要用 cu121 轮子
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 推理与训练
pip install "vllm>=0.11" ms-swift[all] transformers accelerate peft bitsandbytes
pip install timm datasets pycocotools opencv-python-headless pillow
pip install scikit-learn scipy pandas pyyaml tqdm rich jsonlines
pip install lm-eval  # 可选

# 量化工具链
pip install nvidia-modelopt[torch]  # NVFP4
pip install autoawq gptqmodel       # INT4 对照
```

**验收门 Gate-0**（`scripts/check_env.py`）必须全部通过：

| 检查项 | 期望 |
|---|---|
| `torch.cuda.get_device_capability()` | `(12, 0)` |
| `torch.cuda.get_device_properties(0).total_memory` | ≥ 95 GB |
| `torch.cuda.is_bf16_supported()` | `True` |
| vLLM 能加载并推理 `Qwen2.5-VL-7B-Instruct` | 输出非空 |
| 磁盘剩余空间 | ≥ 1.5 TB |
| `nvidia-smi` driver | ≥ 580 |

### 3.3 智能体的运行时纪律

**长任务必须在 tmux 里跑**，不要挂在 SSH 会话上：

```bash
tmux new-session -d -s "elecvqa_${TASK_ID}" \
  "cd /workspace/elecvqa && python scripts/xxx.py --config configs/xxx.yaml 2>&1 | tee -a logs/${TASK_ID}.log; echo $? > state/${TASK_ID}.rc"
```

智能体轮询 `state/${TASK_ID}.rc` 判断结束，轮询 `logs/${TASK_ID}.log` 尾部判断进度。

**显存看门狗**：每个训练任务并行启动 `scripts/vram_watchdog.py`，每 30 秒记录一次 `nvidia-smi` 到 `runs/<run_id>/vram.csv`，峰值写入 `metrics.json`（论文的效率表要用）。

**OOM 自动回退阶梯**（智能体检测到 `CUDA out of memory` 时按序尝试，每次重试前清 `state/<task>.rc`）：

```
1. per_device_train_batch_size 减半
2. 开启 gradient_checkpointing（若未开）
3. max_pixels 降一档：1280² → 1024² → 768² → 640²
4. cutoff_len 降一档：8192 → 6144 → 4096
5. 切 QLoRA（bnb 4bit 基座）
6. 仍失败 → 停止并报告，不要无限重试
```

每次回退都要写进 `runs/<run_id>/fallback.log`，论文的可复现性附录需要这份记录。

---

## 4. Phase 0–1：数据审计与划分（最关键，也最容易被跳过）

> **不做这一步就开始训练，是这个项目最大的风险。** 前面报告里已经暴露了三处异常，必须先坐实。

### 4.1 `T10_audit`——数据审计（脚本：`scripts/audit_dataset.py`）

必须核实并输出的 12 项：

| # | 检查项 | 为什么重要 |
|---|---|---|
| 1 | 三个子集的实际文件数 vs 报告数字 | 报告可能基于目录解析，有偏差 |
| 2 | `yoke-suspension` 分类子集 train 589 / val 5,762 | **train 比 val 小一个数量级，明显反常**，需确认是官方划分还是解析错误 |
| 3 | `polymer-insulator-upper-shackle` 的 `corrosao`/`normal` ↔ `good`/`rust` | 葡英混用，必须建映射表 |
| 4 | det 子集 18 类 vs 官方 17 类，`sphere` val=0 | val 无样本 → 无法算该类 AP，评测时须排除并说明 |
| 5 | crop 文件名能否反解出父图 ID | **决定了能否做防泄漏划分**，是 Gate-1 的核心 |
| 6 | supervised 与 unsupervised 子集之间的 crop 重叠 | 两者都从 det 原图裁出，可能重复 |
| 7 | 每个 crop 的像素尺寸分布（min/median/max） | 决定 `max_pixels` 该设多少，别盲目上 1280² |
| 8 | 图像哈希去重（pHash，汉明距离 ≤ 4） | 无人机连拍会有近重复帧，跨集出现即泄漏 |
| 9 | 损坏/零字节图像 | 训练中途崩溃的常见原因 |
| 10 | COCO 标注中越界 bbox、面积为 0 的框 | 裁剪脚本会炸 |
| 11 | 各缺陷类别的样本数（含 n<10 的类别清单） | 决定哪些类只能报 bootstrap CI |
| 12 | EXIF 中的采集时间/机型（若存在） | 可用于时间分组划分，比随机划分更严格 |

**产出**：`work/audit/data_card.md`（论文附录直接用）、`work/audit/audit.json`（机器可读）、`work/audit/parent_id_map.csv`（crop → 父图 ID）。

**Gate-1（停下来汇报）**：

```
若 检查项 5 失败（crop 文件名无法反解父图 ID）：
    → 停止。改用 pHash 聚类做近似分组，需人类确认阈值。
若 检查项 2 确认 train/val 确实反了：
    → 停止。汇报三个选项：(a) 交换 (b) 沿用官方 (c) 自建划分。
       推荐 (c)，但必须在论文里说明"未沿用官方划分"及理由。
若 检查项 8 发现跨集近重复 > 1%：
    → 停止。必须重做划分。
```

### 4.2 `T11_split`——防泄漏划分（脚本：`scripts/build_splits.py`）

**核心原则：按父图 ID 分组（GroupShuffleSplit），同一张原图裁出的所有 crop 必须在同一个 split 里。**

```
划分比例：train 70% / val 10% / test 20%（按 group 计，不是按 crop 计）
随机种子：42（固定并公开）
分层依据：group 内的主导缺陷类别（近似分层，保证稀有类在三个 split 都有）
```

**特殊处理稀有类**（n < 30 的类别：`peeling-paint` n=4、`torned-up` n=22）：

- 不参与训练，只进 test，作为**开集/长尾观测项**单独报告
- 或全部进 test 并明确声明"n=4，仅报 bootstrap 95% CI，不报点估计"
- **绝对不要**为了凑数把 n=4 的类做 train/test 划分

**产出**：`work/splits/{train,val,test}.jsonl`，每行：

```json
{"crop_id":"...","parent_id":"1-1_DJI_0569","asset":"glass-insulator",
 "label":"missing-cap","source_subset":"supervised","path":"...","bbox":[...]}
```

**Gate-2 自动断言**（`scripts/verify_splits.py`，不通过直接 exit 1）：

- `parent_id` 三集合交集为空
- pHash 跨集碰撞数 == 0
- 每个参与训练的类别在 train/val/test 中样本数均 ≥ 5
- 划分文件的 SHA256 记入 `work/splits/MANIFEST.json`

---

## 5. Phase 2：VQA 数据构造

### 5.1 五类任务定义

| 任务 | 数据来源 | 规模估计 | 回答格式 | 主要指标 |
|---|---|---|---|---|
| **T1** 部件识别与定位 | det | 10,607 图 | JSON：类别 + 归一化 bbox | AP@50、grounding Acc |
| **T2** 缺陷判别（二分类） | supervised_fault | ~11,500 crop | `{"defect": true/false}` | Balanced Acc、幻觉率 |
| **T3** 缺陷类型细分 | supervised + AD | ~1,400 | 闭集单选（含"无缺陷"选项） | Macro-F1、混淆矩阵 |
| **T4** 研判依据生成 | 半自动合成 | 3–5k | 自由文本 + 结构化字段 | 人工评分 + 关键要素命中率 |
| **T5** 缺陷分级 | **仅私有数据** | 视数据量 | 危急/严重/一般 | 代价敏感风险、漏检率 |

### 5.2 三个必须做对的构造细节

**(1) ROI 外扩比例做成消融变量。**

`missing-cap`（玻璃绝缘子盖帽缺失）这类结构型缺陷，模型必须看到相邻的正常盖帽才能判断"少了一个"。裁剪按 bbox 外扩比例 `r ∈ {1.0, 1.5, 2.0, 3.0}` 各生成一套，Phase 7 做消融。

```python
# 外扩规则：中心不变，宽高各乘 r，然后 clip 到图像边界
# 若 clip 后短边 < 64px，则按短边 64px 反推（避免过小 crop）
```

**(2) 难负样本必须来自同一资产类型。**

T2 的负样本从**同一 asset 的 `good` crop** 中采样，比例 1:1。否则模型学到的是"这是玻璃绝缘子 → 有缺陷"的类别捷径。

进一步的强化：从**同一父图**里采同资产的 good crop 做负样本（`hard_negative_level = "same_parent"`），这一档单独做消融，是 DGPTE 那篇没做的点。

**(3) 提问必须中性，且强制提供"无异常"选项。**

反面教材（会诱导幻觉）：
> ❌「这个绝缘子有什么缺陷？」

正确写法：
> ✅「请判断图中玻璃绝缘子的状态。选项：A. 状态正常　B. 存在盖帽缺失　C. 图像质量不足以判断」

**C 选项（拒识）必须存在**，它同时是幻觉率和拒识率两个指标的载体。

### 5.3 Prompt 模板版本管理

所有 prompt 存 `configs/prompts/v1/*.jinja`，**版本号进 run_id**。这一条极其重要——DGPTE 那篇的实验证明 prompt 从 Case 1 到 Case 4 能让准确率从 51.84% 变到 90.26%，如果 prompt 版本没有严格管理，你的所有对比都无效。

四档 prompt（对齐 DGPTE 的 Case 1–4，保证可间接对比）：

| 档位 | 内容 |
|---|---|
| P1 | 裸问题 + 选项 |
| P2 | P1 + 资产类型说明 + 缺陷定义 |
| P3 | P2 + 判定要点（决策树文本化） |
| P4 | P3 + 每类 1 张 ICL 参考图 |

**注意显存**：P4 带参考图会让序列长度暴涨，**训练时用 P3，推理时用 P4**，这个训练/推理分布不一致必须单独做消融验证（Phase 7 的 A5）。

### 5.4 半自动 QA 生成（T4）

流程沿用已被验证的两步法：

1. 用商用 API（Qwen3.8-Max / Gemini 3 Pro）对训练集每张 crop 生成推理链初稿，**把真实标签一并输入**，让模型反推理由而非预测
2. 专家逐条核验推理链正确性，不核验结论（结论来自 ground truth）

**InsPLAD 是公开数据，可以合法走 API**——这是它相对私有数据的又一优势，写进论文。

**成本预算**：5,000 条 × 约 1,500 token 输入（含图）× 约 300 token 输出，按主流 API 定价约 $80–150。

### 5.5 Gold Test Set（`T25_gold`，人类任务）

从 test 集中分层抽 **600 条**，由 **2 名持证运维人员独立标注**，报 **Cohen's κ**。

- κ ≥ 0.75：可用，写进论文
- 0.6 ≤ κ < 0.75：需要第三人仲裁分歧项
- κ < 0.6：标注规范有问题，回炉重写选项定义

**这一条对电力类期刊的过审率影响最大**，不要省。智能体负责准备标注界面（可复用你已有的 `viewer.html`，加一个标注面板）和一致性计算脚本，标注本身必须人做。

---

## 6. Phase 3–8：实验矩阵

### 6.1 完整实验组

| 组 | 内容 | 模型/设置 | 预估耗时 | 论文中的作用 |
|---|---|---|---|---|
| **A** | CNN 基线 | ResNet-50 / EfficientNet-B3 / Swin-T / ConvNeXt-T | 8 h | 证明 VLM 有增量价值；且可与 InsPLAD 官方 Balanced Acc 直接对照 |
| **B** | 无监督 AD 基线 | **直接引用**已发表的 DifferNet (0.905) / CS-Flow (0.903) / AttentDifferNet | 0 h | 零成本获得可比数字 |
| **C** | 开源 VLM zero-shot | InternVL3.5-8B、Qwen3-VL-8B、Qwen2.5-VL-7B、MiniCPM-V 4.5、InternVL3.5-2B、Qwen3-VL-2B | 12 h | baseline，证明通用模型不够用 |
| **D** | 闭源上界 | Gemini 3 Pro / GPT-5.2 / Qwen3.8-Max（API，P1–P4 四档） | 6 h + API 费 | 上界参照 |
| **E** | 本文方法 | InternVL3.5-8B + LoRA + 代价敏感偏好优化 | 30 h | **主结果** |
| **F** | 规模消融 | 2B / 8B / 27B(QLoRA) 同方法 | 25 h | 效率-精度曲线 |
| **G** | 换基座泛化 | Qwen3-VL-8B、Qwen3.8-27B 换基座跑同方法 | 20 h | **挡掉"涨点来自基座"的质疑，这一列不能省** |
| **H** | 长尾交叉点 | 每类 4/8/16/32/64/128/all 七档 × 3 seed | 60 h | **C3 的核心，最耗时** |
| **I** | 量化部署 | BF16 / FP8 / NVFP4 / AWQ-INT4 推理（8B + 27B） | 10 h | **C4，独家角度** |
| **J** | 跨域泛化 | InsPLAD ↔ 私有国网数据双向 | 20 h | 若私有数据可用 |

**总计约 190 GPU-小时 ≈ 8–10 天连续运行**，加上调试与失败重跑，预留 3 周。

### 6.2 主实验（E 组）训练配置

```yaml
# configs/train_lora_internvl35_8b.yaml
model: OpenGVLab/InternVL3_5-8B
train_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: all-linear      # 但见下方"模块化消融"
freeze_vit: true                # 默认冻结，见 §6.3
learning_rate: 1.0e-4
lr_scheduler: cosine
warmup_ratio: 0.03
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # 有效 batch 16
gradient_checkpointing: true
bf16: true
max_pixels: 589824              # 768×768，ROI crop 不需要 1280²
cutoff_len: 4096
seed: 42
save_strategy: epoch
eval_strategy: epoch
metric_for_best_model: balanced_accuracy
```

**为什么 768² 而不是 1280²**：DGPTE 在 1280px 整图训练需要 490 GB 显存。你的输入是**已裁剪的 ROI crop**（中位数尺寸从审计项 7 得到，通常几百像素），高分辨率的边际收益远小于整图场景。这是你单卡的天然优势，但**必须用 Phase 7 的 A1 消融证明**（在 384/512/768/1024 四档上跑，画出精度-显存曲线），否则审稿人会问。

### 6.3 模块化 LoRA 消融（Phase 7 的 A2，强烈建议做）

DGPTE 的关键发现是：**只微调 LLM 层最优**，同时调 VE + MMA + LLM 反而更差。原因是 VE/MMA 微调前就已具备捕获判别信息的能力，缺的是 LLM 层检索关键 token 的能力；少样本下全模块微调容易过拟合。

你需要在 InsPLAD 上验证这个结论是否成立——**如果成立，是对已有发现的独立复现（审稿人喜欢）；如果不成立，是一个新发现（更好）**。四个设置：

| 设置 | ViT | Projector/MMA | LLM |
|---|---|---|---|
| M1 | 冻结 | 冻结 | LoRA |
| M2 | 冻结 | LoRA | LoRA |
| M3 | LoRA (lr×0.1) | LoRA | LoRA |
| M4 | LoRA | LoRA | LoRA |

### 6.4 Stage 3：代价敏感偏好优化（核心创新，别省）

在 SFT 之后加一轮 MPO/DPO，偏好对的构造规则：

| 偏好对类型 | chosen | rejected | 权重 |
|---|---|---|---|
| **漏检惩罚** | 正确识别缺陷 | 判为"正常" | **10.0** |
| 误报惩罚 | 正确识别正常 | 判为"有缺陷" | 1.0 |
| **幻觉强负例** | 「状态正常」 | 在 good crop 上编造缺陷 | 8.0 |
| 依据缺失弱负例 | 结论 + 判定要点 | 只给结论无依据 | 2.0 |
| 拒识合理性 | 模糊图上选"无法判断" | 强行给结论 | 3.0 |

权重矩阵 `C_ij` 由运维专家给定，**矩阵本身作为论文附录贡献**。

生成方式：用 SFT 后的模型对 train 集做 temperature=1.0 的多次采样（n=4），把答错的采样作为 rejected，ground truth 作为 chosen，按上表加权。脚本 `scripts/build_preference_pairs.py`。

### 6.5 长尾交叉点实验（H 组）——设计细节

这是论文 C3 的核心，也是最耗时的一组，必须设计干净。

```
自变量：每类训练样本数 n ∈ {4, 8, 16, 32, 64, 128, all}
路线：  {ResNet-50(ImageNet预训练), Swin-T, InternVL3.5-2B+LoRA, InternVL3.5-8B+LoRA, InternVL3.5-8B ICL(不训练)}
重复：  每格 3 个随机种子（41/42/43），报均值 ± 标准差
测试集：固定不变（full test），保证跨格可比
```

**核心图表**：横轴 log(n)，纵轴 Balanced Accuracy，两族曲线的交点位置。

**必须回答的三个子问题**（这决定了论文有没有 insight）：

1. 交叉点在哪？（预期在 n = 16–64 之间，但这是要测的）
2. 交叉点位置是否依赖缺陷类型？**纹理型缺陷（rust）vs 结构型缺陷（missing-cap）分开画**——这是最可能出好结论的地方
3. few-shot ICL（不训练）与 LoRA 微调的成本-收益拐点在哪？

InsPLAD 恰好提供了一把天然的样本量阶梯：4 / 22 / 64 / 90 / 555，跨两个数量级且同一采集条件。这种数据在公开数据集里非常罕见（通常的长尾基准是人为下采样出来的），**这一点要在论文里明确点出，是选题合理性的重要论据**。

### 6.6 量化实验（I 组）——三个维度都要报

对 E 组的最优 checkpoint（LoRA merge 后）做四档量化，每档报：

| 维度 | 指标 |
|---|---|
| **精度** | Balanced Acc、Macro-F1、**各缺陷类别的漏检率** |
| **可靠性** | **ECE 校准误差**、幻觉率、拒识率、代价敏感风险 R |
| **效率** | 首 token 延迟 P50/P99、单图端到端延迟、吞吐、**显存峰值** |

**核心 claim 与验证方法**：

> 通用榜单上量化"几乎无损"（平均掉 1.5–2 分），但在类别极不平衡、漏检代价远大于误报的工业场景中，损失**集中在边界样本上**，均值指标掩盖了这一点。

验证方式不是只看均值差，而是：
- 对 BF16 与量化版做**逐样本配对比较**，统计翻转率（BF16 对→量化错 的比例）
- 按 BF16 输出置信度分箱，看翻转集中在哪个置信度区间（预期集中在 0.5–0.7 的边界带）
- **McNemar 检验**判断差异是否显著（`scripts/metrics.py` 已实现）

这个分析方法是这一节的方法论贡献，比单纯报四行数字有价值得多。

---

## 7. 指标体系

### 7.1 对齐官方（保证可比）——必报

| 指标 | 任务 | 对照对象 |
|---|---|---|
| Balanced Accuracy | 缺陷分类 | InsPLAD 原论文表 |
| AUROC | 异常检测 | DifferNet 0.905 / CS-Flow 0.903 / AttentDifferNet |
| AP@50 | 目标检测 | InsPLAD 原论文表 |

### 7.2 本文增量——保证创新

| 指标 | 定义 | 实现 |
|---|---|---|
| 缺陷漏检率 | `FN / (TP + FN)`，按类别报 | `metrics.miss_rate` |
| 代价敏感风险 | `R = Σ_ij C_ij · P(ŷ=j | y=i)` | `metrics.cost_risk` |
| 幻觉率 | good crop 上声称有缺陷的比例 | `metrics.hallucination_rate` |
| 拒识率 / 拒识精度 | 选"无法判断"的比例，及其中确实模糊的比例 | `metrics.abstention` |
| ECE | 15 bins 等宽分箱期望校准误差 | `metrics.ece` |
| 依据可追溯率 | T4 输出中正确引用判定要点的比例 | 关键要素字符串匹配 + 人工抽检 |
| 格式失败率 | 无法解析成结构化答案的比例 | **必报**，见下 |

### 7.3 一个容易翻车的细节：格式解析

配电 Agent 那篇论文里出现 Qwen2.5-VL-32B zero-shot 只有 0.3% 的异常结果，几乎可以确定是 **prompt 模板与该模型不兼容导致输出格式解析失败**，而非真实能力差距——因为作者用的是"生成文本是否包含标签"的字符串匹配式准确率。

**你必须避免同样的错误：**

1. 所有 VLM 评测强制 JSON 输出，用 vLLM 的 **guided decoding / structured output**（`guided_json`）而非自由生成
2. **单独报告 format failure rate**，且格式失败**不计入准确率分母的错误项**，而是单独一列
3. 每个模型跑正式评测前，先跑 20 条 smoke test，格式成功率 < 90% 就调 prompt，不要直接跑全量

### 7.4 统计严谨性（审稿人会查）

- 所有点估计配 **bootstrap 95% CI**（n_boot = 2000）
- n < 30 的类别**只报 CI 不报点估计**
- 模型间对比用 **McNemar 检验**（配对二分类）或 **bootstrap 置信区间重叠判断**
- 所有实验固定 seed，H 组跑 3 seed 报 mean ± std
- 报告实际使用的所有随机种子和软件版本（`pip freeze > runs/<run_id>/requirements.lock`）

---

## 8. 智能体执行的 DAG

见 `tasks.yaml`。核心结构：

```
T00_setup ──► T01_check_env [gate]
                  │
                  ▼
              T10_audit [gate] ──► T11_split ──► T12_verify_splits [gate]
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    ▼                 ▼                 ▼
                              T20_build_crops   T21_build_vqa    T25_gold [human gate]
                                    │                 │
                                    └────────┬────────┘
                                             ▼
                                    T30_cnn_baseline (A组)
                                             │
                                             ▼
                                    T40_vlm_zeroshot (C组) [gate: 看 zero-shot 数字定方向]
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                   T50_sft_main (E)    T60_scale (F)     T70_backbone (G)
                          │
                          ▼
                   T51_preference (E-stage3)
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        T80_probe_fp4  T90_longtail  T75_ablation
              │           (H)            (A1-A5)
              ▼
        T81_quant (I)
              │
              └────────► T99_aggregate ──► T100_report
```

### 8.1 三个关键 gate 的判据

**Gate-A（T10_audit）**：见 §4.1。

**Gate-B（T12_verify_splits）**：自动断言，不通过 exit 1。

**Gate-C（T40_vlm_zeroshot）**——这个 gate 决定论文重心，必须人类介入：

| zero-shot Balanced Acc（T2 任务） | 含义 | 行动 |
|---|---|---|
| ≥ 85% | T2 太简单，没有区分度 | **重心移到 T3/T4**，T2 降级为 sanity check；同时考虑加入更难的细粒度选项 |
| 60–85% | 理想区间 | 按原计划全速执行 |
| ≤ 60% | 要么任务确实难，要么 prompt/解析有问题 | **先查格式失败率**，排除技术问题后再确认 |

### 8.2 智能体每个 task 必须写的 `state/<task_id>.json`

```json
{
  "task_id": "T50_sft_main",
  "run_id": "phase5_sft_internvl35_8b_20260820_143022",
  "status": "success",
  "started_at": "2026-08-20T14:30:22+08:00",
  "finished_at": "2026-08-20T21:11:05+08:00",
  "gpu_hours": 6.68,
  "peak_vram_gb": 81.3,
  "fallbacks_applied": ["batch_size: 4->2"],
  "key_metrics": {"balanced_accuracy": 0.871, "macro_f1": 0.854},
  "artifacts": ["runs/.../train/adapter", "runs/.../metrics.json"],
  "config_sha256": "..."
}
```

`T99_aggregate` 扫描所有 `state/*.json` 与 `runs/*/metrics.json`，写入 `results.sqlite`，并自动生成分析用的表格与 matplotlib 图（`scripts/make_tables.py`、`scripts/make_figures.py`）。

---

## 9. 时间线（6 周到初稿）

| 周 | 阶段 | 关键产出 | Gate |
|---|---|---|---|
| W1 | 环境 + 数据审计 + 划分 | data card、splits、MANIFEST | Gate-A、Gate-B |
| W1–2 | VQA 构造 + gold set 标注 | ElecVQA-Bench v0.1、κ 值 | 人类标注 |
| W2 | CNN 基线 + VLM zero-shot | A、C、D 组结果表 | **Gate-C** |
| W3 | 主实验 SFT + 偏好优化 | E 组主结果 | — |
| W3–4 | 长尾曲线（H 组，最耗时） | 交叉点图 | — |
| W4 | 换基座 + 规模消融 + 消融组 | F、G、A1–A5 | — |
| W5 | FP4 探针 + 量化实验 | I 组三维表 | FP4 降级判定 |
| W5–6 | 聚合 + 作图 + 撰写 | 初稿 + 开源仓库 | — |

**并行机会**：D 组（API 闭源上界）不占 GPU，可与任何 GPU 任务并行；B 组零成本；W3–4 的长尾曲线可在夜间无人值守跑。

---

## 10. 开源与可复现（这是本论文的核心卖点，必须做到位）

| 产出 | 位置 | 说明 |
|---|---|---|
| VQA 构造脚本 | GitHub | 从 InsPLAD 原始标注一键重建基准 |
| 划分文件 + MANIFEST | GitHub | 含 SHA256，杜绝"我用的划分和你不一样" |
| Prompt 模板（P1–P4） | GitHub | 四档全部公开 |
| 评测 harness | GitHub | 一条命令复现所有表格 |
| LoRA adapter 权重 | HuggingFace | Apache 2.0 |
| 代价矩阵 + 标注规范 | 论文附录 + GitHub | 专家知识的沉淀 |
| 完整仓库快照 | **Zenodo 打 DOI** | 开源 SCI 期刊的可复现性审查越来越严，永久归档能挡掉很多质疑 |

**related work 里必须明确写一句**：现有五项电力多模态工作（Power-LLaVA、PowerGPT、DGPTE、Power-VGLM、配电 Agent）**全部基于不可获取的私有数据**，并逐篇给出其 data availability 声明作为证据。这一句把你的贡献定位讲透了，是全文最有力的一段。

---

## 11. 风险登记表

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| sm_120 上 NVFP4 kernel 不可用 | **高** | I 组降级 | §1.2 的探针 + 自动降级，把"不支持"本身写成发现 |
| crop 无法反解父图 ID → 无法防泄漏 | 中 | 划分方案重做 | pHash 聚类兜底，Gate-1 拦截 |
| zero-shot 就有 85%+，任务无区分度 | 中 | 重心调整 | Gate-C 的分支预案已写好 |
| gold set κ < 0.6 | 中 | 标注规范回炉 | 先做 50 条 pilot 标注，κ 达标再做全量 |
| 长尾组 60 GPU-h 跑不完 | 中 | 砍格数 | 优先保 {4,16,64,all} 四档 × 2 路线，seed 降到 2 |
| 私有数据审批不下来 | 中 | 删 J 组和 T5 | **论文按 A+B+D 组合依然成立**，这是本方案最重要的鲁棒性设计 |
| InsPLAD 官方划分与自建划分不可比 | 低 | 无法对照官方数字 | 同时报"官方划分"和"自建防泄漏划分"两套数字 |
| API 生成的 T4 数据质量不达标 | 中 | T4 任务弱化 | 专家核验环节前置，先抽 100 条评估通过率 |

---

## 12. 需要你现在确认的六件事

智能体在 `T00` 之前需要这些信息，请逐条回复：

1. **私有国网数据本轮是否可用？** 若可用，给出：类别数、各类样本量、山火部分的标注形式（火点框 / 烟雾框）、是否有气象或距离元数据。若不可用，智能体直接按"仅 InsPLAD"路径执行。
2. **服务器接入方式**：SSH 地址、是否有 Docker、数据是否已在服务器上（还是需要从本地 `D:\05Python\...` 上传）。
3. **API 预算**：D 组闭源上界和 T4 数据生成需要商用 API，预算上限是多少？（建议 ≥ $300）
4. **两名标注专家是否已落实？** gold set 标注约需 2 人 × 8 小时。
5. **目标期刊确认**：*Drones* / *Remote Sensing* / *IEEE Access* 中优先哪个？这会影响论文中"遥感角度"和"电力角度"的配比，进而影响 T1 任务的权重。
6. **是否接受"删掉分级章节"的降级方案？** 若你坚持要分级，且私有数据不可用，那么整个 T5 只能靠自建伪标签，我不建议——风险大于收益。

---

## 附：配套文件清单

| 文件 | 用途 |
|---|---|
| `tasks.yaml` | 智能体执行的 DAG，含依赖、gate、超时、重试策略 |
| `scripts/check_env.py` | Gate-0 环境检查 |
| `scripts/audit_dataset.py` | Phase 0 数据审计，产出 data card |
| `scripts/build_splits.py` | 防泄漏分组划分 + 自动断言 |
| `scripts/build_vqa.py` | ROI 裁剪 + 四类 QA 构造 |
| `scripts/metrics.py` | 全部指标实现（含 bootstrap CI、McNemar、ECE） |
| `scripts/eval_vlm.py` | vLLM 评测 runner，含 guided JSON 与格式失败统计 |
| `scripts/probe_fp4.py` | sm_120 上的 NVFP4 可用性探针 |
| `configs/env.yaml` | 路径、种子、显存预算等全局配置 |
| `configs/models.yaml` | 所有待测模型的 HF ID 与推理参数 |
| `configs/train_lora_internvl35_8b.yaml` | 主实验训练配置 |
| `configs/cost_matrix.yaml` | 代价矩阵（需专家填写） |
