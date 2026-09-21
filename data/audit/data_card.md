# InsPLAD Data Card (auto-generated)

> 生成时间：2026-08-19T09:21:08　脚本版本：1.0


## 1. 目标检测子集

| split | JSON 图数 | 磁盘图数 | 一致 | 标注框 |
|---|---|---|---|---|
| train | 7981 | 7935 | ❌ | 22635 |
| val | 2626 | 2626 | ✅ | 6324 |

- 类别数：18　发现 18 类, 官方文献称 17 类
- val 无样本的类别（无法算 AP，评测须排除）：`['sphere']`
- bbox 异常数（越界/退化，最多列 50）：0

## 2. 有监督缺陷分类子集

| 资产 | train | val/test | 标签 | 异常标记 |
|---|---|---|---|---|
| glass-insulator | 1381 | 59 | good, missing-cap | OK |
| lightning-rod-suspension | 658 | 251 | good, rust | OK |
| polymer-insulator-upper-shackle | 1482 | 64 | corrosão, good, rust | OK / ⚠命名不一致 |
| vari-grip | 998 | 281 | good, nest, rust | OK |
| yoke-suspension | 589 | 5762 | good, rust | ⚠ train(589) 比 val/test(5762) 小 9.8× , 疑似划分反转 |

- **父图 ID 反解成功率：100.00%**（regex 11525 / fallback 0 / failed 0）
- crop 短边分布：min 84 / p25 493 / median 690 / p75 887 / max 1080（n=622）
  - → 建议 `max_pixels` 设为 p75 的 2–3 倍，不要盲目上 1280²
- 损坏/零字节文件：0

## 3. 无监督异常检测子集

| 资产 | train | val/test | 标签 | 异常标记 |
|---|---|---|---|---|
| damper-preformed | 816 | 204 | good | OK |
| damper-stockbridge | 5215 | 1756 | good, rust | OK / ⚠命名不一致 |
| glass-insulator | 2298 | 671 | good, missing-cap | OK / ⚠命名不一致 |
| glass-insulator-big-shackle | 207 | 62 | good, rust | OK / ⚠命名不一致 |
| glass-insulator-small-shackle | 210 | 54 | good, nest | OK / ⚠命名不一致 |
| glass-insulator-tower-shackle | 165 | 37 | good, rust | OK / ⚠命名不一致 |
| lightning-rod-shackle | 165 | 34 | good, rust | OK / ⚠命名不一致 |
| lightning-rod-suspension | 462 | 167 | good, rust | OK / ⚠命名不一致 |
| plate | 182 | 36 | good, peeling-paint | OK / ⚠命名不一致 |
| polymer-insulator | 2421 | 627 | good, torned-up | OK / ⚠命名不一致 |
| polymer-insulator-lower-shackle | 1306 | 642 | good, rust | OK / ⚠命名不一致 |
| polymer-insulator-tower-shackle | 42 | 22 | good, rust | OK / ⚠命名不一致 |
| polymer-insulator-upper-shackle | 935 | 337 | good, rust | OK / ⚠命名不一致 |
| spacer | 75 | 19 | good | OK |
| vari-grip | 477 | 225 | good, nest, rust | OK / ⚠命名不一致 |
| yoke | 1329 | 332 | good | OK |
| yoke-suspension | 4834 | 1256 | good, rust | OK / ⚠命名不一致 |

- **父图 ID 反解成功率：100.00%**（regex 27620 / fallback 0 / failed 0）
- crop 短边分布：min 52 / p25 311 / median 596 / p75 852 / max 1080（n=1277）
  - → 建议 `max_pixels` 设为 p75 的 2–3 倍，不要盲目上 1280²
- 损坏/零字节文件：0

## 4. 近重复与泄漏风险

- 已哈希图像：12000
- 精确碰撞组：203
- **跨子集碰撞组：17（占比 0.142%）**
- 判定：OK

## 5. 稀有类别清单（n < 30，只报 bootstrap CI，不报点估计）

- `nest` @ `glass-insulator-small-shackle`：n = 1
- `rust` @ `lightning-rod-shackle`：n = 4
- `peeling-paint` @ `plate`：n = 4
- `rust` @ `glass-insulator-tower-shackle`：n = 7
- `rust` @ `glass-insulator-big-shackle`：n = 10
- `rust` @ `polymer-insulator-tower-shackle`：n = 12
- `rust` @ `damper-stockbridge`：n = 18
- `torned-up` @ `polymer-insulator`：n = 22

## 6. Gate-A 判定

- ✅ **parent_id_resolvable**：父图 ID 反解成功率 100.00%（阈值 95%）
- ✅ **no_cross_subset_leak**：跨子集近重复占比 0.142%（阈值 1%）
- ❌ **no_split_anomaly**：检查各资产 train/val 规模是否反转（yoke-suspension 重点看）
- ✅ **no_corrupt_files**：损坏/零字节文件数