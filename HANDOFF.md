# 交接笔记 — GNSS ZTD + FuXi → ERA5 同化

> 最后更新：2026-09-18（**口径变更**：只用 `station_halo` / 只看 70 通道 /
> obs 与 H(FuXi) 统一归一化 ⇒ 见 **§14**）
>
> **明天怎么继续**：直接说 ——「读 `da_ngl/HANDOFF.md`，继续」即可。**先看 §14**（新口径），
> 再看 §13.6（halo 实验的结论）。

---

## 1. 任务

以 **ERA5 为 label**，用 **NGL（GNSS ZTD，5 分钟）+ FuXi 预报** 做**数据同化**，
产出分析场。第一步（数据集）和第二步（训练+评估）今天都做完了。

---

## 2. 今天完成了什么

### 2.1 统一网格（已定稿）

* 0.25°，**lat 36.50–56.25（80）× lon −5.25–24.50（120）**
* 站点→格点映射：`dataset/ngl_europe_0p25_80x120_station_grid_map.parquet`
  （每格一站，多站格随机取一个 seed=2021；1378 格有站，8222 格 mask）
* 源场（FuXi/ERA5/IMERG，全球 720×1440）用**最近邻**采到该网格

### 2.2 三个 Zarr（`da_ngl/dataset/`）

| 文件 | 内容 | 形状 |
| --- | --- | --- |
| `ngl_europe_0p25_5min.zarr` | `ztd[time,lat,lon]`，5 分钟；`station`/`mask`；`ztd_train_mean/std` | 394272×80×120 |
| `fuxi_europe_0p25.zarr` | `z[init,step,channel(69),lat,lon]`，**只有 lead=6h** | 5476×1×69×80×120 |
| `label_europe_0p25.zarr` | `label[time,channel(71),lat,lon]` | 5476×71×80×120 |

* 时间：**2022-01-01 → 2025-10-01**（半开；上限取 IMERG 结束处）
* 划分只写在 attrs：train `2022-01-01..2024-05-01`，val `→2025-01-01`，test `→2025-10-01`
* 构建脚本：`da_ngl/preprocessing/`（`common.py` / `build_ngl_zarr.py` /
  `build_fuxi_zarr.py` / `build_label_zarr.py` / `map_stations_to_grid.py`）
* 说明文档：`da_ngl/README.md`

### 2.3 训练代码（`da_ngl/main_code/`）

基于 `xuxiaoze/for_zrx/train_packet` 改造，数据换成上面三个 zarr：

```
configs.py / configs_smoke.py / train.sh / train_FSDP.py / plot_results.py
main/model/{assimilation.py,build_optimizer.py}
main/utils/{utils.py,utils_data.py}
```

### 2.4 训练已跑完

* **3 卡 FSDP**，20000 步，36 个 epoch
* checkpoint：`main_code/work_dir/model/iteration_20000.pth`（每个 epoch 一个，共 36 个）
* 日志：`main_code/work_dir/logs/`、`da_ngl/logs/fuxi_da.log`

### 2.5 评估与画图

`main_code/plot_results.py`，输出到 `main_code/work_dir/plots/`：
`loss_curve.png` / `channel_metrics.png` / `maps_{z500,t850,r700,t2m,msl}.png` /
`timeseries.png` / `timeseries_tp.png` / `tp_maps.png` / `metrics.csv` / `metrics.json`

---

## 3. 关键约定（明天别搞错）

* **FuXi 背景场**：`init = T − fcst_step×6h`，`step = fcst_step×6h`，
  当前 `fcst_step = 1`（store 里只有 lead 6h）。数据集会按 `fcst_step` 去
  zarr 的 `step` 轴上定位 lead，不是硬编码。
* **标准化**：FuXi / ERA5 用 `obs-grid_qc/mean_std/mean_era5.npy`、`std_era5.npy`
  （两者同一套）；**NGL 用训练集全局 mean/std**（2333.6450 / 119.7633，存在 store 里）。
* **tp 变换**：`clip(min=0) → log1p → (x−μ)/σ`；ERA5 的 tp 单位是 m 所以先 ×1000，
  IMERG 已经是 mm 所以**不乘 1000**。
* **标签 71 通道**：0–68 ERA5、69 = IMERG tp、70 = ERA5 tp（两者都建在 store 里）。
  训练用哪个 tp 由 `configs.tp_label_source` 控制（`'imerg'` / `'era5'`，**当前 `'era5'`**）：
  `AssimilationDataset._select_label()` 把选中的那个放到第 69 通道，训练仍只读 70 通道
  （`configs.label_n_chans = 70`）。评估端会同时报 model vs 两个 tp 以及两产品的不一致度。
* **模型要求 H、W 能被 16 整除**：80 满足，120 会被自动 replicate pad 到 128、输出裁回。
* 残差结构 `out = decoder(...) + bg` **只加在前 69 通道**（tp 没有 FuXi 背景）。

---

## 4. 当前结果

| 项 | 数值 |
| --- | --- |
| train loss | 0.1462 → **0.0946** |
| val loss | 0.0923 → 0.0893（最好 0.08892 @ epoch 11） |
| test：分析（69 通道） | **0.08412** |
| test：FuXi 背景（69 通道） | **0.08379** |
| test：气候态 | 1.036 |
| 优于背景的通道数 | 26 / 69 |
| 分析相对背景的平均改动 | 0.0104（背景误差的 12%） |
| 模型 tp（vs IMERG） | 0.329（气候态 0.855，降 61.5%） |
| ERA5 tp vs IMERG tp | **0.436**（比模型自己的误差还大） |

**结论：模型基本退化成恒等映射，整体没超过 FuXi 6h 背景场（−0.40%）。**
有站点格点上同样没改善（−0.49%），但改动幅度更大（0.0112 vs 0.0103），
说明网络确实"看到"并用了 GNSS 观测，只是没换来误差下降。

---

## 5. 下一步候选（今天讨论过，还没做）

1. **先确认瓶颈**：把观测全置零再评估一遍。若结果几乎不变 → 问题在数据端（观测没信息量），
   不是训练问题。
2. **换更长时效**：重建 fuxi zarr 用 `--leads 24`（或 48），改 `fcst_step=4`，
   背景场弱下去同化才有空间。
3. **让观测更可用**：加通道（到最近站点距离、站点密度、逐帧 mask 已经有了），
   或把 ZTD 先做空间扩散再喂进去。
4. **把 tp 拆出去**：tp 的误差是状态量的 4 倍，会分走容量；或直接 `model_out_chans=69`
   只做纯状态量同化，看纯同化的效果。
5. 调参方向：`model_embed_dim` / `num_iteration` / 学习率、损失加权。

---

## 6. 常用命令

```bash
# 环境
conda activate gnss

# 数据集（重建；NGL 有缓存，很快）
cd /cpfs01/.../gnss/da_ngl
python preprocessing/build_ngl_zarr.py   --force --cache dataset/_cache/ngl_5min_series.npz
python preprocessing/build_fuxi_zarr.py  --force --workers 40 --leads 6
python preprocessing/build_label_zarr.py --force --tp-transform standardized

# 训练（本容器只有 1 张卡；3 卡在另一台机器）
cd main_code
CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=22336 nohup bash train.sh > /dev/null 2>&1 &
# 进度看 da_ngl/logs/{model_id}_{模型配置}.log（stdout 那边只有 WARNING）

# 评估出图（结果直接写进同一次实验的 results 文件夹）
CUDA_VISIBLE_DEVICES=0 python plot_results.py --split test
```

---

## 7. 今天踩过并已修掉的坑（别重复踩）

* **zarr 默认 `fill_value`（0/False/''）与真实值冲突** → xarray 把 `time==0`、
  `lon==0.0`、`mask==False` 变成 NaN/NaT。已用 `common.finalize_store` 把坐标类数组的
  `fill_value` 清成 `null` 并写 consolidated metadata。**新增 store 必须调用它。**
* **模型要求 H、W 能被 16 整除**（内部 fussion 两次下采样）；120 不满足，已在模型入口 pad。
* **损失 NaN**：先做差再乘掩码会 `NaN×0=NaN`，已改成先 `nan_to_num` 再作差。
* **DataLoader worker 无法 pickle**：Dataset 里不能存 config 模块对象。
* **多卡两个坑**：① 收尾评估写在 `if rank==0` 里会造成 `all_reduce` 死锁；
  ② `summon_full_params + model.module.state_dict()` 在 SHARD_GRAD_OP 下存的是分片权重，
  已改用 FSDP 的 `FULL_STATE_DICT` API。
* **`nohup` 不能跟 `VAR=val` 前缀**（会当成程序名，exit 127）；要写成
  `VAR=val nohup bash ...` 或用 `env`。
* **`channel` 数组 dtype**：加 `era5_tp`（7 字符）时必须把 `<U5` 加宽，否则截断。
* **训练日志里 `Test model=0.0876 background=0.0838` 不可比**：前者含 tp（70 通道）、
  后者不含（69 通道）。`plot_results.py` 已在同等 69 通道上重算。

---

## 8. 给明天的一句话提示（可直接复制）

> 读 `da_ngl/HANDOFF.md`，继续做同化。**先看第 13 节（2026-09-17）**：把 label loss 限制在
> "站点 + halo3"（`loss_domain='station_halo'`、区域外权重 0）**是负收益**——站内 69ch 从 −0.245%
> 掉到 −0.423%（λ0.2）/ −0.915%（λ0.1），去掉 msl 后从 **+0.375% 掉到 +0.132%**，msl 也照旧没被治好。
> 两个原因（§13.4）：label loss 的分母 9600→5778 格，等效 λ 被稀释成 0.117 / 0.059；被丢掉的那 39.8%
> 格点其实是有学习信号的（§12 那轮非站格点改善了 +0.15%）。**下一步：把 `loss_domain` 改回 `'full'`，
> 然后做 §12.8 第 1 项——只摁 msl**。⚠ §13.5：`configs.py` 里 `loss_domain` 的默认值现在是
> `'station_halo'`，重跑老 checkpoint 的评估要显式 `--set loss_domain=full`，否则指标会被换成区域口径。

---

## 9. 2026-09-11 更新

### 9.1 已修：图的时间轴是 2022

`plot_results.py` 里画图用的是 `label_time[:n]`（store 全局轴的前 n 个 = 2022-01-01 起），
而测试样本其实是第 4384–5475 号（2025-01-01 起）。**只是标签错，数值不受影响。**
已改成按样本取 `sample_times = label_time[[s[0] for s in dataset.samples]]`，
`timeseries*.png` 和 `maps_*.png` / `tp_maps.png` 的标题现在都是 2025 年。
原图在 `work_dir/plots/`（2h 版）已用同样数值重新出过。

### 9.2 观测窗拉长到 6h：零结果

`obs_frames = 25 -> 73`（2h -> 6h，窗口 `[T-6h, T]`，正好覆盖整个 FuXi 预报时段），
3 卡重训 20000 步（日志 `da_ngl/logs/fuxi_da_6h.log`）。同等 69 通道下：

| | 2h（25 帧） | 6h（73 帧） |
| --- | --- | --- |
| 分析 | 0.0841200 | 0.0840768 |
| 背景 | 0.0837865 | 0.0837865 |
| 相对背景 | −0.398% | −0.346% |
| 优于背景通道 | 26/69 | 29/69 |
| 平均改动 | 0.010428 | 0.010245 |
| best val | 0.0889215 | 0.0889266 |
| tp（vs IMERG） | 0.32921 | 0.33005 |

**观测多 3 倍，结果几乎不动**，说明瓶颈不是观测给得太少。失效模式也一模一样
（最差仍是 `t925 r925 r1000 t1000 msl z1000`；改动最大仍是低层 `r` 和 `t`）。

⚠️ 6h 那次训练把 `work_dir/model/` 里的 2h checkpoint **全覆盖了**（2h 的数值还留在
`work_dir/plots/metrics.json`）。6h 的图在 `work_dir/plots_6h/`。

### 9.3 改动：tp 训练标签可切换（当前用 ERA5 tp）

* `configs.py` 新增 `tp_label_source = 'era5'`（可选 `'imerg'`）
* `main/utils/utils_data.py`：`AssimilationDataset._select_label()` 按该开关拼 70 通道标签
  （0..68 ERA5 + 选中的 tp）。请求 71 通道时仍返回原始 store（评估要两个 tp）
* `plot_results.py`：按同一开关构造 `truth`，并同时报告
  `mae_modeltp_vs_era5tp_std` / `mae_modeltp_vs_imergtp_std` / `mae_era5tp_vs_imerg_std`
* 动机：ERA5 tp 与 IMERG tp 的不一致度是 0.436，比模型自己的 tp 误差（0.329）还大，
  训练目标和状态通道（全部来自 ERA5）自相矛盾

### 9.4 一个还没验证的强假设

观测是**绝对 ZTD**，背景场没有 ZTD 通道 -> 网络无法构造增量（obs − H(bg)），
只能拟合绝对 ZTD。而绝对 ZTD 里 **85% 的方差是站点静态偏移**
（跨站时间均值 std 0.92σ vs 站内时间 std 0.39σ）。
计划中的验证：把观测换成"减掉站点气候态"的距平，不需要重训就能测。

---

## 10. 输出目录约定（2026-09-11 起）

一次实验 = 一个文件夹：

```
main_code/work_dir/results/{model_id}_{实验配置}/
    {model_id}_{模型配置}.pth     <- 只保留 val 最好的那个模型
    summary.json                  <- 最佳 val / 逐 epoch loss / 背景-only / test
    train_loss.npy val_loss.npy lr.npy
    loss_curve.png channel_metrics.png maps_*.png timeseries*.png tp_maps.png
    metrics.csv metrics.json
```

日志（信息量精简过，不再每步每卡都写）：

```
da_ngl/logs/{model_id}_{模型配置}.log
```

三个 tag：

| tag | 含义 | 默认 |
| --- | --- | --- |
| `model_id` | 入参，文件夹/文件前缀 | `configs.model_id = 'model'` |
| `实验配置` | 数据/同化设置 | 自动：`lead6h_obs6h_era5tp` |
| `模型配置` | 网络结构 | 自动：`ed128_d222` |

自动 tag 的规则：`lead{fcst_step*6}h_obs{obs_frames*5//60}h_{tp_label_source}tp`
和 `ed{model_embed_dim}_d{depth}`。2026-09-15 起 `实验配置` 还会自动带这些后缀：
`_bgtp`（背景含 FuXi tp）、`_w{站内权重}-{无站权重}`（站点加权损失，如 `_w1-0.1`）、
`_zeroobs`、`_both`/`_residual`（obs_mode）。想自己起名就在 configs 里写死字符串，
或命令行覆盖：

```bash
python train_FSDP.py --configs configs --model_id exp3 --exp_tag obs6h_v2 --arch_tag ed128_d222
python plot_results.py --model_id exp3 --exp_tag obs6h_v2 --arch_tag ed128_d222 --split test
```

其他改动：

* 训练**只在 val 变好时**存模型（`configs.min_delta` 控制阈值），旧的多 checkpoint 不再产生
* 跑完会先 reload 最好的那个 checkpoint 再做最终评估，所以 `summary.json` / 出图对应的是 best
* 进度行改成每 `configs.log_interval = 100` 步一行、且只有 rank 0 写；`[Data]/[Process]/[Model]`
  计时和 `[MaxMem]` 去掉了（11MB 的 log -> 3KB 量级）
* logger 的 stdout handler 降成 WARNING，所以 `nohup ... > xxx.log` 那个重定向文件不会
  再复制一份完整日志

---

## 11. 2026-09-15 更新

### 11.1 lead 24h 的数据集（新建）

`preprocessing/build_fuxi_zarr.py` 加了两个参数（默认值都保持旧行为）：

* `--init-pad-hours N`：把 init 窗口两端各扩 N 小时。lead 24h 时 `init = T - 24h`，
  不扩的话每个 split 两端会各少 4 个样本；`--init-pad-hours 24` 后三个 split 与 6h
  **逐样本对齐**（train/val/test = 3403/980/1092，label 与 obs 索引完全相同）。
* `--with-tp`：额外存 FuXi 自己的 tp 作 channel 69（70 通道），它本来就和标签同一套
  log1p 标准化空间，可以直接用。

产物：

| 文件 | 内容 |
| --- | --- |
| `dataset/fuxi_europe_0p25_24h.zarr` | lead 24h，5484 init（2021-12-31 00:00 起），step=[24]，69ch，4.6GB |
| `dataset/fuxi_europe_0p25_24h_70ch.zarr` | 同上 + FuXi tp，70ch，4.7GB |
| `dataset/ztd_fuxi_europe_0p25_24h.zarr` | H(FuXi@24h) 站格点 ZTD [mm]，5476 时刻，156MB |

`build_ztd_fuxi_zarr.py` 加了 `--set`（可直接覆盖 `fcst_step` / `fuxi_zarr`，不用另写配置模块）。
⚠️ 跑 24h 的 innovation 实验时**必须同时** `--set ztd_fuxi_zarr=.../ztd_fuxi_europe_0p25_24h.zarr`，
否则会静默使用 6h 的 H(FuXi)。

背景误差（test，69ch 纬度加权）：**6h 0.0838 → 24h 0.1261（+50%）**。

### 11.2 lead24 的五次实验（全部跑完）

test split，1092 样本，69 通道纬度加权；"相对背景 = 100×(背景−分析)/背景"，负 = 比背景差。
站内口径 = 只统计 1378 个有站格点。

| 实验（model_id） | 背景 | 损失权重 | 观测 | 分析(全格) | 相对背景 | 优通道 | 站内分析 | 站内相对 | tp MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lead24` | 69ch | 1/1 | 真 | 0.12634 | −0.214% | 24/69 | 0.12738 | −0.232% | 0.3060 |
| `lead24_obs_zero` | 69ch | 1/1 | 置零 | 0.12630 | −0.183% | 26/69 | 0.12732 | −0.180% | 0.3096 |
| `lead24_with_fuxi_tp` | 70ch | 1/1 | 真 | 0.12617 | −0.076% | 33/69 | 0.12719 | −0.079% | **0.2371** |
| `lead24_with_fuxi_tp_obs_zero` | 70ch | 1/1 | 置零 | 0.12617 | −0.077% | 33/69 | 0.12719 | −0.081% | **0.2371** |
| `lead24_fuxi_tp_innov`（obs_mode=both） | 70ch | 1/1 | 真(绝对+innovation) | 0.12622 | −0.119% | 36/69 | 0.12723 | −0.113% | 0.2381 |
| `lead24_fuxi_tp_grid_weight`（w1-0.1） | 70ch | 1/0.1 | 真 | 0.12639 | **−0.250%** | 25/69 | 0.12724 | **−0.119%** | 0.2383 |

参照：背景 0.12607（全格）/ 0.12709（站内），气候态 1.03624，tp 气候态 0.82520。
完整表格另存 `test/lead24_comparison.md` / `.csv`。

### 11.3 四个假设全部被否掉

| 假设 | 实验 | 结果 |
| --- | --- | --- |
| 观测表示不对（绝对 ZTD 无法构造增量） | abs vs abs+innovation（24h） | 无改善（−0.119% vs −0.076%） |
| 观测没信息量 | 观测置零 | 与真观测差 1e-6 |
| 背景太强、没有空间 | lead 6h vs 24h | 24h 更差 |
| 无站格点主导梯度 | 站点加权 w1-0.1 | **显著变差**（配对 t = −24，95% CI [−0.188, −0.159] pp） |

另外两个直接测量：

* 网络修正量 δ 与背景误差 e 的相关 **corr(δ,e) = +0.05**；把 δ 缩放到最优系数 a*=0.40 也只值 **+0.16%**。
* 静态后处理上限（逐格点逐通道仿射，直接在 test 上拟合）**只有 +0.60%**；在 train 上拟合迁移到 test ≈ 0。
* 24h 误差方差分解：**46.5% 是 6h 就有的结构性差异 + 56.6% 是 18h 预报新增 + 交叉 −3.1%**
  （corr(e6, d) = −0.03，即"多预报 18 小时新增的误差"与 6h 误差无关，属于丢掉的信息）。


用训练期逐站逐通道最小二乘拟合 `x_a = x_bg + K·(obs − H(bg))`，在 test 上评估
（1378 站格点，69 通道，纬度加权）：

| 方法 | 参数 | 训练集 | 测试集 | 相对背景 |
| --- | --- | --- | --- | --- |
| 背景（恒等） | — | 0.13590 | 0.12719 | — |
| 线性 DA：pooled K | 69 | 0.13567 | 0.12705 | +0.109% |
| 线性 DA：逐站 K（λ=0） | 95k | 0.13541 | 0.12693 | +0.203% |
| **线性 DA：逐站 K（λ=1，K/(1+λ)）** | 95k | 0.13535 | **0.12671** | **+0.376%** |
| 参照：CNN（lead24+FuXi tp） | 73.7M | — | 0.12719 | −0.079% |

逐通道改善（λ=1）：**r700 +2.31%、r850 +2.12%、r600 +1.67%、r500 +1.00%、r925 +0.92%**、
z850 +0.61%、t2m +0.51%；变差的都是 ZTD 约束不到的高层/风场（r50 −0.30%、r200 −0.16%）。

**含义：信息确实在观测里，一个 95k 参数的线性同化就取出了 0.38%，而 73.7M 的网络是负的。**

### 11.5 innovation 的信噪比（修正后的正确数字）

| | 训练期 | 测试期 |
| --- | --- | --- |
| 观测异常 std | 48.2 mm | 46.6 mm |
| std(obs − H(ERA5)) | **11.3 mm** | **10.9 mm**（= 算子+代表性误差） |
| 信号 std（H(ERA5)−H(FuXi)）6h / 24h | 7.4 / 10.3 mm | 6.7 / 9.8 mm |
| corr(innovation, 真列误差) 6h / 24h | +0.21 / **+0.44** | +0.23 / **+0.48** |

即 lead 24h 的 innovation 比 6h 干净得多（相关翻倍），但网络仍然没利用它——
这进一步把问题定位在训练目标上，而不是观测信息量。

⚠️ 我在中途把 NGL store 的时间轴单位当成小时（实际是 **minutes**），导致帧索引错位，
一度报出"信号只占 4%"的错误结论，已作废。**一律用 `common.decode_time_axis`**。

### 11.6 代码改动清单

| 文件 | 改动 |
| --- | --- |
| `configs.py` | 新增 `include_fuxi_tp`（默认 False）、`loss_station_weight` / `loss_nostation_weight`（默认 1.0/1.0） |
| `build_optimizer.py` | `_LatWeightedLoss.set_cell_weight((H,W))`：纬度权重上再乘逐格点权重；(1,1) 时与旧行为**逐位相同** |
| `utils.py` | `bg_chans(cfg)`、`station_cell_mask(cfg)` / `station_cell_weight(cfg)`（含 mask 极性校验）、exp_tag 新后缀 |
| `utils_data.py` | 背景按 `bg_chans(cfg)` 切通道；开了 `include_fuxi_tp` 但 store 只有 69 通道时**直接报错** |
| `train_FSDP.py` | 同步 `model_bg_chans`、安装站点权重并打印 `[Loss]` 行、summary/ckpt 记录 `include_fuxi_tp`+`obs_mode`+损失权重 |
| `plot_results.py` | 新增站内口径：`mae_*_69ch_station`、`improve_pct_69ch_station`、`n_channels_better_station`、`n_station_cells` |
| `build_fuxi_zarr.py` | `--init-pad-hours`、`--with-tp` |
| `build_ztd_fuxi_zarr.py` | `--set KEY=VALUE` |

### 11.7 今天踩的坑

1. **NGL store 的时间单位是 minutes**（`[0,5,10] minutes since 2022-01-01`），不是 hours。
   手写脚本时硬编码 `unit='h'` 会让 `searchsorted` 出来的帧索引错位，观测就配到了错误时刻。
2. **不要在 run 跑着的时候改 `utils_data.py`**：train/val 的 DataLoader worker 启动时就 fork 好了，
   而 test 的 worker 到收尾评估才第一次 fork——会拿到新代码 + 旧对象，报
   `AttributeError: 'AssimilationDataset' object has no attribute 'bg_n_chans'`，
   `train.sh` 因为 `set -e` 直接终止，链式评估不会跑（那次是手动补的 `plot_results.py`）。
3. 开了站点加权后，**val-best 的选择标准也是加权损失**，所以不同权重档位的 `best_val` 不可比；
   test 指标（`plot_results.py`）保持不加权，另加站内口径。
4. 同 `model_id` 的日志（追加）与 ckpt 文件名会被复用，只靠 exp_tag 区分实验时日志会混在一起——换 `model_id`。

### 11.8 下一步（按优先级）

1. **观测一致性损失**：`J = J_b + λ·|H(x_analysis) − obs| / σ_o`（σ_o ≈ 11 mm，算子已可微）。
   这是唯一能给网络"往哪个方向改"的梯度项；现在的损失里完全没有。
2. **观测去静态偏差**：逐站减掉 `(obs − H(bg))` 的长期均值，别让网络先学 1378 个常数偏移。
3. **混合方案**：把线性分析 `x_bg + K·d` 当额外输入通道（或残差基准），网络上只学修正。
4. **线性 DA 扩展成全网格**（把站格点的增量做空间传播），补上与 CNN 同口径的全格点比较。
5. （可选）**oracle 实验**：用 H(ERA5) 当完美观测，检验架构上限（需要新建 `ztd_era5_*.zarr`，
   脚本已规划：把 `ztd_profile_surface` 作用到 label store 的 0–68 通道）。

### 11.9 今天新增的产物

```
dataset/  fuxi_europe_0p25_24h.zarr, fuxi_europe_0p25_24h_70ch.zarr, ztd_fuxi_europe_0p25_24h.zarr
test/     pipeline_flowchart.{png,md,dot}, model_arch.svg, model_arch.md, model_arch_blocks.svg,
logs/     build_fuxi_24h.log, build_fuxi_24h_70ch.log, build_ztd_fuxi_24h.log
results/  lead24_lead24h_obs6h_era5tp, lead24_obs_zero_*, lead24_with_fuxi_tp_*,
          lead24_with_fuxi_tp_obs_zero_*, lead24_fuxi_tp_innov_*_bgtp_both,
          lead24_fuxi_tp_grid_weight_*_w1-0.1_bgtp
```

---

## 12. 2026-09-16 更新

### 12.1 做了什么：§11.8 的第 1、2 项都实现并跑完

**第 1 项 —— 观测一致性损失**

```
J = J_label + λ · mean_{有效站格点} | H(x_a) − obs' | / σ_o
```

* 新增 `main/model/ztd_torch.py`：可微 ZTD 算子的 torch 版（`StationZTD` 在 1378 个站格点上求值），
  物理常数和通道序**直接 import `preprocessing/ztd_operator.py`**，避免两套物理漂移。
* 新增 `main/model/build_optimizer.ObsConsistencyLoss`（MAE；先 `nan_to_num` 再作差，避免 `0×NaN` 的梯度）。
* `train_FSDP.py` 训练/评估循环里加这一项并**单独记录**：日志行
  `[obs-consistency: term=… |H(x)-obs|=… mm]`。**val-best 仍按 label loss 选**，所以 `best_val` 与历史可比。
* 旋钮 `configs.lambda_obs`（0 = 关）、`obs_sigma_o_mm = 11`（§11.5 量的算子+代表误差）。
  有效权重是 λ/σ_o，λ=0.2 时该项开局 ≈0.115，与 label loss（≈0.128）同量级。

**第 2 项 —— 逐站去静态偏差**

* `preprocessing/build_obs_debias.py` → `dataset/obs_debias_lead{lead}h.npz`：
  `b_s = mean_{训练期}( obs_mm − H(bg)_mm )`，1378 个数，只用 train split。
* 两处必须一致：innovation 从 `obs − H(bg)` 变成 `(obs − b_s) − H(bg)`；一致性损失的目标从 `obs`
  变成 `obs − b_s`。（在 `x_a = x_bg` 处两版算子数值完全相同，所以 `b_s` 不用重建。）
* **实测数字**：`b_s` 均值 **+8.27 mm**、|b| 均值 8.31、站间 std 4.05、范围 [−8.5, +56.4]、
  与站高相关 **+0.42**。**它主要是一个全球常数**（H(FuXi@24h) 整体比观测低 8.27 mm），
  去掉它只削掉 innovation 方差的 **5.8%**（std 11.61 → 10.94 mm）。
  ⚠ §9.4 说的"85 % 方差是站点静态偏移"针对的是**绝对 ZTD**；innovation 里这部分早已被 H(bg) 吸收。

**算子验证**（新增 `preprocessing/check_ztd_torch.py`）：同一份 FuXi 背景同时喂 numpy 版与 torch 版，
在 1378 格 × 6 个时刻上 **worst |差| = 0.0007 mm**（float32 舍入级）。脚本还带梯度检查：
`msl` 的敏感度 `|dH/dx|` = **27100**（全通道最大，是 r850 的 2 倍、r700 的 2.5 倍），
而 `z*/u*/v*` 共 41 个通道**梯度恒为 0**（算子只读 13 层 t、13 层 r、t2m、msl = 28 个通道）。

### 12.2 五次实验（test，1092 样本，纬度加权；"相对背景"负 = 比背景差）

| 实验（results 目录前缀） | λ | ZHD | epoch | best_val | 全格 69ch | 全格 70ch | 站内 69ch | 站内 70ch | tp | 站内优通道 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lead24_oc_debias` | 0.2 | 未冻 | 35 | 0.12891@35 | +0.004% | +0.101% | −0.513% | −0.393% | +3.56% | 40/69 |
| `lead24_oc_debias_smaller_oc` | 0.05 | 未冻 | 35 | — | −0.190% | −0.120% | −0.562% | −0.479% | +2.37% | 24/69 |
| `lead24_frzzhd` | 0.2 | **冻** | 35 | 0.12893@22 | −0.063% | +0.029% | −0.721% | −0.600% | +3.34% | 37/69 |
| `lead24_frzzhd_more_epoch` | 0.2 | **冻** | 40 | 0.12887@21 | +0.037% | +0.137% | −0.507% | −0.383% | +3.69% | 38/69 |
| ★ `lead24_oc_debias_more_epoch` | 0.2 | 未冻 | 40 | 0.12884@27 | **+0.096%** | **+0.196%** | **−0.245%** | **−0.127%** | **+3.77%** | **44/69** |

（完整的目录名 = `{前缀}_{model_id 的 exp_tag}`，例如
`lead24_oc_debias_more_epoch_lead24h_obs6h_era5tp_bgtp_both_oc0.2_debias`。）

参照：背景 69ch = 0.126074（全格）/ 0.127090（站内）；气候态 69ch = 1.036235、tp = 0.825201；
"more_epoch" 两轮用的是 `num_iteration=30000, num_epochs=40` → 实际 **22720 步 = 40 epoch**
（⚠ 和 35ep 那轮比，同时把 LR 调度从 `warmup1000/T_max19000` 拉成了 `warmup1500/T_max28500`，
所以"训练更久"和"退火更慢"是混在一起的，见 12.6）。

**网络确实开始用观测了**（对比 §4 那个恒等映射的失效模式）：

| | 旧最好（lead24+FuXi tp） | ★ 未冻 40ep |
| --- | --- | --- |
| val `\|H(x)−obs\|` | — | 9.22 → **2.56 mm** 收尾（best 那轮 2.76）；train 0.88 mm |
| 站内平均改动（69ch） | 0.0153 | **0.0241**（比 35ep 的 0.0260 更小，但收益更大） |
| 站内优于背景的通道 | 33/69 | **44/69** |
| tp（vs 训练目标 ERA5 tp） | 0.23705 | 0.23463 |

未冻 35ep 那轮还能看到背景的**干偏差被观测修掉**：r700 偏差 −0.0189 → −0.0005、r600 −0.0218 → −0.0021、
r850 −0.0174 → −0.0022（标准化单位）。这是同化系统该有的行为。

### 12.3 账本：唯一堵点是 msl

站内 69ch 的 ΔMAE（分析 − 背景；Δ>0 = 变差；站内背景总量 = 8.7702）：

| 实验 | Δ69 | msl | 其余 68 | r\* | t\* | z\* | u/v\* | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 未冻 35ep | +0.04500 | **+0.05732** | −0.01232 | −0.02858 | +0.02256 | +0.00390 | −0.01021 | −0.00940 |
| ★ 未冻 40ep | **+0.02149** | **+0.05426** | **−0.03277** | **−0.03620** | +0.01981 | −0.00161 | −0.01477 | −0.00997 |
| 冻 35ep | +0.06326 | −0.00021 | +0.06347 | +0.02787 | +0.04209 | +0.00355 | −0.01004 | −0.00893 |
| 冻 40ep | +0.04445 | −0.00107 | +0.04553 | +0.02126 | +0.03722 | −0.00271 | −0.01024 | −0.00978 |

★ 那轮逐通道站内改善：**r600 +2.26%、r700 +2.00%、r850 +2.00%、r1000 +1.56%、r500 +1.54%、
r925 +1.37%、z850 +2.49%、u850 +0.95%**；亏的只有 **msl −114.32%**、t850 −7.43%、t700 −4.15%、t2m −3.98%。

逐通道甚至更好：r600 +2.26（线性 DA +1.67）、r500 +1.54（+1.00）、r925 +1.37（+0.92）、
z850 +2.49（+0.61），只有 r700/r850 略低（+2.00 vs +2.31/+2.12）。

**为什么偏偏是 msl**：ZTD ≈ ZHD + ZWD，而 `ZHD = 2.2768 mm/hPa` **只由地面气压决定**，所以
`|dH/dx_msl| = 27100`（全通道最大杠杆），同时 msl 自己的背景误差又是**所有通道里最小的**
（站内 0.0475 std = 38 Pa）——"杠杆最大 + 底子最好"是最坏组合。损失对 69 个通道**等权**，
于是把 msl 推 Δ=0.091 std（74 Pa，约等于它自身误差的 2 倍）在损失里只值 0.054/69 ≈ 0.0008，
而它换来的 ZTD 拟合却值 0.01 量级。**网络理性地把 msl 卖了。**

### 12.4 冻 ZHD（第 3 条路）为什么不行

`obs_freeze_zhd=True`（新增旋钮，`configs.py`）把算子改成 `H*(x_a) = ZHD(x_bg) + ZWD(x_a)`：
静力项和整个柱几何（p_s、哪些层在地面以下、地表节点的 p_s）全部取自背景并 detach，
分析场只能动热力柱。设计目标 100% 达成——**msl 站内 Δ 从 +0.0573 变成 −0.0002**（改善 +2.26%）、
全格点从 −42.9% 回到 +0.89%。

但净结果更差：其余 68 通道从 **−0.0123（赚）翻成 +0.0635（亏）**，摆动 0.0758 > 省下的 0.0575。
原因很干净——两个版本拿到的 **ZTD 拟合量完全相同**（val `|H(x)−obs|` 尾值都是 2.47 mm），
但冻掉之后湿柱要独扛：r700 的改动量被从 0.136 逼到 **0.210（+55%）**，r 族由赚 0.036 变成亏 0.021。

即：**未冻时约 26% 的 ZTD 削减量走静力这条路**（站内 msl 平均动 0.77 hPa ≈ 1.75 mm ZHD，总削减 6.8 mm），
把它堵上就等于让湿柱多干 1/3 的活，而湿度改动只在某个幅度内是赚的。所以"冻结"是把一个 ±0.054 的
损失换成了 +0.077 的损失，不划算。

### 12.5 代码 / 产物清单（今天新增或改动）

| 文件 | 改动 |
| --- | --- |
| `main/model/ztd_torch.py` | **新增** 可微 ZTD 算子（`zhd_zwd_torch` / `StationZTD`，支持 `freeze_zhd`） |
| `main/model/build_optimizer.py` | **新增** `ObsConsistencyLoss`；`__all__` 更新 |
| `main/utils/utils.py` | **新增** `station_geometry` / `obs_debias_path` / `load_obs_debias` / `obs_debias_grid`；exp_tag 增加 `_oc{λ}` / `_frzzhd` / `_debias` |
| `main/utils/utils_data.py` | innovation 逐站减 `b_s`（`obs_debias`） |
| `train_FSDP.py` | 一致性项接入训练/评估，单独记录；summary/ckpt 记录 `lambda_obs`/`obs_sigma_o_mm`/`obs_debias` |
| `configs.py` | 新增 `lambda_obs` / `obs_sigma_o_mm` / `obs_debias` / `obs_debias_file` / `obs_freeze_zhd`；默认切到 lead24 + `fuxi_europe_0p25_24h_70ch.zarr` + `obs_mode='both'` + `include_fuxi_tp=True` |
| `plot_results.py` | **背景侧也按 70 通道统计**（第 69 通道 = FuXi tp），新增 `improve_pct_70ch` / `improve_pct_70ch_station` / `mae_bg_std_70ch` / `mae_bg_tp_vs_target_std` 等；**新增 `metrics_station.csv`**（逐通道站内口径）；修 `improve_pct` 原来是全局常数、修 loss_curve 里写死的 `bg_tr=0.0938`、修 `mae_modeltp_vs_*` / `mae_era5tp_vs_imerg_std` 漏除像素数 |
| `preprocessing/check_ztd_torch.py` | **新增** 算子数值 + 梯度校验（numpy vs torch、冻/未冻） |
| `preprocessing/build_obs_debias.py` | **新增** 生成 `dataset/obs_debias_lead{lead}h.npz` |
| `test/plot_channel_improvement.py` | **新增** 单 run 的逐通道 MAE 变化图 + csv（`channel_improvement.{png,csv}`） |

新增数据文件：`dataset/obs_debias_lead24h.npz`（1378 个 b_s、bias_grid、bias_std、station_id…）。

### 12.6 今天踩的坑（别重复踩）

1. **`num_epochs` 和 `num_iteration` 谁先到谁生效**。3 卡每 epoch 568 步（= 3403/(2×3)，1 卡是 1702），
   所以 20000 步 = 35.2 epoch：`num_epochs=60` 完全不起作用，只调它等于没调。只调 `num_epochs`
   把它调到小于 `num_iteration/568` 才是"按 epoch 截断"。
2. **截断时 LR 调度不会跟着变**：`warmup` 和 cosine `T_max` 都是按 `num_iteration` 算的。
   所以"多加 epoch"的实验会和"退火更慢"混在一起（今天两轮 40ep 就是 `num_iteration=30000`，
   实际停在 30000 步的 74.5% 处，结束时 lr ≈1.5e-5 没有退火完）。想干净地跑 N 个 epoch 就
   `num_iteration = N * 568` 且 `num_epochs = N`。
3. **`summary.json` 不记录 `num_epochs`**，只能靠日志里最后一个 `[Epoch N]` 判断是哪个上限生效的。
4. **这台机器没有任何中文字体**（`fc-list` 里 CJK = 0，matplotlib 只有 19 个 DejaVu）：
   中文标签会全变方块，出图必须用英文，或者先塞一个 Noto/文泉驿字体进去。
5. 评估（`plot_results.py`）可以随时重跑，是纯前向；但**不要在训练跑着的时候改 `utils_data.py`**
   （§11.7 第 2 条，收尾评估的 worker 会拿到新代码配旧对象）。
6. `plot_results.py` 的 `metrics.csv` / `metrics_station.csv` 会被重跑覆盖，`summary.json` / `.pth` / `.npy` 不会。

### 12.7 还没解决 / 需要注意的方法学问题

* **val-best 选择本身带不确定度**：五轮的 `best_val` 都在 0.1288~0.1289 这个第 4 位上打平，
  但站内指标能差 0.02（未冻 35ep +0.0450 vs 冻 35ep +0.0633，两者 val 只差 0.00002）。
  单次实验的"站内 ±0.01"不宜当作结论，重要判断要换 `rand_seed` 复跑 2 个看散布。
  （已有一个弱证据：`obs_zero` 那对复现差异只有 0.003。）
* λ=0.05 全面差于 λ=0.2（全格 70ch 由正转负、tp 从 +3.56% 掉到 +2.37%），说明**不是"观测用多了"**，
  而是"用在了不该用的通道上"。λ 不再是主要旋钮。

### 12.8 下一步（按优先级）

1. **把 msl 摁住**（唯一堵点，值 +0.054；修好即 ≈+0.37%，追平线性 DA）。两个实现，建议先做 (a)：
   * **(a) 硬约束**：模型 forward 里把 msl（通道 68）的残差置零，即分析场的 msl 恒等于背景。
     一行代码、零超参，预测站内 Δ ≈ −0.033 → **+0.37%**。风险是湿柱负担从 74% 升到约 80%
     （远小于全冻的 100%），r 族应能守住。
   * **(b) 软约束**：逐通道增量惩罚 `J += μ·mean_c |Δx_c| / σ_b,c`（对角 B 的 3D-Var），
     天然让"动 msl"变贵（它的 σ_b 只有 r700 的 1/6），μ 可调。
2. **把"更多 epoch"这个变量做干净**：固定 `num_iteration=20000` 只放开 `num_epochs`（等价于现状），
   或真正跑满 `num_iteration=30000, num_epochs>=53`；两种都和今天的 40ep 对比，才能分出是步数还是调度。
3. **（可选）oracle 实验**（§11.8 第 5 项）：用 `H(ERA5)` 当完美观测测架构上限，
   顺便给 `|H(x_a) − obs|` 一个"真值能到多少"的参照（现在 2.47 mm 已经低于算子的误差水平）。
4. **（可选）把 `plot_channel_improvement.py` 接进 `plot_results.py`**，让每次训练自动出逐通道图。

### 12.9 一句话总结今天

> 观测一致性损失 + 去静态偏差把网络从"恒等映射"救了出来：站内优于背景的通道 33→44/69，
> 现在的问题**只剩一个通道**：msl 被当成拟合 ZTD 的廉价杠杆（杠杆最大、底子最好、损失等权），
> 站内 −114%（占净亏的 127%）。冻 ZHD 能救 msl 但会把湿柱逼过头（r 族由赚 0.036 变亏 0.021），
> 所以正确做法是**只摁 msl**，不是冻整个静力项。

---

## 13. 2026-09-17 更新

### 13.1 做了什么：把 label loss 限制在站点附近（halo）

动机：8222 个非站格点没有观测约束，怀疑那里的回归只会把网络拉回背景。做法是把 label loss 的
**空间域**从全格缩到"站点格 + 半径 3 格的 halo"。

* `main/utils/utils.py`：新增 `loss_region_mask(cfg)`（站点 mask 的 Chebyshev 膨胀）和
  `loss_domain_weight(cfg)`（膨胀后的区域取 `loss_station_weight`，区域外取 `loss_nostation_weight`）；
  `exp_tag` 增加 `_halo{N}` 后缀。
* `configs.py`：`loss_domain = 'station_halo'`（**现在是默认值**，见 §13.5）、
  `loss_halo_cells = 3`、`loss_nostation_weight = 0.0`（区域外**没有任何损失**）。
* 启动时会打印 `[Loss] domain=station_halo halo=3: region 5778/9600 cells (60.2%), weights in=1 out=0`。
* `plot_results.py`：新增 `metrics_region.csv`；并且当 `loss_domain != 'full'` 时 `region_limited=True`，
  **主指标和所有图都被限制到 region 的 5778 格**（否则图里会混进没有梯度的格子）。

区域量级：5778/9600 = **60.2%** 的格点；按 cos(lat) 权重算，区域占全格权重的 **58.7%**，其中站点格
只占区域权重的 **23.9%**（halo 里的 4400 个非站格点占 76.1%）。

### 13.2 两轮实验（只差 λ，其余全同）

lead24h + obs6h + `include_fuxi_tp` + `obs_mode='both'` + `obs_debias` + halo3 + w1-0、
`num_iteration=25000`、warmup1250 / cosine `T_max=23750`（**这次是完整跑满并退火到 0**）、
3 卡 / seed 2000。产物在 `main_code/work_dir/results/stage_two/`。

| 实验（model_id） | λ | best val | best 位置 |
| --- | --- | --- | --- |
| `lead24h_halo3` | 0.2 | 0.13112 | epoch 21（iter 11928） |
| `lead24h_halo3_lambda_obs_0.1` | 0.1 | 0.13168 | epoch 10（iter 5680） |
| 参照：background-only val（同区域口径） | — | 0.13120 | — |

### 13.3 结果：两轮都明显差于 §12 的全域最好那轮

test，1092 样本，**cos(lat) 加权**；"相对背景"负 = 比背景差；站内 = 1378 个有站格点。

| 口径 | S1 best（§12，全域 loss，λ0.2，40ep） | halo3 λ0.2 | halo3 λ0.1 |
| --- | --- | --- | --- |
| 站内 69ch | −0.245% | **−0.423%** | **−0.915%** |
| 站内 70ch（含 tp） | −0.127% | −0.312% | −0.821% |
| halo 格点 69ch（4400 个非站格点） | — | +0.050% | −0.354% |
| 区域 69ch（5778 格） | —（无此口径） | −0.063% | −0.487% |
| 区域 70ch | — | +0.035% | −0.405% |
| **站内去 msl（68 状态通道）** | **+0.375%** | **+0.132%** | **−0.420%** |
| 站内去 msl（halo / 区域） | — | +0.294% / +0.256% | −0.134% / −0.202% |

逐族站内（ΔMAE 相对该族，负 = 改善）：

| 族 | S1 best 全域 | halo3 λ0.2 | halo3 λ0.1 |
| --- | --- | --- | --- |
| r\*（13） | +1.19% | +0.92% | +0.39% |
| t\*（13） | −1.36% | −1.55% | −2.59% |
| z\*（13） | +0.40% | −0.46% | −3.63% |
| u\*（14） | +0.40% | +0.19% | +0.06% |
| v\*（14） | +0.33% | +0.19% | −0.12% |
| msl | −114.32% | −102.34% | −91.82% |
| t2m | −3.98% | −4.26% | −3.54% |
| era5_tp | +3.52% | +3.12% | +2.08% |

几个单通道（站内，物理单位，S1 / λ0.2 / λ0.1）：r700 7.762→7.607（+2.00%）/ 7.682（+1.03%）/
7.775（−0.16%）；r850 6.718→6.584（+2.00%）/ 6.638（+1.19%）/ 6.703（+0.22%）；
r600 8.010→7.829（+2.26%）/ 7.832（+2.22%）/ 7.905（+1.31%）。站内平均改动量 mean|Δx|（69ch）：
0.0241（S1）→ 0.0271 / 0.0280——**改动更大、收益更小**。

### 13.4 四条结论

1. **msl 一点没被治好，依旧是全部净亏**。站内 msl MAE 38.43 → 77.76 Pa（−102%）/ 73.71 Pa（−92%）；
   它单通道 Δ = +0.0486 / +0.0436，而站内 69ch 净 Δ 只有 +0.0371 / +0.0803，
   msl 占净亏的 **131% / 54%**。改成 halo 只让它少亏一点（S1 是 +0.0543），代价是其余通道的收益一起被砍。
2. **真正的信息提取量掉了约 3 倍**：去掉 msl 后站内 68 通道从 **+0.375%** 掉到 **+0.132%**（λ0.2），
   λ0.1 直接变负。r700/r850 收益腰斩，z\* 由赚变亏。
3. **"非站格点只会把网络拉回背景"这个前提和数据不符**。把 §12 那轮（全域 loss）拆开：站内每通道 MAE
   **+0.000312**、非站格点 **−0.000193（−0.15%）**——非站格点在改善。这轮 halo 区里的 4400 个非站格点
   也是改善的（69ch +0.050%、去 msl +0.294%），**反而是站点格点自己被拖到 −0.42%**。
   被丢掉的那 3822 格（39.8%）不是噪声，是学习信号。
4. **有效 λ 被悄悄稀释了**。label loss 按 `sum(weight)` 归一化，分母从 9600 格变成 5778 格
   （cos-lat 权重下区域占 58.7%），每个区域格点的 label 权重涨了 **1.70 倍**，于是观测一致性项
   相对弱了 0.587 倍 → 这两轮等效于全域名义的 **λ≈0.117 和 λ≈0.059**，都是已知会变差的方向（§12.7），
   而且和两轮好坏顺序一致。所以"halo 更差"里有多少来自区域限制、多少来自 λ 稀释，现在分不开。

另外两个观察：

* **val 曲线很难看**。λ0.2 那轮 best 之后一路涨到 0.1323（+0.9%）；λ0.1 那轮的 best 停在 epoch 10
  就再没动过、收尾 0.1347（+2.3%）。而 background-only val = 0.13120——**best 那个点只比背景好 0.06%**，
  label 层面基本还是恒等映射，唯一的真收益仍来自 tp 和 obs 项。
* 两轮都真的在拟合观测（best 处 val `|H(x)−obs|` = 2.67 / 2.80 mm，train 1.90 / 2.43 mm，收尾 train
  0.80 mm），和 §12 的 2.76 mm 同级。所以不是"观测没被用上"，而是**用上之后仍然亏**。

### 13.5 口径陷阱（重要，别踩）

1. **`configs.py` 里 `loss_domain` 的默认值已经改成 `'station_halo'`**。`plot_results.py` 的
   `region_limited` 判定看的是 cfg 而不是 run，所以**拿旧的"全域 loss" checkpoint 重跑评估时，如果不显式
   `--set loss_domain=full`，指标和所有图都会被限制到 5778 格的区域口径**，`metrics.csv` 会被区域数字
   覆盖（`metrics_station.csv` 不受影响，仍是站内口径）。
2. `configs.py` 的 `results_dir` 现在指向 `.../results/stage_two`；评估 stage_one 的老 run 要
   `--set results_dir=.../results/stage_one`。
3. **这两轮没有全域数字**（设计如此），所以它们和 §12 表格里的"全格"列不可比，能比的只有站内列。
   想补全格数字（纯前向；**别忘 `--out`，否则会覆盖 run 目录里的图和 metrics.csv**）：

```bash
cd da_ngl/main_code
CUDA_VISIBLE_DEVICES=0 python plot_results.py --split test --set loss_domain=full \
  --model_id lead24h_halo3 --exp_tag lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_oc0.2_debias \
  --out work_dir/results/stage_two/eval_fullgrid_halo3_oc0.2
```

4. `metrics_region.csv` 在 `loss_domain='full'` 时会退化成"站点格"表（区域 mask = 站点 mask），
   别拿它当"全格"用。

### 13.6 下一步

1. **回到 `loss_domain='full'`**（顺手把 configs 的默认值改回去）。halo 这条路按现在的实现不值得继续：
   两轮没有任何一处比 §12 那轮好，msl 也照旧。
2. 正题仍是 §12.8 第 1 项：**只摁 msl**（forward 里把通道 68 的残差置零）。两轮结果又一次确认
3. 如果还想验证"只关心站点附近"，要做得公平至少三件事：
   * λ 按 1/0.587 放大到 **~0.34**（保持 obs:label 的相对强度），或者干脆承认这是"更弱观测权重"的实验；
   * `num_iteration` 与对照轮对齐，别把"区域限制"和"退火更慢"混在一起；
   * 用 `loss_nostation_weight=0.1~0.3` 的**降权**代替 `0` 的删除，保住那 3822 格的学习信号；
     顺便扫 halo=1/2（halo=3 已覆盖 60% 面积，"限制"其实很温和）。
4. val-best 的噪声问题（§12.7）在这两轮更明显：λ0.1 的 best 停在 epoch 10，之后 35 个 epoch 再没进步。
   重要判断前先换个 `rand_seed` 复跑。

### 13.7 一句话总结今天

> 把 label loss 限制在"站点 + halo3"（区域外权重 0）是**负收益**：站内 69ch 由 −0.245% 掉到
> −0.423%（λ0.2）/ −0.915%（λ0.1），去掉 msl 后由 **+0.375% 掉到 +0.132%**；msl 照旧是全部净亏
> （−102%）。两个原因：① label loss 的分母 9600→5778 格，等效 λ 被稀释到 0.117 / 0.059；
> ② 被丢掉的那 39.8% 格点其实是有学习信号的（全域那轮非站格点改善了 +0.15%）。
> **回到全域 loss，专心做"只摁 msl"。**

---

## 14. 2026-09-18 更新（口径变更 + obs/H(FuXi) 统一归一化）

### 14.1 三条长期约定（今天定）

1. **只用 `loss_domain='station_halo'`**（`loss_halo_cells=3`、`loss_nostation_weight=0.0`），
   不再回 `'full'`。目标是**这个区域**（5778/9600 格）比背景好，而不是全格点。
   §13 那两轮是负收益，其中"λ 被稀释"的那部分已由 §14.2 修掉。
2. **统计一律看 70 通道**（ERA5 0–68 + tp），不再看 69 通道；并且要**逐通道**说清楚
   哪些提升、哪些下降。
3. **obs 和 `ztd_fuxi` 用同一套归一化标准**——都用 NGL 训练集的 `ztd_train_mean/std`，见 §14.3。

### 14.2 λ 域补偿：把 §13.4 item 4 的稀释修回来

label loss 是加权**平均**，缩到 halo 之后每个保留下来的格点权重涨了 1/0.587 = 1.70 倍，
于是观测一致性项被相对稀释 0.587 倍（§13.4 第 4 条），所以那两轮名义 λ=0.2 实际只相当于
全域口径的 0.117（λ=0.1 那轮 ≈0.059）。

* `main/utils/utils.py`：新增 `loss_domain_weight_share(cfg)`，返回 label loss 保留的
  cos(lat) 权重占比（halo3 = **0.587**；`loss_domain='full'` 且权重 1/1 时 = 1.0）。
* `configs.lambda_obs_domain_compensation = True`（默认开）：`train_FSDP.build_obs_loss()` 里
  `lambda_eff = lambda_obs / share`，λ=0.2 → **0.341**。日志打印
  `[ObsLoss] lambda=0.2 -> effective 0.3407 (domain share 0.587)`。
* exp_tag 加 `_occ`；summary/ckpt 记录 `lambda_obs_effective` / `lambda_obs_domain_share`。
* ⚠ **`_occ` 之前和之后的 halo 轮次不能直接比**：老的等效 λ≈0.117/0.059，新的是 0.341。

### 14.3 obs 与 H(FuXi) 统一归一化

原来：obs 通道是 `(obs_mm − μ)/σ`（σ=119.76 mm），而创新通道是 `(obs_mm − H(FuXi)_mm)/15 mm`
——两个观测通道各用一套尺度，网络看到的"绝对观测"和"增量"不可比。

现在（`configs.obs_res_scale_mm = None`，默认）：

```
ztd_norm   = (obs_mm     − μ_ngl) / σ_ngl     # NGL store 里本来就是它
fuxi_norm  = (H(FuXi)_mm − μ_ngl) / σ_ngl     # 新增
innovation = ztd_norm − fuxi_norm             # = (obs_mm − H(FuXi)_mm) / σ_ngl
```

* μ = `ztd_train_mean` = 2333.6450 mm、σ = `ztd_train_std` = 119.7633 mm
  （NGL **训练集**全局统计，存在 NGL zarr 里）。
* `obs_res_scale_mm` 写正数 = 保留旧行为（旧的 15.0）；exp_tag 加 `_obsstd`。
* **副作用（已实测，不是 bug）**：创新通道 std 从 ~1.04 掉到 ~0.13
  （≈ 10.9 mm / 119.76 mm），比绝对通道（std≈1）小一个量级。这正是"同一套标准"的直接后果。
  出图都按 mm 用），归一化只在 dataset 里做，**不需要重建 store**。
* 观测一致性损失不受影响：它本来就在 mm 空间比 `H(x_a)` 与 `obs_mm`（σ_o = 11 mm）。

### 14.4 新增可选项 `freeze_msl`（§12.8 第 1 项的实现，默认关）

`main/model/assimilation.py`：`freeze_msl=True` 时 forward 把通道 68（msl）的分析值换回背景值，
网络再也不能用地面气压去买 ZTD 拟合；exp_tag 加 `_frmsl`。
和 `obs_freeze_zhd` 的区别：后者把整个静力项冻掉（湿柱独扛 100%），这里只堵住 msl 这一个出口
（湿柱扛 ~80%）——§12.4 的教训是前者不划算。

### 14.5 70 通道口径（`plot_results.py`）

* 新 headline：**70 通道**，region 在前、station 在后；终端打印 improved / worse 名单（各前 8）。
* `channels_70ch.csv`：逐通道 `improve_pct_region` / `improve_pct_station` + 两边 MAE，按 region 改善排序。
* `summary.json` 新增 `headline_metric` / `n_channels_70ch` / `channels_improved_70ch_{region,station}` /
  `channels_worse_70ch_{region,station}` / `channel_ranking_70ch_{region,station}`。
* `channel_metrics.png` 改成 70 通道。
* 69 通道字段保留（历史连续性），但**不再作为结论口径**。

### 14.6 代码改动清单

| 文件 | 改动 |
| --- | --- |
| `configs.py` | `obs_res_scale_mm=None`（统一归一化）、`lambda_obs_domain_compensation=True`、`freeze_msl=False`、loss_domain 长期策略注释 |
| `main/utils/utils_data.py` | `__getitem__` 统一归一化的 innovation；启动打印归一化方式 |
| `main/utils/utils.py` | 新增 `loss_domain_weight_share()`；exp_tag 新增 `_obsstd` / `_occ` / `_frmsl` |
| `main/model/assimilation.py` | `freeze_msl` 参数 + forward 里通道 68 的硬约束 |
| `train_FSDP.py` | λ 补偿接入 `build_obs_loss()`、`freeze_msl` 传入模型、日志/summary/ckpt 记录新字段 |
| `plot_results.py` | 70 通道 headline + 逐通道排名 CSV + 70 通道图；`freeze_msl` 传入模型；summary 记录 λ_eff / obs_res_scale_mm |
| `main/utils/__init__.py` | 导出 `loss_domain_weight_share` |

### 14.7 验证（2026-09-18，只做了离线检查，还没跑训练）

* `py_compile` 全过（configs / train_FSDP / plot_results / utils / utils_data / assimilation）。
* 数值：`innovation` 与 `(obs_mm − H(FuXi))/σ` 在 5 个样本 × 8 万有效格点上最大差
  **4.4e-7**（float32 舍入级）。
* 通道尺度：绝对 1.0021，统一创新 **0.1304**，旧创新 1.0409（比值 0.125）。
* `freeze_msl=True`：`max|out[68] − bg[68]| = 0.0`，相邻通道照常变（2.70）；关掉时是 2.02。
* exp_tag 实测：`lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias`
  （再开 freeze_msl 会多一个 `_frmsl`）；`loss_domain_weight_share = 0.587`、有效 λ = 0.3407。

### 14.8 下一步

1. 跑新一版 halo 实验（统一归一化 + λ 补偿，单卡 25000 步 ≈ 15 epoch）：
   `stage3_obsstd`（不冻 msl）与 `stage3_obsstd_frmsl`（冻 msl）A/B 对比，看
   ①统一归一化本身值不值、②msl 硬约束能不能把 region / station 的 70 通道拉正。
2. ⚠ 单卡 25000 步只有 ~15 epoch（3 卡同样步数是 44 epoch），和 §13 那两轮**不是同一 epoch 数**，
   跨机器比较要留这个口子（§12.6）。
3. 若 A/B 都还是负的，回到 §12.7：val-best 的噪声，换 `rand_seed` 复跑再看。
