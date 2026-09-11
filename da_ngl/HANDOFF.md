# 交接笔记 — GNSS ZTD + FuXi → ERA5 同化

> 最后更新：2026-09-10（当天工作结束时）
>
> **明天怎么继续**：直接说 ——「读 `da_ngl/HANDOFF.md`，继续」即可。

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
* **标签 71 通道**：0–68 ERA5、69 = IMERG tp（训练用）、**70 = `era5_tp`
  （仅评估用，训练不读）**。训练读多少由 `configs.label_n_chans = 70` 控制。
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
CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=22336 nohup bash train.sh > ../logs/fuxi_da.log 2>&1 &

# 评估出图
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

> 读 `da_ngl/HANDOFF.md`，继续做同化。今天已经建好三个 zarr（NGL/FuXi/label）并完成一次
> 3 卡训练（结果：分析 0.08412 vs 背景 0.08379，基本没超过背景场）。我想先做第 5 节里的第 N 项。
