# da_ngl 全链路数据处理说明（按数据流顺序）

> 生成时间：2026-09-18。描述的是**当前代码**（`configs.py` 默认值 = lead 24 h / obs 6 h /
> `station_halo` / 统一归一化 `_obsstd` / `_occ` / obs-debias / `include_fuxi_tp`）。
> 数字都来自代码、store 元数据或 `HANDOFF.md` 记录过的实测值；标注「口径陷阱」的地方
> 是可以直接踩的坑。
>
> 配套流程图源码：[`data_pipeline.dot`](data_pipeline.dot)。本机没装 graphviz 的 `dot`，
> 想看图片要 `dot -Tpng data_pipeline.dot -o data_pipeline.png`（在有 graphviz 的机器上）。
> 旧的 `pipeline_flowchart.{md,dot,png}` 是 2026-09-15 的版本，**不含**观测一致性损失、去偏、
> halo、统一归一化这些东西，别当现状用。

---

## 0. 一页速览

```
                        ┌────────────────────────── 离线建库（preprocessing/, 一次性） ──────────────────────────┐

 NGL *.trop.zip ──┐                                                   ┌──────────┐
 (5min TROTOT)    ├─►[1] 站点→格点映射 (test/map_stations_to_grid.py)  │ ETOPO2   │(仅校验用)
 站点元数据        │   → 80x120 grid map parquet (1378 站 / 8222 mask)│ 高程      │
                  │                                                   └──────────┘
                  ├─►[2] build_ngl_zarr.py ──► ngl_europe_0p25_5min.zarr  ztd[394272,80,120]  std=(x-μ)/σ
                  │       · 解析 +TROP/SOLUTION，只留 300s 整倍数
                  │       · μ/σ 用 train 段 (2022-01-01..2024-05-01) 实测
                  │
 FuXi (3 个源) ───┼─►[4] build_fuxi_zarr.py ─► fuxi_europe_0p25_24h_70ch.zarr  z[5484,1,70,80,120]
                  │       · init/step 保留，lead 24h，+FuXi tp 作 ch69
                  │
 ERA5 6h ─────────┼─►[3] build_label_zarr.py ─► label_europe_0p25.zarr  label[5476,71,80,120]
 IMERG tp ────────┘       · ch0..68 ERA5(已标准化) / ch69 IMERG tp / ch70 ERA5 tp(仅评估)
                          · tp: clip0 → log1p → z-score (mean[69]=0.2754, std[69]=0.4103)

              [5] build_ztd_fuxi_zarr.py   FuXi 背景 → H(FuXi)  ztd_fuxi/zhd/zwd [5476,80,120] mm
              [6] build_obs_debias.py      b_s = mean_train(obs_mm - H(bg)_mm) → obs_debias_lead24h.npz

                        └──────────────────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────────────── 在线训练/评估（main_code/） ──────────────────────────┐
 三个 zarr + b_s ──► [7] AssimilationDataset (main/utils/utils_data.py)
                          · 每个样本 T: bg=FuXi(init=T-lead), obs=73 帧×5min 窗(到 T 为止), label=ERA5@T
                          · obs_mode='both' → 通道0 绝对 ZTD(norm), 通道1 innovation = ztd_norm - H(FuXi)_norm
                          · innovation 里已减 b_s（obs_debias）
                    ──► [8] process_bg / process_obs（设备侧）
                          · obs 追加 mask(1) + lat(1) + lon(1) → (B,73,5,80,120)
                          · bg (B,1,70,80,120)（含 FuXi tp）
                    ──► [9] AssimilationNetv6 (main/model/assimilation.py)
                          · 80x120 → replicate pad 到 80x128 → 三块 encoder/decoder + fusion
                          · out = decoder(...) + bg（残差只加在前 bg_chans 个通道）
                          · 再叠加两层 EnhanceStack 残差；freeze_msl 时可把 ch68 换回背景
                    ──► [10] loss = 纬度加权 MAE（halo 域内，权重 1/0）
                              + λ_eff · MAE_station|H(x_a) - (obs - b_s)| / σ_o
                                λ_eff = λ / 0.587 = 0.3407, σ_o = 11 mm
                    ──► [11] FSDP(fp16) + AdamW + warmup/cosine；val-best 存 {model_id}_{arch_tag}.pth
                    ──► [12] plot_results.py → metrics.csv / metrics_station.csv / metrics_region.csv /
                              channels_70ch.csv / metrics.json / summary.json + 6 类图
                        └────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 全局约定

### 1.1 目标网格

| 项 | 值 | 来源 |
| --- | --- | --- |
| 分辨率 | 0.25° | `preprocessing/common.py: RES` |
| 纬度 | 36.50 … 56.25，**80** 点，升序 | `LAT_MIN/LAT_MAX/N_LAT` |
| 经度 | −5.25 … 24.50，**120** 点，升序 | `LON_MIN/LON_MAX/N_LON` |
| 格点总数 | 9600 = 80×120 | |
| 有站格点 | **1378**（每格一站） | `ngl_europe_0p25_80x120_station_grid_map.parquet` |
| 无站格点 | **8222**（`mask == True`） | 同上 |

源场（FuXi / ERA5 / IMERG）是**全球 720×1440**、经度 0…360 的规则经纬网格，裁剪逻辑在
`common.Region`：

* 纬度：源轴自北向南（降序），目标轴升序 → 取 `la0:la1+1` 后**翻转**（`flip_lat`）；
* 经度：源经度按 0…360 表示，目标经度取模后最近邻 `argmin|lon_src - (lon_tgt % 360)|`；
* 采样方式是**最近邻**，不是插值。
* 注意源纬度并非严格 0.25° 等分（720 点覆盖 ±90°，间距 0.250347°），纬度格心最大偏差
  **0.075°**；经度完全对齐。`Region.max_lat_err / max_lon_err` 记了这两个量，但没有断言。

### 1.2 时间轴与划分

三个 store 都用 CF 风格（int64 数值 + `units` 字符串），但**基准时刻和单位各不相同**：

| store / 轴 | 单位 | 基准 | 长度 |
| --- | --- | --- | --- |
| NGL `time` | **minutes** | 2022-01-01 00:00 UTC | 394272（5 分钟） |
| FuXi `init` | hours | 2022-01-01 00:00 UTC | 5484（24h 版；含两端 pad） |
| FuXi `step` | hours | — | `[24]`（当前）/ `[6]`（旧 6h 版） |
| label `time` | hours | 2022-01-01 00:00 UTC | 5476（6 小时） |
| `ztd_fuxi` `time` | hours | **1970-01-01 00:00 UTC** | 5476 |

⚠️ **NGL 的时间单位是 minutes，不是 hours**。手工脚本里写 `unit='h'` 会让 `searchsorted`
出来的帧索引静默错位（HANDOFF §11.7 踩过）。一律用 `common.decode_time_axis` /
`utils_data.decode_axis`。

覆盖范围与划分（半开区间 `[start, end)`，**只写在 attrs 里，物理上不切分**）：

| split | 区间 | 样本数（lead 24h + `--init-pad-hours 24`） |
| --- | --- | --- |
| train | 2022-01-01 … 2024-05-01 | 3403 |
| val | 2024-05-01 … 2025-01-01 | 980 |
| test | 2025-01-01 … 2025-10-01 | 1092 |

上限取 2025-10-01 是因为 IMERG 的 tp 序列到此为止。三个 split 的样本数与 lead 6h 版**逐样本对齐**
（`build_fuxi_zarr.py --init-pad-hours 24` 的作用：lead 24h 时 `init = T − 24h`，两端各补 4 个起报，
否则每个 split 两端会各少 4 个样本）。

### 1.3 通道约定（69 / 70 / 71）

`preprocessing/common.py` 里三份通道表：

```python
SOURCE_CHANNELS = [z50..z1000(13), t50..t1000(13), u50..u1000(13),
                   v50..v1000(13), r50..r1000(13), t2m, u10, v10, msl, tp]   # 70 个
CHANNELS              = SOURCE_CHANNELS[:69]                 # 0..68，状态量（去掉 tp）
TRAIN_LABEL_CHANNELS  = CHANNELS + ["tp"]                    # 70，训练目标
LABEL_CHANNELS        = TRAIN_LABEL_CHANNELS + ["era5_tp"]   # 71，store 实际布局
```

逐通道索引（0 基）：

| 通道 | 内容 |
| --- | --- |
| 0–12 | z50, z100, z150, z200, z250, z300, z400, z500, z600, z700, z850, z925, z1000 |
| 13–25 | t50 … t1000（13 层） |
| 26–38 | u50 … u1000（13 层） |
| 39–51 | v50 … v1000（13 层） |
| 52–64 | r50 … r1000（13 层） |
| 65 | t2m |
| 66 | u10 |
| 67 | v10 |
| 68 | msl（单位 **Pa**） |
| 69 | label store 里 = IMERG tp；训练目标里 = `tp_label_source` 指定的那个 tp |
| 70 | label store 里 = ERA5 自己的 tp，**只用于评估/画图，不进训练** |

三种「通道数」在不同地方出现，别混：

| 名字 | 值 | 含义 |
| --- | --- | --- |
| `model_bg_chans` / `bg_chans(cfg)` | 69 或 **70** | 背景场通道数；`include_fuxi_tp=True` 时 70（FuXi tp 作 ch69） |
| `label_n_chans` | **70** | 训练时从 label store 取的通道数（0..68 + 选中的 tp） |
| `model_out_chans` | **70** | 模型输出通道数 |
| store 的 `label` 第 1 维 | **71** | 0..68 ERA5 + 69 IMERG tp + 70 ERA5 tp |

`AssimilationDataset._select_label()` 负责 71 → 70：请求 71 通道（评估）时原样返回；
否则取 0..68 再把 `tp_label_source` 指定的通道（`'imerg'` → 69，`'era5'` → 70）放到第 69 位。
当前 `tp_label_source = 'era5'`。

### 1.4 标准化常量

| 量 | 值 | 存在哪 |
| --- | --- | --- |
| NGL ZTD 训练集均值 μ | **2333.6450 mm** | NGL store 的 `ztd_train_mean` |
| NGL ZTD 训练集标准差 σ | **119.7633 mm** | NGL store 的 `ztd_train_std` |
| ERA5/FuXi 各通道 mean/std | `mean_era5.npy` / `std_era5.npy`（各 70 个） | `database/fuxi-obs/obs-grid_qc/mean_std/` |
| tp 通道 mean[69] / std[69] | 0.2754 / 0.4103 | 同上 |

标准化定义：

```
ERA5/FuXi 状态通道（0..68）：  x_std = (x_phys - mean_era5[c]) / std_era5[c]   ← 源 store 里已经是这个
NGL ZTD：                      ztd_norm = (ztd_mm - μ_ngl) / σ_ngl             ← 建库时算好写进去
tp（ERA5 源）：                x_mm = tp_m × 1000 → log1p → (x - 0.2754)/0.4103
tp（IMERG）：                  x_mm = tp_mm（已 mm，不再 ×1000）→ log1p → (x - 0.2754)/0.4103
```

**2026-09-18 起（`obs_res_scale_mm = None`）obs 与 H(FuXi) 共用上面这一套 μ/σ**：

```
ztd_norm   = (obs_mm     - μ_ngl) / σ_ngl                           # 观测通道 0
fuxi_norm  = (H(FuXi)_mm - μ_ngl) / σ_ngl                           # 只在 dataset 内部算
innovation = ztd_norm - fuxi_norm = (obs_mm - H(FuXi)_mm) / σ_ngl   # 观测通道 1
```

旧口径（`obs_res_scale_mm = 15`）是 `innovation = (obs_mm - H(FuXi)_mm) / 15`。
通道尺度实测：绝对通道 std≈1.0021，统一创新 std≈0.1304，旧创新 std≈1.0409
（比值正好 15/119.7633 = 0.125）。`exp_tag` 用 `_obsstd` 区分新旧。

### 1.5 目录与产物清单

```
da_ngl/
├── dataset/
│   ├── ngl_europe_0p25_5min.zarr              3.7 G   ztd[394272,80,120]
│   ├── fuxi_europe_0p25_24h_70ch.zarr         4.6 G   z[5484,1,70,80,120]   ← 当前 config 用这个
│   ├── fuxi_europe_0p25_24h.zarr              4.6 G   同上但 69 通道
│   ├── fuxi_europe_0p25.zarr                  4.6 G   lead 6h / 69 通道（历史）
│   ├── label_europe_0p25.zarr                 4.8 G   label[5476,71,80,120]
│   ├── ztd_fuxi_europe_0p25_24h.zarr          156 M   ztd_fuxi/zhd/zwd[5476,80,120] + 站点几何
│   ├── ztd_fuxi_europe_0p25_6h.zarr           156 M   lead 6h 版
│   ├── obs_debias_lead24h.npz                 28 K    b_s(1378) + bias_grid + station_id …
│   ├── obs_debias_lead6h.npz                  28 K    同上（6h）
│   ├── ngl_europe_0p25_80x120_station_grid_map.parquet   站点→格点映射（在用的）
│   ├── ngl_europe_stations.parquet                       站点经纬度 + height_m
│   └── _cache/ngl_5min_series.npz                        NGL 解析结果缓存（可加速重建）
├── preprocessing/     建库与算子（§2–§8）
├── main_code/         训练与评估（§9–§14）
├── test/              校验脚本、基线、画图、本文档
├── logs/              训练日志 {model_id}_{arch_tag}.log
└── plots/             零散出图
```

Zarr 写盘的统一约定（`common.finalize_store`）：

* 数组用 `Blosc(cname='zstd', clevel=3, shuffle=BITSHUFFLE)`，唯一例外是
  `ztd_fuxi` 的 `time`（历史遗留：`lz4 / clevel=5 / shuffle=1`）；
* 数据数组 `fill_value = NaN`；坐标/掩膜数组的 `fill_value` 被**手工清成 null**；
* 最后 `zarr.consolidate_metadata`，`xr.open_zarr` 才能走快路径。

为什么必须清 `fill_value`：zarr 默认值是 `0/False/''`，与真实值冲突
（`lon == 0.0`、`time == 0`、`mask == False`），xarray 会把 `_FillValue` 当缺失，
把这些值变成 NaN/NaT。**新增 store 一定要调用 `finalize_store`。**

---

## 2. Step 0：源数据

| 数据 | 路径 | 形式 | 用到的字段 |
| --- | --- | --- | --- |
| NGL GNSS ZTD | `/cpfs01/.../public/ngl_ztd_all_downloaded/data/top10_2022_2026/Western_and_central_Europe/` | `{year}/{station}.{year}.trop.zip`（zip 里再套 gzip 文本） | `TROTOT` [mm] |
| 站点元数据 | `da_ngl/dataset/ngl_europe_stations.parquet` | parquet | `gnss_station_id`, `lat`, `lon`, `height_m` |
| FuXi 预报 | `/cpfs01/.../public/huangyuanqing/data/` 下 `Fuxi_pred_2017_2024`、`FuXi_Pred_2024_2025`、`FuXi_Pred_2025_other` | zarr | `z[time, step, channel, lat, lon]`，前 69 通道与 `CHANNELS` 一致 |
| ERA5 | `/cpfs01/.../public/huangyuanqing/data/ERA5_2017_2025` | zarr | `z[time, channel, lat, lon]`，70 通道（ch69 = 已标准化的 tp） |
| IMERG | `/cpfs01/.../public/database/fuxi-obs/imerg/zarr_25_720_more` | zarr | `data[time, channel, lat, lon]`，取 `channel == 'tp'` |
| ERA5/FuXi 统计量 | `/cpfs01/.../public/database/fuxi-obs/obs-grid_qc/mean_std/{mean_era5,std_era5}.npy` | npy (70,) | 全通道 mean/std |
| ETOPO2 高程 | `/cpfs01/.../public/xuxiaoze/shape/ETOPO2v2c_f4.nc` | netCDF | **只用于算子校验**（`check_ztd_operator.py --height etopo`），建库不用 |

---

## 3. Step 1：站点 → 格点映射

脚本：`test/map_stations_to_grid.py`（⚠️ README 里写的 `preprocessing/map_stations_to_grid.py`
已不成立，文件在 `test/`；且脚本默认参数还是旧的 0.05° 版本）。

生成在用映射的调用：

```bash
python test/map_stations_to_grid.py \
  --station-info dataset/ngl_europe_stations.parquet \
  --data-root  /cpfs01/.../Western_and_central_Europe \
  --lat-min 36.50 --lat-max 56.25 --lon-min -5.25 --lon-max 24.50 --res 0.25 \
  --seed 2021 \
  --output-map dataset/ngl_europe_0p25_80x120_station_grid_map.parquet
```

处理细节：

1. `discover_stations()` 用正则 `^(?P<station>[^./\\]+)\.\d{4}\.trop\.zip$` 递归扫描归档目录，
   得到**实际有数据**的站点集合；
2. 与 `ngl_europe_stations.parquet` 取交集（缺坐标的打印 warning 后丢弃），按 ID 去重；
3. 每站取最近格心：`idx = floor((coord - coord_min)/res + 0.5)`，越界的丢掉；
4. 同一格有多站 → `np.random.default_rng(seed=2021).choice(sorted(ids))` 随机保留一个
   （先排序再抽样，结果**可复现**）；
5. 输出 long-form parquet，每行一个格点：

| 列 | 含义 |
| --- | --- |
| `lat`, `lon` | 格心坐标（严格 0.25° 整数倍） |
| `n_stations` | 该格落进来的站点数（0 = 无站） |
| `mask` | `n_stations == 0`，**True = 无站** |
| `station_id` | 选中的站点号；无站格为 `None` |

结果：9600 行 = **1378 站格 + 8222 无站格**。`mask` 极性（True = 无站）后续到处在用，
`utils.station_cell_mask()` 会拿 store 里的 `station` 数组反查校验，极性错了直接抛异常。

（`test/plot_stations_on_map.py`、`test/plot_ngl_grid_stations.py` 是配套可视化；
`dataset/ngl_europe_0p25_station_grid_map.parquet` 是早期版本，别用。）

---

## 4. Step 2：NGL 观测库 `ngl_europe_0p25_5min.zarr`

脚本：`preprocessing/build_ngl_zarr.py`。

### 4.1 解析原始归档

每个 `{station}.{year}.trop.zip` 里可能有多个成员，逐个当 gzip 文本流读：

```
+TROP/SOLUTION        ← 开始标记
* ...                 ← 以 * 开头的是注释，跳过
 <epoch> <TROTOT> ... ← 数据行
-TROP/SOLUTION        ← 结束标记（break）
```

* 只取 `fields[1]`（epoch）与 `fields[2]`（`TROTOT`，mm）；`len(fields) < 5` 的行丢弃；
* `fields[1]` 形如 `YY DDD:SSSSS`，解析见 `_parse_epoch`：
  `year += 2000 if year < 80 else 1900`，然后 `year-01-01 + (DDD−1) 天 + SSSSS 秒`；
* **只保留秒数能被 300 整除的记录**（`seconds % 300 != 0 → skip`），即严格对齐 UTC 5 分钟整；
* 落在 `[TIME_START, TIME_END)` 之外的点丢弃；索引 `step = (time - start) / 5min`；
* 同一时刻重复出现时**后写覆盖**（不做平均、不做 QC）；
* zip/gzip 读取整体包在 `try/except` 里，坏文件静默跳过，只累计成功记录数。

并行：`ProcessPoolExecutor`，每站一个任务、chunksize=1，默认 `min(64, cpu_count)`。
给了 `--cache` 时把每站序列存成 npz，**下次直接读缓存**；缓存长度不足直接报错退出，
不会静默截断。

### 4.2 装配与统计

* 时间轴：`pd.date_range(2022-01-01, 2025-10-01, freq='5min', inclusive='left')` → **394272** 步
  （1369 天 × 288）；
* 网格：`np.full((394272,80,120), NaN)`，把每站序列写进它那一格，其余保持 NaN；
* 训练统计量：只取 train 段 `[2022-01-01, 2024-05-01)` 的有限值（跨全部格点与时刻）算
  `train_mean/std` → **2333.6450 / 119.7633 mm**；
* 存盘前整体做 `(ztd - μ) / σ`（**存的是标准化值**，不是 mm）。

### 4.3 写盘

| 数组 | shape | chunk | dtype | fill_value | 备注 |
| --- | --- | --- | --- | --- | --- |
| `time` | 394272 | 4096 | int64 | null | `units="minutes since 2022-01-01 00:00:00"`, `timezone=UTC` |
| `lat` / `lon` | 80 / 120 | 全量 | float64 | null | degrees_north / degrees_east |
| `station` | (80,120) | 全量 | `<U{maxlen}` | 清空 | 站点号字符串，无站为 `""` |
| `mask` | (80,120) | 全量 | bool | null | `description="True where no GNSS station is present"` |
| `ztd` | (394272,80,120) | **(288,80,120)** | f4 | NaN | `raw_units=mm`，`standardisation="(ztd_mm - ztd_train_mean)/ztd_train_std"` |
| `ztd_train_mean` / `ztd_train_std` | (1,) | 1 | f4 | | units mm |

group attrs 另记 `source`、`grid`、`time_sampling`、`n_stations`、`masked_cells`、`normalisation`、
`splits`。最后 `finalize_store()`。文件大小 3.7 G。

---

## 5. Step 3：标签库 `label_europe_0p25.zarr`

脚本：`preprocessing/build_label_zarr.py`。

### 5.1 时间对齐

* 目标轴：`pd.date_range(2022-01-01, 2025-10-01, freq='6h', inclusive='left')` → **5476** 步；
* ERA5 与 IMERG 的时间轴各自建成 `time → 下标` 的 `pd.Series`，用 `reindex(times)` 映射，
  缺的填 **−1**（`read_one` 遇到 −1 就跳过，该通道保持 NaN）；
* stdout 打印 `[src] ERA5 missing N, IMERG missing M`。

### 5.2 逐时刻读取

`read_one(i, e_i, m_i, …)` 每个时刻产出一块 `(71,80,120)`：

* **ERA5（通道 0..68 + ERA5 tp）**：`region.extract(group["z"][e_i, :70])` 最近邻裁到目标网格。
  源 ERA5 的 0..68 已经标准化，**直接拷贝**；源第 69 通道（tp）拷到目标 **ch70**
  （`era5_tp`，只评估用）；
* **IMERG tp（通道 69）**：按 `channel` 数组找 `'tp'`，`region.extract` 后做 `_tp_transform`：

```
standardized  （默认，本库用的）：x = log1p(clip(tp_mm, 0));  x_std = (x - 0.2754) / 0.4103
standardized_m                ：同上但先把输入 ×1000（把"按米给的"输入折成 mm）
raw                           ：直接存 mm
```

* `ProcessPoolExecutor` + `initializer=_init_worker`（把变换模式和 mean/std 塞进每个进程，
  避免反复读 npy）；`as_completed` 收结果按时刻下标写回，**写回顺序与并发无关**。

### 5.3 写盘

| 数组 | shape | chunk | dtype | fill_value |
| --- | --- | --- | --- | --- |
| `time` | 5476 | 4096 | int64 | null（`units="hours since 2022-01-01 00:00:00"`） |
| `channel` | 71 | 71 | `<U{maxlen}` | `LABEL_CHANNELS` |
| `lat` / `lon` | 80 / 120 | 全量 | float64 | |
| `label` | (5476,71,80,120) | **(6,71,80,120)** | f4 | NaN |

group attrs：`era5_source`、`imerg_source`、`channels`、`era5_normalisation`、`tp_transform`、
`train_channels`(70)、`eval_only_channels=["era5_tp"]`、`splits`。文件大小 4.8 G。

---

## 6. Step 4：背景场库 `fuxi_europe_0p25_24h_70ch.zarr`

脚本：`preprocessing/build_fuxi_zarr.py`。当前 config 用 `--leads 24 --with-tp`。

### 6.1 时间轴拼接

* 三个源各自 decode `time` 轴 → 拼表 → **按起报时间排序 + `drop_duplicates(keep='first')`**：
  同一 init 在多源都有时取源列表里靠前的那个；
* `--init-pad-hours N` 把窗口从 `[TIME_START, TIME_END)` 扩成
  `[TIME_START−N h, TIME_END+N h)`。lead 24h 时取 N=24 → **5484** 个起报（5476 + 两端各 4）；
  不加 pad 每个 split 两端都会少样本，与 lead 6h 的实验不能逐样本对齐；
* `--leads` 逗号分隔（如 `--leads 6,12,24`），默认 `6`；当前只存 `[24]`。

### 6.2 通道

* 打开每个源时**校验** `channel[0:69] == CHANNELS`，不一致直接退出；
* `--with-tp` 时额外要求源的第 69 个通道名是 `'tp'`，并存成输出 ch69（共 70 通道）：
  FuXi 自己的 tp 已与标签 tp 在同一 log1p 标准化空间，可直接当背景；
  ⚠️ 写 `channel` 数组时 dtype 要能容纳名字（历史上 `era5_tp` 7 字符被 `<U5` 截断过）；
* 所有源的 `step` 轴必须一致，否则报错。

### 6.3 写盘

| 数组 | shape | chunk | dtype | fill_value |
| --- | --- | --- | --- | --- |
| `init` | 5484 | 4096 | int64 | null（`hours since 2022-01-01`） |
| `step` | 1 | 1 | int64 | null（`units="hours"`，值 `[24]`） |
| `channel` | 70 | 70 | `<U…` | |
| `lat` / `lon` | 80 / 120 | 全量 | float64 | |
| `z` | (5484,1,70,80,120) | **(1,1,70,80,120)** | f4 | NaN |

文件大小 4.6 G。历史版本 `fuxi_europe_0p25.zarr`（lead 6h / 69 通道）仍在磁盘上，6h 对照实验用它。

---

## 7. Step 5：H(FuXi) 算子库 `ztd_fuxi_europe_0p25_24h.zarr`

脚本：`preprocessing/build_ztd_fuxi_zarr.py`；物理实现：`preprocessing/ztd_operator.py`。

### 7.1 物理

```
ZHD = 0.0022768 · p_s / (1 - 0.00266·cos2φ - 0.00028·h_km)          [m]
ZWD = 1e-6 · (R_d/g) · ∫ [ k2'·e/p + k3·e/(T·p) ] dp                 [m]
      k2' = 16.52 K/hPa, k3 = 3.776e5 K²/hPa, R_d = 287.05, g = 9.80665
e   = RH/100 · es(T),  es = 6.1078 · exp(17.269·Tc/(Tc+237.3))       [hPa, Tetens over water]
ZTD = ZHD + ZWD
```

单位（极易搞错，代码里写死）：`T` 用 K，`p/msl` 用 hPa（**store 里的 msl 是 Pa，这里除以 100**），
`RH` 用 %，`h` 用 m，`lat` 用度，输出 m（`*_mm` 变体输出 mm）。
湿度默认按**水面**饱和（`over_ice=False`），与 ERA5 的 `r` 定义一致。

### 7.2 三个关键设计

1. **底层塌缩**：ERA5 层只到 1000 hPa，而站点平均地面气压约 975 hPa，
   即 1000 hPa 对多数站**在地下**。做法是把所有 `p >= p_s` 的层压到 `p_s`（厚度 0、值换成地面值），
   再在底部**追加地表节点**：`T = t2m`，`e_sfc = RH_low/100 · es(t2m)`，
   其中 `RH_low` 是**最低一个仍在地面之上的层**的 RH（store 没有 2 m 湿度，这是唯一替代）。
   这样模型的地下外推不会漏进柱积分。
2. **地面气压**：`p_s = msl/100 · exp(-g·h/(R_d·t2m))`（hypsometric，忽略湿度）。
   高度用 **GNSS 站点元数据高度**，不用 ETOPO：实测站点高度给总 RMSE 14.3 mm，
   ETOPO2 给 80 mm（100 m 高度误差≈12 hPa≈27 mm ZHD）。
3. **只读取 28 个通道**：13 层 t、13 层 r、t2m、msl；z/u/v 共 41 个通道梯度恒为 0。

### 7.3 时间对齐与写盘

* `init = T − lead`（lead = `fcst_step × 6h`），在 FuXi `init` 轴上 `searchsorted` 定位，
  要求**逐点精确命中**（`assert n_valid >= lab_t.size - 1`）；基准轴是 label 轴；
* 分块并行：默认 `--block 32` 时刻一块、`--workers 16`；每块取
  `gf["z"].oindex[bg_idx, lead_idx, 28 通道]`，反标准化回物理量后算 ZHD/ZWD/ZTD；
* `--set KEY=VALUE` 可直接覆盖 `fcst_step` / `fuxi_zarr`，**跑 24h 实验必须同时覆盖
  `ztd_fuxi_zarr`**，否则会静默用 6h 的 H(FuXi)（HANDOFF §11.1 的坑）。

| 数组 | shape | chunk | 单位 | 说明 |
| --- | --- | --- | --- | --- |
| `zhd` / `zwd` / `ztd_fuxi` | (5476,80,120) | (64,80,120) | mm | 站格外 NaN |
| `time` | 5476 | 4096 | hours since 1970-01-01 | |
| `lat` / `lon` | 80 / 120 | 全量 | 度 | float32 |
| `mask` / `station` | (80,120) | 全量 | | 从 NGL store 拷来的站点几何，让 store 自描述 |
| `height_m` | (1378,) | 1378 | m | 逐站高度，顺序与 `station_id` 一致 |

group attrs 记了算子、背景约定、层表、高度来源、公式与校验结果。文件大小 156 M。

### 7.4 校验数字

`preprocessing/check_ztd_operator.py`（numpy 版 vs NGL 实测）：

* 总 ZTD：RMSE **14.3 mm**，`r = 0.9959`；
* 站点距平（各站去掉自身时间均值）：RMSE **12.1 mm**，解释观测距平方差的 **93%**；
* 该脚本还能对比 `profile` / `+surface` / `surface` 三种形式，以及 `--height station|etopo|both`。

`preprocessing/check_ztd_torch.py`（torch 版 vs numpy 建好的 store）：worst |差|
**0.0007 mm**（float32 舍入级），并做梯度检查 —— plain 模式 `msl` 敏感度 **27100**（全通道最大，
是 r850 的 2 倍、r700 的 2.5 倍），`z*/u*/v*` 共 41 个通道**恒为 0**；
`freeze_zhd` 模式在 `x_a = x_bg` 处与 plain **完全相等**，且 `msl` 敏感度精确为 0。

---

## 8. Step 6：逐站去偏 `obs_debias_lead24h.npz`

脚本：`preprocessing/build_obs_debias.py`。

动机：绝对 ZTD 里约 85% 的方差是站点静态量（高度误差、局地气候、算子偏差）。
让网络先学 1378 个常数偏移是浪费容量。定义（**只用 train split**）：

```
b_s = mean_{T ∈ train} ( obs_mm(T, s) − H(bg)_mm(T, s) )
```

实现细节：

* 直接把 `cfg.obs_debias` 关掉后 `AssimilationDataset(cfg, dates_train_range)` 建起来，
  用它的 `samples` **枚举同一套 (label, bg, obs) 三元组**，保证时间对齐与训练完全一致；
* 观测取窗口最后一帧（`obs_end = samples[:,2] + obs_frames − 1`，即有效时刻 T），
  读出来是标准化值，用 `× σ_ngl + μ_ngl` 变回 mm；`H(bg)` 直接读 `ztd_fuxi[lb_i]`；
* 分批（`read_frames(..., block=512)`）读取，控制内存；
* `bias = nanmean(obs − H(bg))`；有效样本数为 0 的站格填 0 并打 warning；
* 同时输出 `bias_std_mm`（该站差值的**时间标准差**）。

产物字段：`bias_mm (1378)`、`bias_grid (80,120，站格外 0)`、`bias_std_mm (1378)`、
`station_id (1378)`、`n_samples`、`lead_hours`、`fcst_step`、`ztd_fuxi_zarr`、`fuxi_zarr`、`split`、
`obs_frames`。

实测（lead 24h）：`b_s` 均值 **+8.27 mm**、`|b|` 均值 8.31、站间 std 4.05、范围
**[−8.5, +56.4]**、与站高相关 **+0.42**；主要是"H(FuXi) 整体比观测低 8.27 mm"这样的**全局常数**，
只削掉创新方差的 **5.8%**（std 11.61 → 10.94 mm）。

⚠️ `b_s` 依赖 lead（`H(bg)` 依赖 lead），所以**一个 lead 一个文件**；
`load_obs_debias()` 会拿 `station_id` 与当前 grid map 逐元素比对，顺序不符直接抛异常。

---

## 9. Step 7：Dataset 组装（`main/utils/utils_data.py`）

### 9.1 时间轴解码与自检

`decode_axis(store, name)` 读 `.zattrs` 里的 `units` 字符串
（形如 `"5 minutes since 2022-01-01 00:00:00"`），按 `minutes / hours / days` 前缀决定单位，
再展开成 `pd.DatetimeIndex`。`_assert_regular()` 断言相邻差值恒定 —— 一旦有缺口就抛错，
因为**后续所有索引都是算术下标**（`(t − t0) / step`），有缺口会静默错位。

### 9.2 样本枚举

对每个 6 小时时刻 `T`（按 label 轴步长在 split 区间内生成）做三重存在性检查：

```
init      = T − fcst_step×6h         必须在 FuXi init 轴上    → bg_i
T                                    必须在 label 轴上         → lb_i
obs_end   = T + obs_end_offset_minutes
obs_start = obs_end − (obs_frames−1) × 5min
                                     两端必须在 NGL 轴上       → obs_i
```

通过的存进 `self.samples`，元素是**下标三元组 `(lb_i, bg_i, obs_i)`**，不是数据本身
（数据在 `__getitem__` 里按需读）。当前参数：`fcst_step=4`（lead 24h）、`obs_frames=73`、
`obs_end_offset_minutes=0` → 观测窗 `[T−6h, T]`，正好覆盖整个 FuXi 预报时段。

初始化结束会打印：

```
[Dataset] bg: init = T - 24h, step = 24h (fcst_step=4, zarr step axis [24])
[Dataset] background channels = 70 (FuXi tp included)
[Dataset] ZTD standardisation: obs and H(FuXi) share the NGL train mean/std (unified=True)
[Dataset] innovation de-biased per station: 1378 cells, mean |b| = 8.31 mm
[Dataset] label channels used = 70
[Dataset] tp training target = era5 (store channel 70)
[Dataset] 2025-01-01 00:00:00 ~ 2025-09-30 18:00:00  wanted=1092 usable=1092 (...)
```

### 9.3 每个样本的张量（`__getitem__`）

三个 store 的句柄**按进程懒加载**（`_store()` 检查 pid），因为 DataLoader worker 是
`forkserver` 启的，不能在主进程里长期持有句柄。

| 输出 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `bg` | `(1, bg_chans, 80, 120)` | f32 | `z[bg_i, lead_index, :bg_chans]`；`lead_index` 由 `step` 轴查 `lead_hours` 得到（不是硬编码 0） |
| `obs` | `(F, C, 80, 120)` | f32 | `F = obs_frames`；`C = 2`（`both`）/ `1`（`absolute`、`residual`） |
| `label` | `(70, 80, 120)` | f32 | `_select_label()` 之后的 70 通道 |

观测通道的构造（`obs_mode='both'`）：

```
ztd     = NGL store 的 ztd[obs_i : obs_i+F]        ← 已经是 (obs_mm − μ)/σ
fuxi_mm = ztd_fuxi[lb_i]                           ← 物理量，mm
if obs_debias:  fuxi_mm += bias_grid               ← 等价于把 b_s 减到观测侧

if obs_res_scale_mm > 0:      # 旧口径
    innovation = (ztd×σ + μ − fuxi_mm) / obs_res_scale_mm
else:                         # 2026-09-18 起
    fuxi_norm  = (fuxi_mm − μ) / σ
    innovation = ztd − fuxi_norm

obs = concat([ztd[:,None], innovation[:,None]], dim=1)     # 'both'
```

`ztd` 在无站格点是 NaN，`fuxi_mm` 在站格外也是 NaN，所以创新通道天然只在站格有值。
**去偏加在 `fuxi_mm` 上**，等效于 `(obs − b_s) − H(bg)`；顺序是"先去偏、再归一化"。

### 9.4 DataLoader

`build_dataloader()`：单进程用 `SubsetRandomSampler`（train）或 `SequentialSampler`（val/test），
多卡用 `DistributedSampler`。默认 `batch_size=2`、`num_workers=8`、`prefetch_factor=3`、
`persistent_workers=True`、`multiprocessing_context='forkserver'`、`pin_memory=False`。

⚠️ **不要边训练边改 `utils_data.py`**：train/val 的 worker 在启动时就 fork 好了，
test 的 worker 到收尾评估才第一次 fork，会拿到"新代码 + 旧对象"，报
`AttributeError: 'AssimilationDataset' object has no attribute 'bg_n_chans'`，
而 `train.sh` 带 `set -e`，整条链会直接终止（HANDOFF §11.7）。

---

## 10. Step 8：设备侧预处理（`train_FSDP.py`）

### 10.1 `process_bg(data, rank, hw)`

搬到 GPU、转 float；若 `cfg.grid_hw` 与数据空间尺寸不同则双线性插值（当前都是 (80,120)，不触发）。

### 10.2 `process_obs(obs, cfg, rank)`

```
finite = isfinite(obs)                      # 无站格点 / 缺记录 = False
data   = nan_to_num(obs)                    # NaN → 0
mask   = finite.any(dim=2, keepdim=True)    # (B,T,1,H,W)：该帧该格点是否有有效观测

if cfg.zero_obs:  data = 0                  # 「无 GNSS」消融：先算 mask，再整体清零

extra = [mask]                              # obs_add_mask=True
      + [lat 广播到 (B,T,1,H,W)]            # obs_add_latlon=True
      + [lon 广播到 (B,T,1,H,W)]
data = cat([data, *extra], dim=2)
return data * mask                          # 非站格点上所有通道清零
```

当前 `obs_mode='both'` → 每帧 **5** 个通道：`[绝对 ZTD, innovation, mask, lat, lon]`，
整体 `(B, 73, 5, 80, 120)`。

`zero_obs=True` 时**只清 ZTD 数值，mask/lat/lon 保留**：网络仍能看见"哪里有站、何时有观测"，
但看不到观测值。训练和 `plot_results.py`（import 同一个函数）因此行为一致。

### 10.3 `obs_at_valid_time(obs_raw, cfg, rank)`

一致性损失的观测目标：取窗口**最后一帧**（`[:, -1, 0]`）的标准化值，用
`× ztd_train_std + ztd_train_mean` 变回 **mm**，NaN 保留（损失里再 mask）。

⚠️ **口径陷阱**：这里假定通道 0 是"绝对标准化 ZTD"，即 `obs_mode ∈ {absolute, both}`。
若用 `obs_mode='residual'`（通道 0 = innovation）再开 `lambda_obs > 0`，
会把创新按 119.76 mm 的 σ 还原，得到没有物理意义的 mm 值。
**要跑观测一致性损失就用 `obs_mode='both'`（或 `absolute`）。**

### 10.4 AMP

`cfg.amp=True` 时 `bg / label / obs` 都 `.half()`；FSDP 的 `MixedPrecision` 用
param fp16、reduce fp32、buffer fp32，配 `ShardedGradScaler`。

---

## 11. Step 9：模型（`main/model/assimilation.py`）

### 11.1 输入/输出形状（当前配置）

| 量 | 值 | 说明 |
| --- | --- | --- |
| `bg_chans` | 70 | ERA5 69 状态 + FuXi tp |
| `obs_chans` | 5 | 2 观测通道 + mask + lat + lon |
| `obs_frames` | 73 | 6 小时 5 分钟窗 |
| `out_chans` | 70 | 69 状态 + 1 tp |
| `embed_dim` | 128 | |
| `depth` | (2,2,2) | 三个 block 里 FussionStack 的层数 |
| `pad_multiple` | 16 | |

`forward(bg (B,1,70,80,120), obs (B,73,5,80,120))`：

1. 展平时间维后 replicate pad：`80×120 → 80×128`（80 能被 16 整除、120 不能），输出再裁回 120；
2. `bg → (B,70,H,W)`，`obs → (B,365,H,W)`，`side_info = cat([bg, obs]) → (B,435,H,W)`；
3. 三个 `AssimilationBlockv2`：
   * block0：三个 `EncoderBlock`（stride-2）→ 40×64，通道 128；
   * block1：同上 → 20×32，通道 256；
   * block2：三个 `DecoderBlock`（`PixelShuffle(2)`）→ 40×64，通道 128；
   每个 block 内部是 `FussionStackv2`（`depth` 层 `FussionNetv2`）：
   `h = cat([bg, obs, side]) → 两次下采样 → 三条各自的 up 支路 + 跳连`，
   即 bg/obs/side 三路各自更新（bg 支路是残差 `bg = bg + out_bg`）；
4. `bg2 = cat([bg0, bg2], 1)` → `decoder`（`DecoderBlock`）→ `(B,70,80,128)`；
5. **残差只加在有背景的通道上**：
   `out = out + F.pad(bg_cp, (0,0,0,0,0,out_chans−bg_chans))`。
   当前 `out_chans == bg_chans == 70`，pad 为 0（tp 的残差直接叠在 FuXi tp 上）；
   历史 `include_fuxi_tp=False` 时 bg=69，tp 通道补 0 背景，即 tp 从零学；
6. 两层 `EnhanceStack` 残差：`out = enhance1(out) + out; out = enhance2(out) + out`；
7. `freeze_msl=True` 时把通道 68 换成背景值
   （`out = cat([out[:,:68], bg_cp[:,68:69], out[:,69:]], 1)`），分析场 msl 恒等于背景；
8. `rearrange → (B,1,70,H,W)`，裁回 80×120。

### 11.2 参数量与 padding 约束

* 默认 `embed_dim=128 / depth=(2,2,2)` → 约 **73.7 M** 参数（HANDOFF §11.4）；
* 每个 block 有两次 stride-2 下采样 ⇒ **H、W 必须能被 16 整除**；80 满足，120 需要 pad 到 128；
* `ln_norm` 是 channel 维 LayerNorm（`(n,c,h,w) → (n,h,w,c)` 归一化后再换回）。

---

## 12. Step 10：损失（`main/model/build_optimizer.py` + `train_FSDP.build_obs_loss`）

### 12.1 label 项：纬度加权 MAE

```
w_lat(h)    = cos(lat_h) / mean_h(cos(lat))          # (1,1,80,1)
w_cell(h,w) = loss_station_weight   if (h,w) ∈ region
              loss_nostation_weight  otherwise        # (1,1,80,120)
mask        = isfinite(label)

err  = |out − nan_to_num(label)|      ← 先 nan_to_num 再作差（NaN×0 = NaN 的坑）
loss = Σ(err · w_lat · w_cell · mask) / Σ(w_lat · w_cell · mask)
```

区域由 `utils.loss_region_mask(cfg)` 给出：

* `loss_domain='full'` → 区域 = **站点格本身**（要让"全格加权"生效需配合
  `loss_station_weight / loss_nostation_weight`；两者都是 1 时等价于历史无加权全格 loss）；
* `loss_domain='station_halo'`（当前）→ 站点 mask 按 `loss_halo_cells=3` 做**方形（Chebyshev）膨胀**，
  区域 = 5778/9600 格 = 60.2%（cos-lat 权重下占 **58.7%**）；
* 当前权重 `(1.0, 0.0)`：区域内权重 1，区域外**完全没有梯度**。

`set_cell_weight()` 在权重全为 1 时会丢掉 buffer，保证与历史无加权实现**逐位一致**。

### 12.2 观测一致性项

```
J_o   = λ_eff · mean_{有效站格点} | H(x_analysis) − (obs_mm − b_s) | / σ_o
λ_eff = λ_obs / share          # share = loss_domain_weight_share(cfg) = 0.587（halo3）
σ_o   = obs_sigma_o_mm = 11 mm # = 实测的算子+代表性误差 std(obs − H(ERA5))
```

* `H` = `main/model/ztd_torch.StationZTD`，对分析场前 69 通道求值，只在 1378 个站格点上算
  （站外 NaN 自动被 mask 掉）；
* 名义 λ=0.2，补偿域稀释后 **λ_eff = 0.3407**，再除以 11 mm。日志：
  `[ObsLoss] lambda=0.2 -> effective 0.3407 (domain share 0.587); sigma_o=11 mm
  -> 0.03097 per mm; debias=True stations=1378`；
* `loss_domain_weight_share()` 的算法：把 cell weight 乘上 cos(lat) 权重后，区域内占比。
  halo3 = 0.587；`loss_domain='full'` 且权重 1/1 时 = 1.0；
* 这一项**不参与 val-best 的选择**（`evaluate()` 的返回值不含它），只单独记录
  `[obs-consistency: term=… |H(xa)−obs|=… mm]`；
* `obs_freeze_zhd=True` 时算子变成 `H*(x_a) = ZHD(x_bg) + ZWD(x_a)`（静力项 detach 自背景）。

### 12.3 总损失

```
loss = J_label + (J_o if lambda_obs > 0 else 0)
```

反传前 `clip_grad_norm_(model.parameters(), max_norm=32)`。

---

## 13. Step 11：训练循环（`train_FSDP.py`）

### 13.1 初始化

1. `init_dist`：CLI 覆盖写进 config 模块 → 按 `obs_mode` / `include_fuxi_tp` 重算
   `model_obs_chans` / `model_bg_chans`（`configs.py` 里这两个值是 import 时算的，`--set` 后必须重算）
   → 安装 cell weight、打 `[Loss] domain=…` 行 → 建 run 目录与 logger；
2. logger：文件 handler 收 INFO，**控制台 handler 只到 WARNING**
   （避免 `nohup … > xx.log` 再抄一份大日志）；
3. `dist.init_process_group(nccl)`，`seed = rand_seed + rank`（当前 2000）；
4. 建 `obs_loss`（λ≤0 时 None）、三个 DataLoader、模型、FSDP、优化器与调度。

### 13.2 优化器与调度

```python
opt_type='AdamW', start_lr=1e-8 → stop_lr=1e-4, weight_decay=0.1
warmup=True, warmup_rate=0.05
warmup_steps = int(num_iteration × 0.05) = 1250
T_max        = num_iteration − warmup_steps = 23750
scheduler    = CosineAnnealingLR(T_max=23750)
```

每步：`if cfg.warmup: cfg.warmup = warmup_scheduler()`（线性升到 1e-4）；
升满后 `cfg.warmup` 变 False，改为 `scheduler.step()`（cosine 退火到 0）。

⚠️ **warmup 和 `T_max` 都是按 `num_iteration` 算的**，所以"改 `num_epochs` 多跑几轮"
会同时改变退火曲线，两个变量混在一起（HANDOFF §12.6）。

### 13.3 一个 epoch / 一次评估

* `train_one_epoch`：`zero_grad` → forward → `loss_fn + oc` → scaler 反传 →
  `clip_grad_norm_` → `scaler.step/update` → 记 lr → 每 `log_interval=100` 步由 rank 0 打一行
  `[iter N/25000] loss=… lr=… …s/it` → `if cfg.iteration >= cfg.num_iteration: break`；
  结束时 `dist.all_reduce` 求 epoch 平均（label 项与一致性项分别 all_reduce）；
* `evaluate`：no_grad 前向，返回 label loss（不含一致性项）；`get_fcst_loss=True` 时直接把背景场
  当预测（`batch_fcst[..., :label_chans]`）。

### 13.4 val-best 与存档

* 每个 epoch 后跑 val，`loss_val < best_val − min_delta`（默认 0）才保存，
  因此 run 目录里**只有一个** `{model_id}_{arch_tag}.pth`（val-best）；
* 保存走 `FSDP.state_dict_type(FULL_STATE_DICT, rank0_only=True)` + `FSDP.optim_state_dict`
  （旧写法 `summon_full_params + module.state_dict()` 在 SHARD_GRAD_OP 下存的是分片权重，已废弃）；
* payload 记录 `model_id / exp_tag / arch_tag / zero_obs / include_fuxi_tp / obs_mode /
  loss_*_weight / lambda_obs / lambda_obs_effective / lambda_obs_domain_share / obs_sigma_o_mm /
  obs_debias / obs_res_scale_mm / freeze_msl`；
* 训完**重新 load 回 best checkpoint** 再做最终评估。

### 13.5 收尾产物

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 超参 + `best_val_loss / best_iteration / best_epoch` + 逐 epoch `train_loss / val_loss` + `background_only_train/val` + test 的 model/background loss + checkpoint 路径 |
| `train_loss.npy` / `val_loss.npy` / `lr.npy` | 逐 epoch 曲线 / 逐步学习率 |
| `iter_loss.npy` | 只在 `log_batch_loss=True` 时产生 |
| `{model_id}_{arch_tag}.pth` | val-best 全量权重 |

日志路径 `da_ngl/logs/{model_id}_{arch_tag}.log`，run 目录
`{results_dir}/{model_id}_{exp_tag}/`。

⚠️ 收尾那句 `[Test] model=… background=…` **两边口径可能不同**
（`include_fuxi_tp=False` 时 model 70 通道、background 69 通道），不要直接比；
可比口径在 `plot_results.py`。

---

## 14. Step 12：评估与出图（`plot_results.py`）

### 14.1 数据与真值

* Dataset 用 `n_label_chans=71`（要拿到两个 tp）；
* `truth = cat([full[:,:69], (era5_tp if tp_label_source=='era5' else imerg_tp)])` → `(B,70,H,W)`，
  与模型输出同口径；
* 背景侧也按 70 通道统计：`bg69 = bg[:,0,:69]`，第 69 通道取 FuXi tp（`include_fuxi_tp=True` 时），
  否则填 NaN（对应通道指标直接是 NaN，而不是悄悄算成 0）。

### 14.2 三种口径

| 口径 | mask | 出现在 |
| --- | --- | --- |
| 全格 | 所有 9600 格（`loss_domain='full'` 时）或**训练区域**（region-limited） | `metrics.csv` |
| 站内 | `station_cell_mask(cfg)` = 1378 格 | `metrics_station.csv` |
| 区域 | `loss_region_mask(cfg)` = halo3 的 5778 格 | `metrics_region.csv` |

所有权重都是 `w = cos(lat) 归一化 × plot_domain`，分母 `Σ(w · valid)`，
所以**每个数都是纬度加权均值**。

⚠️ **口径陷阱**：`region_limited = (loss_domain != 'full')`。当前默认 `station_halo`，于是

* `metrics.csv` 里其实是**训练区域**的数字（列名却没有 `_region`）；拿旧 checkpoint 重跑评估时
  必须显式 `--set loss_domain=full`，否则指标被换成区域口径；
* `metrics_region.csv` 在 `loss_domain='full'` 时会退化成"站点格"表
  （`loss_region_mask('full')` 返回站点 mask），**不能当全格用**；
* `maps_*.png` / `timeseries*.png` / `tp_maps.png` 在区域外涂灰并置 NaN。

### 14.3 输出文件

| 文件 | 内容 |
| --- | --- |
| `metrics.csv` | 逐通道 `mae_analysis_std / mae_bg_std / rmse_* / bias_* / std_era5 / mae_climatology_std / mean_abs_analysis_minus_bg / mae_*_phys / improve_pct` |
| `metrics_station.csv` | 站内口径，同上加 `_station` 后缀 |
| `metrics_region.csv` | 区域口径，同上加 `_region` 后缀 |
| `channels_70ch.csv` | 按 **region 改善**排序的 70 通道表：`improve_pct_region / improve_pct_station / mae_{bg,analysis}_{region,station}` |
| `metrics.json` | headline 汇总（见 14.4） |
| `loss_curve.png` | train/val 曲线 + 背景参考线（从 `summary.json` 读，不再写死）+ lr |
| `channel_metrics.png` | 70 通道 bg/analysis MAE 柱状 + 改善率 |
| `maps_{z500,t850,r700,t2m,msl}.png` | 4 联图：ERA5 真值 / FuXi 背景 / 分析 / 分析−背景（取该 split 的**第一个样本**，时间戳按样本取，不是 store 开头） |
| `timeseries.png` | 6 个通道 × 2 列：域平均时间序列 + 逐时刻域平均 \|误差\| |
| `timeseries_tp.png` | tp 域平均与 \|误差\|（ERA5 tp / IMERG tp / 分析 / 两产品差值 / 气候态） |
| `tp_maps.png` | IMERG tp / ERA5 tp / 分析 tp / 分析−目标 / IMERG−ERA5 |

`plot_results.py --split {train,val,test}` 决定评估集合；`--checkpoint` 可显式指定权重；
`--out` 指定输出目录（默认写进 run 目录）。

### 14.4 headline（2026-09-18 起）

```
"headline_metric": "improve_pct_70ch_region"（区域口径）或 "improve_pct_70ch"（全格）
"n_channels_70ch" / "channels_improved_70ch_{region,station}" / "channels_worse_70ch_{...}"
"channel_ranking_70ch_{region,station}"          ← 逐通道改善排序
"improve_pct_70ch_region" / "improve_pct_70ch_station" / "improve_pct_70ch"
"mae_{analysis,bg}_std_{70ch,69ch}{,_region,_station}"
"n_channels_better*" / "mean_abs_analysis_minus_bg*"
"mae_modeltp_vs_era5tp_std" / "mae_modeltp_vs_imergtp_std" / "mae_era5tp_vs_imerg_std"
"lambda_obs_effective" / "lambda_obs_domain_share" / "obs_res_scale_mm" / "obs_innov_unified"
```

终端打印（region 在前、station 在后）：

```
  region  70 ch | analysis 0.xxxxx vs background 0.xxxxx -> +0.000% | better NN/70
    improved (N): r700 +2.31%, ...
    worse    (M): msl -114.32%, ...
```

**口径政策（长期）**：一律只看 `loss_domain='station_halo'`，headline 只看 **70 通道**，
并且逐通道说明"哪些提升、哪些下降"；69 通道字段只为历史连续性保留，不作结论。

---

## 15. Step 13：辅助 / 校验脚本

| 脚本 | 作用 |
| --- | --- |
| `preprocessing/check_ztd_operator.py` | numpy 算子 vs NGL 实测；三种算子形式 × 两种高度来源；总 ZTD 与站距平两套指标 |
| `preprocessing/check_ztd_torch.py` | torch 算子 vs `ztd_fuxi` store（应 <0.05 mm）；`freeze_zhd` 的数值等价性与 `msl` 梯度归零 |
| `test/plot_channel_improvement.py <run_dir>` | 把某个 run 的 `metrics*.csv` 画成逐通道改善图 + tidy `channel_improvement.csv` |
| `test/plot_stations_on_map.py` / `plot_ngl_grid_stations.py` / `plot_grid_map_points.py` | 站点与 halo 范围可视化（输出到 `da_ngl/plots/`） |
| `test/make_pipeline_flowchart.py` | 旧的流程图生成器（2026-09-15 版内容） |

---

## 16. 从零复现的命令序列

```bash
conda activate gnss
cd /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl

# --- 建库（一次性；NGL 有 cache 会快很多） ---
python test/map_stations_to_grid.py \
    --lat-min 36.50 --lat-max 56.25 --lon-min -5.25 --lon-max 24.50 --res 0.25 --seed 2021 \
    --output-map dataset/ngl_europe_0p25_80x120_station_grid_map.parquet
python preprocessing/build_ngl_zarr.py   --force --cache dataset/_cache/ngl_5min_series.npz
python preprocessing/build_fuxi_zarr.py  --force --workers 40 --leads 6
python preprocessing/build_fuxi_zarr.py  --force --workers 40 --leads 24 \
       --init-pad-hours 24 --with-tp --output dataset/fuxi_europe_0p25_24h_70ch.zarr
python preprocessing/build_label_zarr.py --force --tp-transform standardized

# --- H(FuXi) 与去偏（24h；必须同时覆盖 ztd_fuxi_zarr） ---
cd main_code
python ../preprocessing/build_ztd_fuxi_zarr.py --workers 16 --set fcst_step=4 \
       --out ../dataset/ztd_fuxi_europe_0p25_24h.zarr
python ../preprocessing/build_obs_debias.py --set fcst_step=4

# --- 训练（单卡；多卡去掉 CUDA_VISIBLE_DEVICES 限制并换端口） ---
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=22346 nohup bash train.sh \
    --model_id stage3_obsstd \
    --set results_dir=/cpfs01/.../main_code/work_dir/results/stage_three \
    > /dev/null 2>&1 &
# 进度：tail -f ../logs/stage3_obsstd_ed128_d222.log

# --- 评估（train.sh 会自动接一次；也可手动重跑，纯前向） ---
CUDA_VISIBLE_DEVICES=0 python plot_results.py --split test --set loss_domain=full \
    --model_id lead24h_halo3 --exp_tag <tag> --out work_dir/results/stage_two/eval_fullgrid
```

---

## 17. 口径陷阱与已知坑（汇总）

1. **NGL 时间单位是 minutes**，不是 hours；一律用 `decode_time_axis`。
2. **新 store 必须 `finalize_store()`**，否则 `lon==0 / time==0 / mask==False` 会变成 NaN/NaT。
3. `obs_mode='residual'` + `lambda_obs>0` 会让 `obs_at_valid_time` 读错通道（见 §10.3）。
4. **不要边训练边改 `utils_data.py`**（test 的 worker 最后才 fork）。
5. `num_epochs` 与 `num_iteration` 谁先到谁生效；截断时 **LR 调度不会跟着变**，
   "多跑几轮"和"退火更慢"会混在一起。干净跑 N 个 epoch 要
   `num_iteration = N × 每 epoch 步数` 且 `num_epochs = N`
   （3 卡 568 步/epoch，单卡 1702 步/epoch）。
6. `--set obs_res_scale_mm=None` **不可用**（`_coerce` 得到字符串 `'None'`，`float()` 报错）；
   回旧口径写 `--set obs_res_scale_mm=15`。
7. `metrics.csv` 在 `station_halo` 下是**区域**数字；重跑旧 run 的评估要显式
   `--set loss_domain=full`。
8. `metrics_region.csv` 在 `loss_domain='full'` 时退化成站点表。
9. 重跑 `plot_results.py` 会覆盖 `metrics*.csv`（`summary.json` / `.pth` / `.npy` 不覆盖）；
   想留档加 `--out`。
10. val-best 本身带噪声（多轮 `best_val` 在第 4 位小数打平，站内指标能差 0.02），
    重要结论要换 `rand_seed` 复跑。
11. 跨机器比较要留 epoch 数口子：单卡 25000 步 ≈ 14.7 epoch，3 卡 25000 步 ≈ 44 epoch。
12. 跑 24h 实验必须把 **`ztd_fuxi_zarr` 和 `fcst_step` 一起覆盖**，否则静默用 6h 的 `H(FuXi)`；
    `obs_debias` 也要用同一 lead 重建。
13. `msl` 通道单位是 **Pa**（算子内部才 /100 转 hPa）。
14. 本机没有中文字体，matplotlib 出图必须用英文。
15. 同一 `model_id` 的日志是**追加**、checkpoint 文件名会复用；换实验请换 `model_id`。
16. `include_fuxi_tp=True` 需要 70 通道的 FuXi store（`--with-tp`），
    否则 `AssimilationDataset` 直接抛错。
17. `--model_id` / `--exp_tag` / `--arch_tag` / `--set` 要一起传给 `train.sh`
    （它会把 `$@` 原样转发给评估脚本，两边才解析到同一个 run 目录）。

---

## 附：一次实验的完整目录布局

```
main_code/work_dir/results/{results_dir}/
└── {model_id}_{exp_tag}/          ← 例: stage3_obsstd_lead24h_obs6h_era5tp_w1-0_halo3_bgtp_both_obsstd_oc0.2_occ_debias
    ├── {model_id}_{arch_tag}.pth  ← val-best（例: stage3_obsstd_ed128_d222.pth）
    ├── summary.json               ← 训练超参 + best val + 背景 loss + test
    ├── train_loss.npy / val_loss.npy / lr.npy
    ├── metrics.csv / metrics_station.csv / metrics_region.csv / channels_70ch.csv
    ├── metrics.json               ← headline 与逐通道排名
    └── loss_curve.png / channel_metrics.png / maps_*.png / timeseries*.png / tp_maps.png

da_ngl/logs/{model_id}_{arch_tag}.log   ← 训练日志（只有 rank 0 写）
```

`exp_tag` 自动拼装规则（`utils.exp_tag`，顺序即拼接顺序）：

```
lead{fcst_step×6}h_obs{obs_frames×5//60}h_{tp_label_source}tp
  + _w{station_w}-{nostation_w}   （权重不是 1/1）
  + _halo{N}                      （loss_domain='station_halo'）
  + _bgtp                         （include_fuxi_tp=True）
  + _zeroobs                      （zero_obs=True）
  + _{obs_mode}                   （不是 'absolute'）
  + _obsstd                       （obs_res_scale_mm ≤ 0，统一归一化）
  + _frmsl                        （freeze_msl=True）
  + _oc{lambda}                   （lambda_obs > 0）
      + _frzzhd                   （obs_freeze_zhd=True）
      + _occ                      （lambda_obs_domain_compensation=True）
  + _debias                       （obs_debias=True）
```

`arch_tag`：`ed{model_embed_dim}_d{depth[0]}{depth[1]}{depth[2]}`，例如 `ed128_d222`。
