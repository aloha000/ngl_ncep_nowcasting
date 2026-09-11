# da_ngl 数据集（GNSS ZTD + FuXi → ERA5/IMERG）

在以**下统一网格**建了三个 Zarr，构建脚本在 `preprocessing/`：

| 数据 | 文件 | 维度 | dtype |
| --- | --- | --- | --- |
| NGL | `dataset/ngl_europe_0p25_5min.zarr` | `ztd[time,lat,lon]` | f4 |
| FuXi | `dataset/fuxi_europe_0p25.zarr` | `z[init,step(1),channel(69),lat,lon]` | f4 |
| 标签 | `dataset/label_europe_0p25.zarr` | `label[time,channel(71),lat,lon]` | f4 |

## 网格

* 0.25°，`lat 36.50..56.25`（80 点），`lon -5.25..24.50`（120 点），都是严格的 0.25 整数倍。
* 站点→格点映射：`dataset/ngl_europe_0p25_80x120_station_grid_map.parquet`
  （每格一站；多站则随机取一个，seed 2021）。
* 源场（FuXi/ERA5/IMERG）是全球 720×1440、经度 0..360，按区域裁剪后**最近邻**采到目标格点
  （实现在 `common.Region`）。注意源纬度轴是 ±90 之间 720 点等分（间距 0.250347°，非严格 0.25°），
  所以纬度中心最大偏差 0.075°；经度完全对齐。

## 时间

三个 store 的时间轴都用 CF 格式（int64 数值 + `units` 属性），公共基准时刻为 `2022-01-01 00:00 UTC`。

* 覆盖范围：`2022-01-01 00:00` ～ `2025-10-01 00:00`（半开区间；上限取到 IMERG 结束处）。
* 划分（只写在 attrs 里，物理上不切分）：
  * 训练集 `2022-01-01 .. 2024-05-01`
  * 验证集 `2024-05-01 .. 2025-01-01`
  * 测试集 `2025-01-01 .. 2025-10-01`
* NGL 为 5 分钟；FuXi 为 6 小时起报、**只保留 lead = 6 h**；标签为 6 小时。

## 三个 store

**NGL**（`ngl_europe_0p25_5min.zarr`）
* `ztd[time,lat,lon]`：NGL 的 `TROTOT`（mm），已用**训练集**统计量标准化成
  `(ztd_mm - ztd_train_mean) / ztd_train_std`；`ztd_train_mean = 2333.6450`、
  `ztd_train_std = 119.7633`（同时存成数组和 attrs）。没有站点的格点是 NaN。
* `station[lat,lon]`（站点号，空为 `""`）、`mask[lat,lon]`（True = 无站点，共 8222 格）。
* 原始数据：`ngl_ztd_all_downloaded/data/top10_2022_2026/Western_and_central_Europe`。

**FuXi**（`fuxi_europe_0p25.zarr`）
* `z[init,step,channel,lat,lon]`，**起报和 lead 都保留**（`step` 轴只有 1 个值 `6`，即 lead = 6 h）；
  69 个通道（去掉 `tp`）。数值按原样存，即已经用 `mean_era5.npy` / `std_era5.npy` 标准化过。
  xarray 会把 `step` 按 CF 解码成 timedelta（6 h）；想要整数 6 的话去掉它的 `units` 属性即可。
  用 `--leads 6,12,24` 可以保留多个 lead。
* 来源按起报时间拼接：`Fuxi_pred_2017_2024`、`FuXi_Pred_2024_2025`、`FuXi_Pred_2025_other`。

**标签**（`label_europe_0p25.zarr`）
* `label[time,channel,lat,lon]`，共 **71** 通道：
  * `0..68`：ERA5（同上标准化）
  * `69 = tp`：IMERG 的 `tp`，变换与 ERA5 tp 一致但**不乘 1000**：
    `(log1p(clip(tp_mm,0)) - 0.2754) / 0.4103`
  * `70 = era5_tp`：ERA5 自己的 `tp`（直接取源 ERA5 的第 69 通道，同一套 log1p+标准化）。
    **仅用于评估/画图，不参与训练**（`attrs.eval_only_channels = ["era5_tp"]`）。
* 来源：`huangyuanqing/data/ERA5_2017_2025`（6 小时）、`database/fuxi-obs/imerg/zarr_25_720_more`。

## 读取

```python
import xarray as xr
ds = xr.open_zarr("da_ngl/dataset/label_europe_0p25.zarr")   # 已写 consolidated metadata
```

坐标类数组的 `fill_value` 统一写成 `null`（见 `common.finalize_store`）。原因是 zarr 默认
`fill_value` 是 `0/False/''`，会和真实值撞车（`lon == 0.0`、`time == 0`、`mask == False`），
xarray 会把 `_FillValue` 当缺失值，导致这些值变成 NaN/NaT。

## 训练代码

`main_code/` 是基于 `xuxiaoze/for_zrx/train_packet` 改写、适配上面三个 zarr 的同化训练代码
（背景场 FuXi + 观测 NGL ZTD → 分析场 ERA5，FSDP + fp16）。用法和改动清单见
[`main_code/README.md`](main_code/README.md)。

## 重建

```bash
python da_ngl/preprocessing/map_stations_to_grid.py --lat-min 36.50 --lat-max 56.25 \
    --lon-min -5.25 --lon-max 24.50 --res 0.25 \
    --output-map da_ngl/dataset/ngl_europe_0p25_80x120_station_grid_map.parquet
python da_ngl/preprocessing/build_ngl_zarr.py   --force --cache da_ngl/dataset/_cache/ngl_5min_series.npz
python da_ngl/preprocessing/build_fuxi_zarr.py  --force --workers 40
python da_ngl/preprocessing/build_label_zarr.py --force --tp-transform standardized
```
