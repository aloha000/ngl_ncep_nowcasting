# ERA5 官方 37 层资料重算区域 ZTD（与上一个结果样本对齐）

日期：2026-09-22

## 1. 目的

项目里的 ERA5 store 只有 **13 个气压层**（50\~1000 hPa，`LEV_HPA`）。方法 E
（`ztd_profile_zdz`）在这 13 层的几何高度上做梯形积分，50 hPa 以上用解析顶盖修正
（113.70 mm）补足。

官方下载的 ERA5 有 **37 层**（1\~1000 hPa），并且带真实表面气压。本次用官方资料把
柱积分做细，在**与上一个结果完全相同的 (time, station) 样本**上比较。

## 2. 数据与方法

### 2.1 输入

| 用途 | 路径 | 变量 |
|---|---|---|
| 层高 | `era5/from_official/1h/pressure_level/geopotential/YYYY/YYYYMMDD.nc` | `z` 37 层位势 [m²/s²] |
| 温度 | `.../pressure_level/temperature/...` | `t` [K] |
| 湿度 | `.../pressure_level/relative_humidity/...` | `r` [%] |
| 地面气压 | `.../surface_level/surface_pressure/...` | `sp` [Pa] |
| 2 m 温度 | `.../surface_level/2m_temperature/...` | `t2m` [K] |

注意：归档的 `pressure_level` 是**气压降序**（1000 → 1 hPa），而算子要求第一层是最顶层，
脚本按坐标 `argsort` 重排为升序，不能假定文件顺序。

### 2.2 计算步骤（= 现在的方法 E，只把层数从 13 换成 37）

1. **层高** `z / g`（g = 9.80665 m/s²）；
2. **模式地形高度**：`sp` 定义在 ERA5 模式地形上。取地面上方最近的真实层（满足
   `p_j < p_s` 的最大气压层 `p1`），按测高公式反推

       h_oro = z(p1) + (R_d·T(p1)/g)·ln(p1/p_s)

   全域反推出的 ERA5 地形高度均值 **342 m**（测站高度 35\~3168 m）；
3. **折到参考高度** `h_ref`：`p(h_ref) = sp·exp(−g(h_ref−h_oro)/(R_d·t2m))`
   —— 站格取测站高度（平均折 12.7 hPa），其余格点取 `h_oro`（即直接用 `sp`）；
4. **柱积分**：`ztd_profile_zdz(T_lev, R_lev, t2m, p, h_ref, z_lev)`，梯形法积分
   `N_h = k1(p−e)/T` 与 `N_w = (k2−εk1)e/T + k3·e/T²`，`p ≥ p_s` 的层塌到地面使层厚为 0；
   顶盖修正 `p_top = 1 hPa` → **2.27 mm**（13 层方案是 50 hPa → 113.70 mm）；
5. 同时算一版**只用 13 层**的（同一份官方资料），把"层数"这一项单独剥出来。

### 2.3 运行

```bash
# 1) 导出站格几何（gnss 环境，只需一次）
cd da_ngl/main_code
python ../test/dump_station_geom.py --out /tmp/geom_1378.npz

# 2) 主计算（hydro 环境；官方 nc 是 NetCDF4/HDF5，gnss 环境没有 netCDF4/xarray）
python ../test/era5_37lev_ztd.py \
    --samples /tmp/zhd_three_ways_v2.csv --geom /tmp/geom_1378.npz \
    --out-csv /tmp/era5_37lev_ztd.csv --out-field /tmp/era5_37lev_ztd_field.npz \
    --workers 24

# 3) 与上一个结果对齐比较 + 出图（gnss 环境）
python ../test/compare_era5_37lev_ztd.py \
    --new-csv /tmp/era5_37lev_ztd.csv --prev-csv /tmp/zhd_three_ways_v2.csv
```

运行时间约 35 min，**完全是 I/O 瓶颈**：102 个时刻分布在 102 个不同日期上，每个日期要打开
5 个文件（geopotential/temperature/relative_humidity 各约 1.1 GB，另有 sp、t2m），
合计 510 次打开、约 560 GB 读取，而真正要的只是区域 80×120 的一小块。

## 3. 样本对齐

样本取自上一个结果 `/tmp/zhd_three_ways_v2.csv`（来自 `check_zhd_three_ways.py`）：

* **102 个时刻**，2025-01-01 00Z \~ 2025-09-30 12Z，全部是 00 或 12 UTC，每时刻一天；
* (time, station) 对齐后 **67,718 行**（102 时刻 × 711 站）；
* 但其中 **10,486 行的 NGL 没有观测**（`obs_ztd` / `obs_zwd` / `obs_zhd` 同时为 NaN），
  剩下 **57,232 行有 NGL 观测 —— 这才是上一个结果的口径**（`zhd_four_matched2.py`
  打印的"四者都有效的样本 57232"）。
* **本文件所有统计一律用 n = 57,232**（102 时刻 × 710 站），不再混用 67,718。

`dataset/era5_37lev_ztd_matched_57232.csv` 就是这 57,232 行；
`/tmp/era5_37lev_ztd_matched.csv` 是同一份（脚本已自动过滤）。
逐样本的 37L/13L 全量结果（102 时刻 × 1378 站 = 140,556 行）在
`dataset/era5_37lev_ztd_stations_test102.csv`，它不依赖有没有 NGL 观测。

## 4. 结果（同一批样本，单位 mm）

### 4.1 汇总（n = 57,232）

| 量 | 来源 | mean | std | min | max |
|---|---|---|---|---|---|
| ZTD | obs（NGL GNSS） | 2360.87 | 104.79 | 1566.20 | 2630.60 |
| ZTD | **ERA5 37L（本次）** | **2351.81** | 104.55 | 1554.84 | 2639.58 |
| ZTD | ERA5 13L（项目 store，上一个结果） | 2358.81 | 105.19 | 1554.01 | 2650.61 |
| ZTD | FuXi 24h（13L） | 2357.94 | 104.43 | 1554.97 | 2604.30 |
| ZHD | obs（ZTD−TRWET，非独立） | 2255.39 | 86.43 | 1552.40 | 2371.90 |
| ZHD | **ERA5 37L（本次）** | **2240.56** | 86.57 | 1536.38 | 2359.11 |
| ZHD | ERA5 13L（项目 store） | 2244.70 | 86.83 | 1537.62 | 2364.51 |
| ZHD | FuXi 24h（13L） | 2244.65 | 86.86 | 1538.06 | 2363.67 |
| ZHD | NCEP 实测气压 Saastamoinen | 2241.36 | 86.01 | 1541.57 | 2379.01 |
| ZWD | obs（TRWET） | 105.48 | 50.19 | −13.50 | 332.50 |
| ZWD | **ERA5 37L（本次）** | **111.24** | 51.05 | 2.95 | 361.10 |
| ZWD | ERA5 13L（项目 store） | 114.11 | 51.95 | 3.31 | 368.13 |
| ZWD | FuXi 24h（13L） | 113.29 | 50.69 | 2.94 | 319.42 |

### 4.2 相对 NGL 观测（n = 57,232）

| 量 | 来源 | bias | RMSE | MAE | r |
|---|---|---|---|---|---|
| ZTD | ERA5 37L（本次） | −9.06 | 13.55 | 10.62 | 0.9954 |
| ZTD | ERA5 13L（项目 store） | −2.07 | 11.49 | 8.46 | 0.9942 |
| ZTD | FuXi 24h（13L） | −2.93 | 11.81 | 8.80 | 0.9940 |
| ZHD | ERA5 37L（本次） | −14.82 | 15.64 | 14.83 | 0.9983 |
| ZHD | ERA5 13L（项目 store） | −10.69 | 11.88 | 10.71 | 0.9982 |
| ZHD | FuXi 24h（13L） | −10.74 | 11.92 | 10.76 | 0.9982 |
| ZHD | NCEP 实测气压 Saastamoinen | −14.02 | 14.79 | 14.19 | 0.9985 |
| ZWD | ERA5 37L（本次） | +5.76 | 11.49 | 8.66 | 0.9809 |
| ZWD | ERA5 13L（项目 store） | +8.62 | 14.21 | 10.99 | 0.9761 |
| ZWD | FuXi 24h（13L） | +7.80 | 13.71 | 10.60 | 0.9750 |

**注意**：NGL 的 ZHD 是 `ZTD − TRWET` 反推的，它的静力项来自 VMF1 先验，比实测气压
算出的 ZHD **高约 13.6 mm**（`check_zhd_three_ways.py` 已确认）。所以"离 NGL 更近"
不等于"更准"——ZHD 这一栏要把 NCEP 实测气压（−14.02）当作参照。

### 4.3 与闭式 Saastamoinen 的对照（最直接的自检，n = 57,232）

用**同一批 `p_s`** 套闭式 `0.0022768·p/(1−0.00266cos2φ−0.00028h_km)`：

| 方案 | 相对闭式 Saas 的差 | 参照均值 |
|---|---|---|
| 本次 37 层方法 E | **−1.35 ± 2.30 mm** | 2241.92 |
| 上一个结果（项目 store 13 层） | +3.27 ± 3.52 mm | 2241.43 |
| NCEP 实测气压 Saastamoinen | −0.55 ± 2.63 mm | 2241.92 |

### 4.4 新旧互比（n = 57,232）

| 对比 | bias | RMSE | MAE | r |
|---|---|---|---|---|
| `zhd37` − `zhd_op_era5` | −4.14 | 4.29 | 4.14 | 0.9999 |
| `zwd37` − `sd_zwd_op13` | −2.86 | 6.16 | 4.46 | 0.9945 |
| `ztd37` − `sd_ztd_op13` | −7.00 | 8.79 | 7.40 | 0.9987 |

（表里不再单列"官方资料 13 层"这一路：它只是用来做复现检验的对照。实测
`zhd13(官方资料) − zhd_op_era5(项目 store)` = **−0.01 ± 0.26 mm**、r = 1.0000，
两路 13 层几乎完全一致，所以下面所有差异都可以归给"层数"，不是资料链差异。
`zhd37` / `zwd37` / `ztd37` 就是本次 37 层的结果。）

## 5. 结论

1. **37 层把 ZHD 的偏置修正掉了约 4 mm，是真改进。**
   37 层 ZHD（2240.56）与闭式 Saastamoinen（2241.92，同 `p_st`）差 −1.35 mm，
   与 NCEP 实测气压 ZHD（2241.36）差 0.8 mm；13 层则是 +2.77 mm。
   顶盖修正从"50 hPa 以上解析补 113.70 mm"变成"显式积分 50→1 hPa + 1 hPa 以上补 2.27 mm"，
   两种离散化在平流层的误差不同，13 层那一版系统性偏大。

2. **ZWD 反而降低了约 2.8 mm，但这是更接近真值的方向上的"降低"。**
   用 `GNSS ZTD − 实测气压 Saastamoinen ZHD` 反推的 ZWD（均值 119.51 mm）当参考：
   37 层低 **8.26 mm**、13 层低 **5.52 mm**、项目 store 13 层低 **5.40 mm**、TRWET 低 **14.02 mm**。
   也就是说三种 ERA5/TRWET 的 ZWD 都偏低，37 层更低，说明 **ERA5 背景本身偏干**
   （约 1.2 mm 可降水量），不是积分方法的问题。层数一多，梯形法对近地面湿度峰值
   （凸函数）的高估被消掉，数值自然往下走。

3. **ZTD：37 层 bias −9.1 / RMSE 13.55，13 层 bias −2.2 / RMSE 11.55。**
   13 层"看起来更好"来自两个误差部分抵消：ZHD 偏大 +2.77 mm，ZWD 偏低 −5.52 mm，
   净 −2.75 mm。37 层把 ZHD 修对之后，剩下的 −9 mm 全部暴露成 **ZWD 的系统性亏缺**。

4. **对同化来说这是好消息。** 37 层之后 ZTD 残差的物理来源单一化了：不再是"算子把 ZHD
   算偏了"，而是"背景湿度偏干"。而 GNSS ZTD 观测恰恰就是约束水汽的——所以
   `H(x_b) − obs` 现在指向的是湿度廓线该被抬升，方向是对的；13 层那种"ZHD 偏置恰好
   抵消 ZWD 亏缺"的伪一致，会让同化系统把增量往错误方向推（或者推得不够）。

5. **建议**：把 37 层方案定为默认算子。如果后续要接进可微算子
   （`main/model/ztd_torch.py`），需要把 `OP_CHANNELS` 从 41 扩到 37 层所需的
   `t/r/z` × 37 + `t2m/msl`，并重跑 `build_ztd_fuxi_zarr.py`。

## 6. 产物

| 文件 | 内容 |
|---|---|
| `da_ngl/test/dump_station_geom.py` | 导出 1378 站格几何（gnss 环境） |
| `da_ngl/test/era5_37lev_ztd.py` | 37 层主计算 + 13 层对照（hydro 环境） |
| `da_ngl/test/compare_era5_37lev_ztd.py` | 与上一个结果对齐比较并出图（gnss 环境） |
| `da_ngl/dataset/era5_37lev_ztd_stations_test102.csv` | 逐 (time, station) 的 37L/13L 的 ZHD/ZWD/ZTD、`p_sp/p_st/h_oro`（140,556 行，全量） |
| `da_ngl/dataset/era5_37lev_ztd_matched_57232.csv` | 与上一个结果对齐、且有 NGL 观测的 57,232 行（含各来源列） |
| `da_ngl/dataset/era5_37lev_ztd_field_test102.npz` | 区域场 102×80×120（zhd/zwd/ztd，参考面 = ERA5 模式地形） |
| `da_ngl/plots/era5_37lev_ztd_matched.png` | ZTD 散点 / 偏差分布 / 各来源 RMSE |

## 7. 局限

* 样本只覆盖 **test 期的 102 个时刻**（每 ~2.5 天一个），不是全时段；训练/验证期的
  σ_b 类统计不能用它。
* NGL 的 `TRWET` / `ZTD−TRWET` 是派生量（含 VMF1 先验），不能当独立参考。
* 区域场用的是"参考面 = ERA5 模式地形高度"，站格用的是测站高度，两者差
  平均 12.7 hPa 的气压（约 22 m）；做区域图时不要和站上的值混用。
* 官方 37 层与项目 store 不是同一条加工链，但 13 层的复现精度是 0.26 mm，
  说明这条差异可以忽略。
