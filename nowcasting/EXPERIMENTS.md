# GNSS → NCEP Nowcasting 实验记录

日期：2026-08-27 ~ 2026-09-01。
任务：用目标站邻近 NGL 站的 ZTD/ZWD（5 分钟分辨率，窗口 T-2h..T，25 步）nowcast
NCEP 地面观测（p/slp/t2m/r2m/u10/v10）在整点 T 的值。模型：iTransformer
（`separate_output` 适配），时间切分 train 2018-01~2023-10 / val 2023-11~2024-02 /
test 2024-03~2024-08。除特别说明外均为全站 1915、`hour_stride=6`、`val_stride=6`、
`test_stride=1`、d128/4头/2层/512、batch 256、lr 1e-3、patience 3、seed 2021，
测试集 1,259,729 样本。

## 1. 数据

| 项 | 结果 |
| --- | --- |
| 5 分钟 NGL 库 | `dataset/ngl_5min.zarr`（701,569 步 × 7518 站，29.3 亿条，压缩后 7.5GB）。**真实 5 分钟观测**：从原始 NGL TRO 归档直接抽“整 5 分钟时刻”记录（`seconds % 300 == 0`），非小时平均/插值；与小时库在整点交叉验证一致，06:05 有独立值 |
| 小时 NGL 库 | `dataset/ngl_hourly.zarr`（整点抽点） |
| 样本索引优化 | 滑动窗口 → 累积和（数学等价，20 组随机数据验证一致），全量索引构建大幅加速；每 100 站打印进度 |

## 2. 实验总表

| # | 配置 | 最佳 val | early stop | overall test RMSE | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | 小时数据 T-6h，d128 | 0.896（e1） | e4 | 13.66 | 基线；预测压缩向均值 |
| 2 | 小时数据 T-6h，d256 中模型（lr 未调） | 1.044（e2） | e5 | 14.31 | 中模型无提升 |
| 3 | **5min + T-2h**，d128 | 0.902（e4） | e7 | **13.43** | 5 分钟分辨率首测 |
| 4 | **+空间编码（ENU + 高度）** | 0.709（e9） | 跑满 10 | **8.35** | 最大提升来源 |
| 5 | 空间编码，20 epoch（val_stride=1） | 0.709（e9） | e12 | 8.35 | 与 #4 逐位相同 → 平台期 |
| 6 | 空间编码 + 时间特征 5min | 0.719（e5） | e8 | 8.44 | 加分钟特征略差 |
| 7 | 空间编码 + 无时间标记 | 0.786（e2） | e5 | 9.75 | 时间信息重要 |
| 8 | 空间编码 + hour sin/cos（2 通道） | 0.775（e2） | e5 | 9.02 | p 最好（6.70）但 t2m 差 |
| 9 | 空间编码 + 全周期 sin/cos（8 通道） | 0.750（e5） | e8 | 8.71 | 第二好，仍不如 linear |
| 10 | 空间编码 + 目标站高度进输出头 | 0.706（e5） | e8 | 8.15 | 目标站高度接到最后一层线性回归前；后发现 NCEP target 高度单位需修正 |
| 11 | #10 + NCEP target 高度 `/9.80665` | 0.690（e8） | e11 | 8.18 | 高度物理量修正；val 更好但 test 略差 |
| 12 | **#11 + 经纬度进空间编码（n_geo=8）** | **0.684（e6）** | e9 | **8.02** | 当前最好；空间编码追加 target/GNSS lat/lon |

## 3. 关键分变量 RMSE（测试集，物理单位）

| 变量 | #1 小时基线 | #3 5min+T-2h | #4 +空间编码 | #7 无时间标记 | #10 +目标站高度 | #11 高度修正 | #12 +经纬度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p [hPa] | 24.55 | 24.55 | 6.80 | 7.16 | **5.96** | 6.26 | 6.00 |
| slp [hPa] | 6.68 | 6.45 | 6.40 | 6.51 | **6.18** | 6.49 | 6.21 |
| t2m [K] | 9.16 | 7.49 | 6.85 | 8.92 | 6.64 | 6.83 | **6.58** |
| r2m [%] | 19.25 | 19.06 | 16.35 | 19.49 | 16.23 | 16.01 | **15.85** |
| u10 [m/s] | 2.99 | 2.97 | 2.86 | 2.93 | 2.86 | **2.85** | **2.85** |
| v10 [m/s] | 3.04 | 3.02 | 2.95 | 2.98 | 2.95 | 2.94 | **2.93** |
| overall | 13.66 | 13.43 | 8.35 | 9.75 | 8.15 | 8.18 | **8.02** |

## 4. 结论

1. **5 分钟 + T-2h 窗口**比小时 + T-6h 略好（13.66 → 13.43），主要收益在 t2m/slp；
   注意窗口长度与分辨率同时改变，未单独归因。
2. **空间编码是最大功臣**（13.43 → 8.35）：以目标站为原点的 ENU 相对位置
   `[dE_km, dN_km, dU_m, ngl_h]`（WGS84 ECEF→ENU，dU = ngl_h − target_h），
   z-score 后经小 MLP（4→d_model）加到该邻近站的 ztd/zwd/mask 三个 token 上。
   p 的改善是“水准面级”的：基线每站偏 ~24 hPa（回归到全局平均气压），空间编码
   用绝对高度把站的气压水平锚对，逐站误差降到 3.5–10.5 hPa。
3. **目标站高度放到输出头前有用，但 NCEP 高度单位要先修正**：`target_h_feat: true` 将目标站
   绝对高度 z-score 后拼到 encoder token 输出后、`separate_output` 线性层之前，避免
   常数高度通道被 per-variate instance normalization 清零。旧 #10 使用未修正的 NCEP target
   高度（约为实际米值 × 9.80665）得到 8.15；把 NCEP target 高度 `/9.80665` 后，
   物理量合理（target_h mean/std 约 216/230 m），val 0.706 → 0.690，但 test 8.15 → 8.18。
4. **把 target/GNSS 经纬度加入空间编码后当前最好**：#12 的 `n_geo=8` 使用
   `[dE_km, dN_km, dU_m, ngl_h, target_lat, target_lon, gnss_lat, gnss_lon]`，
   在高度修正基础上把 overall 8.18 → 8.02，主要改善 p/slp/t2m/r2m。
5. **时间信息有用，分钟粒度无用，线性小时标记最优**：无时间标记 9.75；
   hour sin/cos（2 通道）9.02、全周期 sin/cos（8 通道）8.71 均不及线性小时标记
   8.35；加 MinuteOfHour（5min）8.44 略差。最终默认配置
   `time_encoding: linear` + `time_freq: h`。
6. **模型容量/训练时长**：中模型不调 lr 无提升；加大 epoch 不突破
   （最佳在第 9 epoch 左右，后续 val 在 0.71~0.73 波动）。

## 5. 特征/代码改动

- `spatial_enc: true`：per-neighbor 位置编码，模型侧小 MLP 加到该站 3 个 token 上；
  输出头与 token 布局不变，`false` 时行为与旧版逐位一致。`n_geo=4` 为
  `[dE_km, dN_km, dU_m, ngl_h]`；`n_geo=8` 追加
  `[target_lat, target_lon, gnss_lat, gnss_lon]`，这是 #12 当前最佳空间特征。
- `target_h_feat: true`：目标站绝对高度作为每样本标量，在最后 `separate_output`
  线性回归前拼接。NCEP target station 原始 `h` 为 geopotential，已按 `/9.80665`
  转成 geopotential height 后使用。
- `time_encoding: none | linear | hour_sincos`：none 无时间标记（15 token）；
  linear 用 TSL timeF 特征（`time_freq` 控制，如 `h`/`5min`）；hour_sincos 为
  `[sin(2πh/24), cos(2πh/24)]`（2 通道）。
- 代码结构：训练实现拆到 `nowcasting/main_code/`（config/dataset/model/train/plot/main），
  `nowcasting/train_iTransformer_nowcast.py` 保留为兼容入口；绘图/网格推理脚本移到
  `nowcasting/test/`。
- 配置收敛：脚本删除 `DEFAULTS`，所有参数只从 `config.yaml` 读取，缺失/未知键启动报错。
- mask 通道分析：per-variate 实例归一化会清零常数通道，mask 在当前实现中不携带信息
  （有效性由 ztd/zwd 零填充隐式表达），未做消融实验。

## 6. 运行环境注意事项

- 使用 `gnss` conda 环境（torch 2.5.1+cu121、zarr 2.18.2、numcodecs 0.12.1）。
- 后台训练必须 `setsid -f ... > log 2>&1` 脱离会话；`nohup &` 会被环境连带杀掉。
- 小模型 + `num_workers=0` 时 GPU 利用率 ~20% 属正常（CPU 数据生成是瓶颈）。
- 运行命令见 `nowcasting/README.md`；参数全部在 `nowcasting/config.yaml`。

## 7. 当前状态

- 四种时间编码（none / linear / hour_sincos / sincos）全部完成对比，
  **linear + time_freq=h 表现最好**，已设为 config.yaml 默认值。
- NCEP target station 高度已从 geopotential 按 `/9.80665` 转为 geopotential height；
  corrected-height 基线 #11 test overall RMSE 8.18。
- 当前最佳为 #12：`n_geo=8` 经纬度空间编码，best val 0.684（epoch 6），
  全测试集推理已完成（test samples 1,259,729，batches 4,921），test overall RMSE 8.02，
  输出见
  `nowcasting/outputs/gnss_nowcast_hg_ll_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf/`。
- 可选的后续实验：mask 通道消融（10 通道版）、更大模型 + 调低 lr、
  `num_workers`/`batch_size` 提速、经纬度编码方式消融（原始 lat/lon vs sin/cos/归一化投影）。

## 8. 可视化输出

- 最佳模型 #12 的全测试集推理文件：
  `nowcasting/outputs/gnss_nowcast_hg_ll_s1915_off0_h6_dm128_el2_nh4_df512_sp_thf/test_predictions.npz`。
- 五个固定时刻 station-level 对比图已用
  `nowcasting/test/plot_station_compare.py` 重新生成，行列布局为 `3 × 6`
  （obs / pred / ERA5 × p/slp/t2m/r2m/u10/v10），每个变量底部一个共享 colorbar。
  输出文件：
  - `station_cmp_2024031400.png`
  - `station_cmp_2024052012.png`
  - `station_cmp_2024060112.png`
  - `station_cmp_2024070418.png`
  - `station_cmp_2024081006.png`
- 新增连续时间动图脚本 `nowcasting/test/plot_station_compare_anim.py`。
  默认画 `2024-05-19 00:00` 到 `2024-05-20 23:00 UTC` 的逐小时变化，
  18 个子图同步更新，输出：
  `station_cmp_anim_20240519_20240520.gif`。
  本次生成结果为 48 帧，尺寸 `4320 × 1440`，文件约 18.8 MB。
  日志：
  `nowcasting/logs/plot_station_compare_anim_hg_ll_20240519_20240520.log`。
