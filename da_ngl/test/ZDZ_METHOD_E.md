# 变更说明：ZTD 算子改为方法 E + 逐通道增量惩罚

日期：2026-09-21

## 一、动机

1. **ZTD 算子**：外部调研给出的实现是在**几何高度上积分**（`ZWD = 1e-6 ∫[(k2-εk1)e/T + k3 e/T²]dz`），
   与项目原来的气压坐标实现数学等价，但常数、积分格式、顶层处理都不同。
   逐项对比后发现（`test/compare_zwd_methods.py`，110 时次 × 1378 站）：

   | 实现 | 与 NGL GNSS 实测 ZTD 的 RMSE |
   | --- | --- |
   | 原实现 A（ZHD 解析 + ZWD 在 p 上梯形） | 14.1 ~ 14.4 mm |
   | 外部实现 B（在 z 上矩形积分，缺顶层） | 125 ~ 126 mm |
   | B + 梯形（C） | 116 ~ 117 mm |
   | **C + 顶层干柱修正（D）** | **12.2 ~ 12.4 mm** |
   | D + 地表 log-p 插值（E） | 11.9 ~ 12.4 mm |

   偏差来源全都定位清楚了：矩形法在 13 层粗廓线上高估 ZHD ~+210 mm、ZWD ~+29%；
   顶层（50 hPa 以上）干空气柱值 ~113.7 mm。修正后 dz 版**略优于**原实现。

2. **损失**：诊断（HANDOFF §12.3）与独立地面站检验都指向同一件事——网络把 msl 当成
   拟合 ZTD 的廉价杠杆（msl 站内 −108%，而其余 68 个通道合计是赚的）。
   方法 E 之后 z 通道（层高）也成为算子输入，杠杆问题会更严重。

## 二、改动内容

### 2.1 ZTD 算子 → 方法 E

| 文件 | 改动 |
| --- | --- |
| `preprocessing/ztd_operator.py` | 新增 `ztd_profile_zdz()`（几何高度梯形积分 + 顶层修正，常数 k1=77.6890、k2=71.2952、k3=375463、ε=0.62198）与 `zhd_top_correction()`；旧函数保留 |
| `preprocessing/build_ztd_fuxi_zarr.py` | `_block()` 多读 13 个 z 通道（位势 ÷ g = 层高），调用 `ztd_profile_zdz` |
| `main/model/ztd_torch.py` | 新增可微实现 `zdz_torch()`；`StationZTD` 的 `OP_CHANNELS` 从 28 → 41（加 z）；`forward` 改用方法 E；新增 `freeze_geometry`（层高与地面气压取背景并 detach；`obs_freeze_zhd` / `freeze_zhd` 作为等价别名保留） |
| `configs.py` | `ztd_fuxi_zarr` → `ztd_fuxi_europe_0p25_24h_zdz.zarr`；`obs_debias_file` → `obs_debias_lead24h_zdz.npz` |
| `preprocessing/check_ztd_torch.py` | 默认库改为跟随 `cfg.ztd_fuxi_zarr`；梯度检查适配（z 由"不读"改为"读，冻结时归零"） |

**重建产物**（旧文件全部保留，便于复现旧实验）：

* `dataset/ztd_fuxi_europe_0p25_24h_zdz.zarr`（156 MB，结构同旧库）
* `dataset/obs_debias_lead24h_zdz.npz`

### 2.2 逐通道增量惩罚（对角 B 的软约束）

```
J_B = increment_penalty_mu · mean_{c<69, 站点格} | x_a − x_b |_c / σ_b,c
```

| 文件 | 改动 |
| --- | --- |
| `preprocessing/build_bg_err_std.py` | **新增**：统计逐通道背景误差标准差（`obs − label`，train 段），输出 `dataset/bg_err_std.npz` |
| `main/model/build_optimizer.py` | **新增** `IncrementPenalty` |
| `train_FSDP.py` | 新增 `build_increment_penalty()`；训练循环累加该项并单独记录 `[increment penalty: term=…]`；summary / ckpt 记录 `increment_penalty_mu`、`obs_freeze_geometry` |
| `main/utils/utils.py` | exp_tag 新增 `_inc{mu}`、`_frgeom` |
| `configs.py` | 新增 `increment_penalty_mu`（默认 0 = 关闭）、`increment_penalty_file`、`increment_penalty_mode`、`obs_freeze_geometry` |

**为什么用背景误差标准差做分母**：store 已经用气候态 std 标准化过，若再用气候态 std
则每个通道都等于 1、失去区分度。实测（站点格，train 段）：

| 族 | σ_b |
| --- | --- |
| z50…z1000 | 0.025 ~ 0.067（**最便宜**） |
| msl | 0.066 |
| t 族 | 0.10 ~ 0.15 |
| u10 / v10 | 0.17 ~ 0.18 |
| r 族 | 0.38 ~ 0.52（**最贵**） |
| tp | 0.97 |

除以 σ_b 之后，同样的增量在 z/msl 上比在 r 上"贵" 6~10 倍，
于是观测一致性要求的修正会被推向热力与湿度廓线——这正是"不再集中在 msl、分散到各层"。

## 三、验证结果

| 项目 | 旧实现 | **方法 E** |
| --- | --- | --- |
| H(FuXi) vs NGL GNSS ZTD（55 时次 × 1378 站） | bias −10.14, RMSE 15.61 mm | **bias −3.83, RMSE 12.37 mm** |
| 逐站静态偏差 b_s 均值 | +8.27 mm | **+2.77 mm** |
| corr(b_s, 站点高度) | +0.42 | **+0.21** |
| torch vs numpy（`check_ztd_torch.py`） | 0.0007 mm | **0.0007 mm（OK）** |
| 冻结几何时的梯度 | msl=0，z\* 不读 | **msl=0 且 z\*=0，r\* 保留** |
| 端到端 smoke（含 `_inc0.1`） | — | 跑通，`[IncPenalty] mu=0.1 … sigma median=0.172` |

## 四、怎么用

```bash
cd da_ngl/main_code

# A/B 对照（其余开关与之前一致）
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash train.sh --model_id mE_inc0 \
    --set results_dir=.../results/stage_three > /dev/null 2>&1 &      # mu=0（只有算子改动）
CUDA_VISIBLE_DEVICES=0,1,2 nohup bash train.sh --model_id mE_inc01 \
    --set increment_penalty_mu=0.1 \
    --set results_dir=.../results/stage_three > /dev/null 2>&1 &      # 加软约束
# 可选第三条：彻底堵住两个廉价出口
    --set obs_freeze_geometry=true
```

建议先跑两条：`mu=0` 与 `mu=0.1`，比 region / station 的 70 通道、以及**逐通道账本里
msl 与 z\* 的 Δ**（若软约束有效，msl 的亏损应显著缩小、r 族收益应保住）。

## 五、注意事项

1. **旧 checkpoint 不能复用**：创新通道的定义变了（H(bg) 换算子），新旧不可比。
2. **方法 E 打开了新杠杆**：算子现在会读 z 通道，而 z 是网络输出之一，且 σ_b(z)≈0.043
   比 msl 还便宜。梯度检查显示 `|dH/dx_z| = 57829`（大于 t\* 的 45839、r\* 的 44494）。
   所以：**要么开 `increment_penalty_mu`，要么开 `obs_freeze_geometry`**，否则很可能
   只是把"卖 msl"换成"卖 z"。
3. **以下文档已过时**（仍按旧算子描述）：`test/DATA_PIPELINE.md` §7、`test/CODE_GUIDE.md`
   的相关段落、`da_ngl/HANDOFF.md` §12/§14 中涉及 `ztd_profile_surface` 的描述。
4. 旧产物 `ztd_fuxi_europe_0p25_24h.zarr` 与 `obs_debias_lead24h.npz` **未删除**，
   要复现旧实验把 `cfg` 里两个路径改回去即可。
