# da_ngl 代码导读

> 生成时间：2026-09-20。目的：按"代码地图 → 逐个文件 → 端到端调用链 → 必须懂的设计 → 雷区 → 阅读顺序"
> 讲清楚整个项目的代码。行数为生成时实测。
>
> 相关文档：
> * [`DATA_PIPELINE.md`](DATA_PIPELINE.md) —— 数据流全细节（建库参数、dtype/chunk、归一化公式、口径陷阱）
> * [`FLOWCHART.md`](FLOWCHART.md) + [`data_pipeline.dot`](data_pipeline.dot) —— 中文流程图
> * `work_dir/results/stage_two/experiments/` —— 各轮实验的对比结论

项目规模：30 个 Python 文件、约 7600 行。

---

## 一、代码地图

```
da_ngl/
├── preprocessing/          ① 数据生产层：把原始数据造成 zarr（离线、一次性）
├── main_code/
│   ├── main/utils/         ② 数据接口层：把 zarr 变成训练张量
│   ├── main/model/         ③ 模型与损失层
│   ├── configs.py          ④ 所有开关的唯一来源
│   ├── train_FSDP.py       ⑤ 训练主程序（把 ②③④ 串起来）
│   └── plot_results.py     ⑥ 评估与出图
└── test/                   ⑦ 校验脚本、独立基线、观测检验、文档
```

两个跨层共享的"事实来源"：

* **通道表** —— `preprocessing/common.py`（`CHANNELS` / `TRAIN_LABEL_CHANNELS` / `LABEL_CHANNELS`）；
* **ZTD 物理** —— `preprocessing/ztd_operator.py`（numpy 版）与 `main/model/ztd_torch.py`（torch 版，
  常数与通道序直接 import numpy 版，防止两套物理漂移）。

---

## 二、逐个文件

### ① preprocessing/ —— 数据生产（一次性，离线）

| 文件 | 行数 | 干什么 |
| --- | --- | --- |
| `common.py` | 207 | **全局常量与工具**：目标网格（lat 36.50–56.25 × lon −5.25–24.50，80×120）、时间基准与划分、源数据路径、三套通道表、`Region` 类（全球 720×1440 最近邻裁到目标网格，含纬度翻转与经度 0–360 映射）、`era5_channel_stats()`、`standardize_tp()`、`finalize_store()`、`decode_time_axis()` |
| `build_ngl_zarr.py` | 255 | 解析 `*.trop.zip` 的 `+TROP/SOLUTION` 段，只取秒数能被 300 整除的 `TROTOT`，装进 5 分钟网格；用 train 段有限值算 μ/σ 并整体标准化 |
| `build_label_zarr.py` | 185 | ERA5 的 0–68 直接拷贝（源里已标准化）；ERA5 自己的 tp → ch70；IMERG tp 经 `clip(0)→log1p→z-score` → ch69 |
| `build_fuxi_zarr.py` | 227 | 三个 FuXi 源按起报时间拼接去重；选 lead；`--init-pad-hours` 两端补起报；`--with-tp` 附 FuXi 自己的 tp |
| `build_ztd_fuxi_zarr.py` | 228 | 把 FuXi 背景喂给 ZTD 算子，产出 `ztd_fuxi / zhd / zwd` [mm] —— 网络看到的"创新"的基准 |
| `ztd_operator.py` | 261 | **numpy 版 ZTD 物理**：`ztd_surface`（经典地面形式）、`ztd_profile`、`ztd_profile_surface`（本项目采用：13 层廓线 + 地面节点）；物理常数；ETOPO 高程读取 |
| `build_obs_debias.py` | 142 | 逐站静态偏差 `b_s = mean_train(obs − H(bg))`，1378 个值 |
| `check_ztd_operator.py` | 154 | numpy 算子 vs NGL 实测（总 RMSE 14.3 mm，距平 12.1 mm） |
| `check_ztd_torch.py` | 163 | torch 算子 vs numpy store（差 < 0.001 mm）+ 梯度检查（msl 灵敏度 27100，z/u/v 恒为 0） |

### ② main_code/main/utils/ —— 数据接口

| 文件 | 行数 | 干什么 |
| --- | --- | --- |
| `utils.py` | 417 | **运行身份与派生量**：`model_id / exp_tag / arch_tag / experiment_dir / checkpoint_file / log_file`（run 目录命名规则都在这里）；`create_logger`；checkpoint 存取（FSDP `FULL_STATE_DICT`）；`obs_chans` / `bg_chans`（由开关推导输入通道数）；`station_cell_mask`（1378 站格 + 极性校验）；`loss_region_mask`（`full` → 站点格；`station_halo` → 膨胀 3 格）；`loss_domain_weight`（生成 cell 权重图）；`loss_domain_weight_share`（算 λ 补偿用的权重占比）；`station_geometry`（站格下标 + 高度）；`load_obs_debias` / `obs_debias_grid` |
| `utils_data.py` | 287 | **数据集**：`decode_axis`（解 CF 时间轴，NGL 是 **minutes**）+ `_assert_regular`；`AssimilationDataset`（三重存在性检查枚举样本 → `__getitem__` 拼背景/观测/标签 → `_select_label` 做 71→70）；`build_dataloader` |

`AssimilationDataset.__getitem__` 是理解一切的入口函数，它只做三件事：取 FuXi 背景 →
取 73 帧观测窗并构造观测通道 → 取标签。所有归一化、去偏、创新计算都在这里完成。

### ③ main_code/main/model/ —— 模型与损失

| 文件 | 行数 | 干什么 |
| --- | --- | --- |
| `assimilation.py` | 289 | **网络**：`pad_replicate`、`ln_norm`、`EncoderBlock` / `DecoderBlock`、`FussionNetv2` / `FussionStackv2`（bg/obs/side 三路双尺度融合）、`AssimilationBlockv2`、`EnhanceStack`、`AssimilationNetv6.forward` |
| `ztd_torch.py` | 217 | **可微 ZTD 算子**：`StationZTD` 在 1378 个站格上求值；`freeze_zhd` 让静力项取自背景并 detach |
| `build_optimizer.py` | 191 | 优化器/调度器（AdamW + warmup + cosine）；`_LatWeightedLoss`（纬度加权 × cell 权重、NaN-safe 的 MAE）；`ObsConsistencyLoss` |

`AssimilationNetv6.forward` 的八步：

1. 补齐到 80×128（replicate pad；80 能被 16 整除、120 不能）；
2. 展平时间维：obs `(B,73,5,H,W) → (B,365,H,W)`；
3. `side_info = cat([bg, obs]) → (B,435,H,W)`；
4. 三个 block：编码 40×64 → 20×32 → 解码回 40×64；
5. `decoder` 回到 80×128，输出 70 通道；
6. **残差 `out = decoder + bg`，只加在前 `bg_chans` 个通道**；
7. 两层 `EnhanceStack` 残差叠加；
8. `freeze_msl` 时把 ch68 换回背景值；最后裁回 80×120。

### ④⑤ 训练

| 文件 | 行数 | 干什么 |
| --- | --- | --- |
| `configs.py` | 251 | **所有开关的唯一来源**：路径与 run 命名；网格；样本布局（`fcst_step` / `obs_frames` / `obs_mode` / `obs_res_scale_mm`）；数据与损失（`loss_domain` / `loss_*_weight` / `lambda_obs` / `obs_debias` / `freeze_msl` / `include_fuxi_tp` / `tp_label_source`）；模型结构；优化器与训练超参 |
| `train_FSDP.py` | 657 | 训练主程序（见下面调用链） |
| `train.sh` | 46 | 启动脚本；训练结束后自动接一次 `plot_results.py`（带 `set -e`，所以评估失败会让整条链失败） |
| `configs_smoke.py` | 20 | 4 步跑通的小配置（注意 `work_dir` 改了但 `results_dir` 继承了绝对路径，要显式覆盖） |

`train_FSDP.py` 里三个设备侧函数值得单独记住：

* `process_bg` —— 把背景搬到 GPU、按需插值；
* `process_obs` —— 追加掩膜/经度/纬度通道，`zero_obs` 消融在这里实现（先算掩膜再清零）；
* `obs_at_valid_time` —— 取观测窗最后一帧还原成 mm，供观测一致性损失使用。

### ⑥ 评估

`plot_results.py`（717 行）：`resolve_checkpoint` → `load_model` → 用 `n_label_chans=71` 建数据集
（这样能同时拿到 IMERG 与 ERA5 两个 tp）→ 构造 `truth` 与 `bg70` → 三种口径统计
（全格 / 区域 / 站内）→ 写 `metrics.csv`、`metrics_station.csv`、`metrics_region.csv`、
`channels_70ch.csv`、`metrics.json` → 出六类图。

它和 `train_FSDP.py` 共享 `process_bg` / `process_obs`，所以**训练与评估的输入口径永远一致**，
不会出现"训练看到的观测和评估看到的不一样"。

### ⑦ test/ —— 不参与训练，但决定结论对错

| 类别 | 文件 |
| --- | --- |
| 数据生产辅助 | `map_stations_to_grid.py`（实际建站点映射用的就是它） |
| 算子校验 | `preprocessing/check_ztd_operator.py`、`preprocessing/check_ztd_torch.py` |
| 观测检验 | `verify_surf_obs.py`、`verify_surf_obs_with_analysis.py`、`check_ztd_from_surf_obs.py`、`check_ztd_from_analysis.py`、`plot_surf_obs_stations.py` |
| 可视化 | `plot_channel_improvement.py`、`plot_stations_on_map.py`、`plot_ngl_grid_stations.py`、`plot_grid_map_points.py` |
| 文档 | `DATA_PIPELINE.md`、`FLOWCHART.md`、`CODE_GUIDE.md`、`data_pipeline.dot` |

---

## 三、端到端调用链

```
bash train.sh --model_id X --set ...
   └─ python train_FSDP.py --configs configs [--set KEY=VAL] ...
        └─ mp.spawn(main, nprocs=GPU数)
             ├─ init_dist()         覆盖配置；重算 model_obs_chans / model_bg_chans；
             │                      装 cell weight；建 logger 与 run 目录
             ├─ build_obs_loss()    读 NGL μ/σ；load_obs_debias()；建 StationZTD；
             │                      算 λ_eff = λ / share
             ├─ make_loaders()      AssimilationDataset × (train/val/test)
             ├─ AssimilationNetv6 + FSDP(SHARD_GRAD_OP, fp16)
             ├─ 训练循环
             │    train_one_epoch: process_obs/bg → forward → loss_fn + oc → 反传
             │    evaluate:        同样前向，但只算 label loss（不含 oc）
             │    val 变好才存 {model_id}_{arch_tag}.pth
             ├─ 载回 best → 背景评估 → summary.json / train_loss.npy / val_loss.npy / lr.npy
             └─ test 评估
   └─ python plot_results.py --split test      （train.sh 自动接）
        三种口径指标 + 逐通道排名 + 六类图
```

---

## 四、五个必须懂的设计

1. **残差同化**：`out = decoder(...) + bg`，且只加在前 `bg_chans` 个通道 —— 这就是
   "分析 = 背景 + 增量"的物理结构，也是 `bg_chans` 这个数不能随便动的原因。
2. **通道语义有三层**：store 里 71 个；训练标签 70 个（0–68 + 选中的 tp）；背景 70 个
   （含 FuXi tp）或 69 个。`_select_label()` 负责 71→70，`bg_chans(cfg)` 负责背景通道数。
3. **观测通道与统一归一化**：`obs_mode='both'` 时网络看到两个通道 —— 绝对 ZTD 的 z-score，
   以及"创新"。创新 = `ztd_norm − fuxi_norm`（两者共用 NGL 训练集的 μ/σ）；
   逐站去偏 `b_s` 加在 `fuxi_mm` 上，等效于 `(obs − b_s) − H(bg)`。
4. **损失域与权重**：`loss_domain` 决定"区域"是站点格还是站点+halo；
   `loss_station_weight` / `loss_nostation_weight` 决定区域内外权重。
   **λ 域补偿用的是同一张权重图**，所以改权重会顺带改观测项的等效强度
   （`(1.0, 0.0)` + halo3 → share 0.587 → λ=0.2 变成 λ_eff=0.3407）。
5. **观测一致性损失**：唯一一个"告诉网络该往哪个方向改"的项。可微 ZTD 算子作用在分析场上，
   目标是去偏后的观测，权重 λ/σ_o；它**不参与 val-best 选择**，只单独记录。

---

## 五、改代码的雷区

1. **新增静态/地理通道不要塞进 bg 的通道轴**。`bg_chans` 被三处语义占用：
   残差 `out_chans − bg_chans`、ZTD 算子读前 69 通道、`freeze_msl` 的 ch68 索引。
   要加就走独立分支拼进 `side_info`。
2. **新 store 必须调用 `finalize_store()`**，否则 `lon==0 / time==0 / mask==False` 会被
   xarray 当成缺失值变成 NaN/NaT。
3. `model_obs_chans` / `model_bg_chans` 是 import 时算的，`--set` 之后必须重算
   （现有代码在 `init_dist` 里做了；自己加分支要照做）。
4. **不要边训练边改 `utils_data.py`** —— test 的 worker 最后才 fork，会拿到新旧混合的代码，
   `train.sh` 的 `set -e` 会让整条链失败。
5. **指标口径**：`metrics.csv` 在 `station_halo` 下其实是区域口径；val loss 的尺度随损失域变化，
   跨域不可比；全格/区域口径都被非站格主导（站格只占全格 cos(lat) 权重的 14%、halo 区域的 23.9%），
   所以判断"观测有没有用好"必须看站内口径 + 逐通道账本。
6. **观测经度是 0–360**，判域前必须先折到 −180–180，否则会漏掉本区西侧（lon<0）整条带。
7. `freeze_msl`（只锁 msl 一个通道）与 `obs_freeze_zhd`（冻整个静力项）不是一回事，
   后者被验证过不划算（湿柱独扛会让 r 族由赚转亏）。

---

## 六、建议的阅读顺序

1. `configs.py` —— 先知道有哪些旋钮；
2. `preprocessing/common.py` —— 网格 / 时间 / 通道这三个"坐标系"；
3. `main/utils/utils_data.py` —— 样本是怎么被组装出来的；
4. `preprocessing/ztd_operator.py` + `main/model/ztd_torch.py` —— 物理算子（两份对照着看）；
5. `main/model/assimilation.py` —— 网络结构；
6. `main/model/build_optimizer.py` —— 损失怎么算的；
7. `train_FSDP.py` —— 把它们串起来的主程序；
8. `plot_results.py` —— 指标是怎么定义、怎么聚合的（这决定了你怎么读结果）；
9. 最后回头看 `preprocessing/build_*.py` —— 数据是怎么造出来的。
