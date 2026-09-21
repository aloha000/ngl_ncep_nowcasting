# da_ngl 数据流程图（中文）

> 与 [`DATA_PIPELINE.md`](DATA_PIPELINE.md) 配套；那份文档讲细节，这里给图。
> 生成时间：2026-09-18，对应当前 `configs.py` 默认口径。
>
> 三种呈现方式，按需取用：
> 1. **Mermaid**（下面三张图）—— GitHub / VS Code / Typora 等支持 Mermaid 的 Markdown 阅读器可直接渲染；
> 2. **Graphviz** —— [`data_pipeline.dot`](data_pipeline.dot)，`dot -Tpng data_pipeline.dot -o data_pipeline.png`；
> 3. **纯文本** —— `DATA_PIPELINE.md` 第 0 节的 ASCII 图。
>
> ⚠️ 字体：本机 `fc-list` 里 CJK 字体为 0，**在本机渲染 Graphviz 会显示成方块**；
> 请在有中文字体的机器上渲染，或把 `.dot` 里的 `fontname` 换成系统已有的中文字体
> （"Source Han Sans SC" / "Noto Sans CJK SC" / "WenQuanYi Zen Hei" / "Microsoft YaHei" / "PingFang SC"）。

---

## 1. 总体流程（离线建库 → 数据集产物 → 在线训练/评估）

```mermaid
flowchart TB
    subgraph SRC["原始数据源"]
        NGL["NGL GNSS 天顶总延迟<br/>{station}.{year}.trop.zip<br/>+TROP/SOLUTION 段，TROTOT 毫米<br/>UTC 5 分钟整点"]
        META["站点元数据 parquet<br/>站点号 / 经度 / 纬度 / 高程"]
        FUXI["FuXi 预报（三个源目录）<br/>z[time, step, channel70, lat, lon]"]
        ERA5["ERA5 6 小时分析<br/>ch0..68 已标准化，ch69 = 降水"]
        IMERG["IMERG 降水<br/>data[time, channel, lat, lon]"]
        STATS["mean_era5.npy / std_era5.npy"]
        ETOPO["ETOPO2 地形高程<br/>仅用于校验"]
    end

    subgraph BUILD["离线建库（preprocessing/，一次性）"]
        MAP["[1] 站点到格点映射<br/>最近邻落格，同格多站随机取一<br/>seed = 2021"]
        BNGL["[2] 建 NGL 库<br/>只留秒数能被 300 整除的记录<br/>train 段算均值/标准差后整体标准化"]
        BLAB["[3] 建标签库<br/>ERA5 0..68 直接拷贝<br/>ERA5 降水 → ch70，IMERG 降水 → ch69"]
        BFUXI["[4] 建背景库<br/>三源按起报时间拼接去重<br/>lead 24 小时，两端各补 24 小时<br/>附带 FuXi 自身降水作 ch69"]
        BZTD["[5] 建 H(FuXi) 库<br/>ZHD + 湿延迟 + 地面节点<br/>低于地面的层压到地面气压"]
        BDEB["[6] 建逐站去偏<br/>b_s = train 段均值(观测 − H(背景))<br/>只用训练集，共 1378 个值"]
    end

    subgraph STORE["数据集产物（da_ngl/dataset/）"]
        SMAP["站点格点映射 parquet<br/>9600 格 = 1378 有站 + 8222 无站"]
        SNGL["NGL 库 ztd[394272,80,120]<br/>标准化值 + station/mask<br/>+ ztd_train_mean/std"]
        SLAB["标签库 label[5476,71,80,120]<br/>0..68 ERA5，69 IMERG 降水，70 ERA5 降水"]
        SFX["背景库 z[5484,1,70,80,120]<br/>lead 24 小时，ch69 = FuXi 降水"]
        SZT["H(FuXi) 库<br/>ztd_fuxi / zhd / zwd [5476,80,120] 毫米"]
        SBS["obs_debias_lead24h.npz<br/>bias_mm(1378) + bias_grid"]
    end

    subgraph ONLINE["在线训练与评估（main_code/）"]
        DS["[7] 数据集组装<br/>每个样本 T：<br/>背景 = FuXi(init = T − lead)<br/>观测 = 到 T 为止的 73 帧 × 5 分钟<br/>标签 = ERA5@T（70 通道）<br/>观测通道 = [绝对 ZTD, 创新]<br/>创新 = ztd_norm − (H(FuXi) − μ)/σ<br/>含逐站去偏 b_s"]
        PREP["[8] 设备侧预处理<br/>背景 (B,1,70,80,120)<br/>观测追加 mask/经度/纬度<br/>→ (B,73,5,80,120)，站格外清零"]
        MODEL["[9] 同化网络<br/>补齐到 80×128 → 编码 40×64 → 20×32<br/>→ 解码 40×64 → 80×128<br/>out = 解码结果 + 背景（仅前 bg_chans 通道）<br/>叠加两层增强残差<br/>freeze_msl 时 ch68 强制等于背景"]
        LOSS["[10] 损失<br/>标签项：cos(纬度) 加权 MAE，<br/>限定在站点 halo3 区域（5778 格），区域外权重 0<br/>观测一致性项：λ_eff × 站内 MAE<br/>|H(分析场) − (obs − b_s)| / σ_o<br/>λ=0.2 → λ_eff=0.3407"]
        TRAIN["[11] 训练<br/>FSDP SHARD_GRAD_OP + fp16<br/>AdamW，预热 1250 步 → 余弦退火 23750 步<br/>只保存验证集最优权重"]
        EVAL["[12] 评估出图<br/>70 通道头条指标<br/>区域口径 + 站内口径<br/>逐通道改善排名<br/>metrics 系列 csv + 六类图"]
    end

    NGL --> MAP
    META --> MAP
    MAP --> SMAP

    SMAP --> BNGL
    NGL --> BNGL
    BNGL --> SNGL

    ERA5 --> BLAB
    IMERG --> BLAB
    STATS --> BLAB
    BLAB --> SLAB

    FUXI --> BFUXI
    STATS --> BFUXI
    BFUXI --> SFX

    SFX --> BZTD
    SMAP --> BZTD
    META --> BZTD
    STATS --> BZTD
    ETOPO -.->|仅校验| BZTD
    BZTD --> SZT

    SLAB --> BDEB
    SNGL --> BDEB
    SZT --> BDEB
    BDEB --> SBS

    SNGL --> DS
    SLAB --> DS
    SFX --> DS
    SZT --> DS
    SBS --> DS

    DS --> PREP --> MODEL --> LOSS --> TRAIN --> EVAL

    SNGL -.->|纯前向重读| EVAL
    SLAB -.->|纯前向重读| EVAL

    classDef src fill:#fdf3d8,stroke:#c9a227,color:#000
    classDef bld fill:#eef3fb,stroke:#4a6fa5,color:#000
    classDef sto fill:#e8f5e9,stroke:#4c8c4a,color:#000
    classDef onl fill:#f3e8f7,stroke:#7a4a9c,color:#000
    class NGL,META,FUXI,ERA5,IMERG,STATS,ETOPO src
    class MAP,BNGL,BLAB,BFUXI,BZTD,BDEB bld
    class SMAP,SNGL,SLAB,SFX,SZT,SBS sto
    class DS,PREP,MODEL,LOSS,TRAIN,EVAL onl
```

---

## 2. 单个样本的数据流（形状与通道）

```mermaid
flowchart LR
    subgraph S1["三个 Zarr + 去偏文件"]
        A1["背景库<br/>z[init, step, 70, 80, 120]"]
        A2["NGL 库<br/>ztd[394272, 80, 120]（标准化）"]
        A3["标签库<br/>label[5476, 71, 80, 120]"]
        A4["H(FuXi) 库<br/>ztd_fuxi[5476, 80, 120] 毫米"]
        A5["obs_debias_lead24h.npz<br/>bias_mm / bias_grid"]
    end

    B1["背景张量<br/>(1, 70, 80, 120)<br/>FuXi 在 T − lead 时刻"]
    B2["观测张量<br/>(73, 2, 80, 120)<br/>通道 0 = 绝对 ZTD<br/>通道 1 = 创新"]
    B3["标签张量<br/>(70, 80, 120)<br/>0..68 ERA5 + ERA5 降水"]

    C1["process_bg<br/>搬到 GPU，转 float"]
    C2["process_obs<br/>追加掩膜、经度、纬度<br/>→ (73, 5, 80, 120)<br/>站格外全部清零"]
    C3["维度补齐<br/>80×120 → 80×128<br/>（复制边缘）"]

    D1["展平时间维<br/>背景 (70, 80, 128)<br/>观测 (365, 80, 128)<br/>侧支 = 拼接 (435, 80, 128)"]
    D2["同化网络<br/>三块编码/解码 + 双尺度融合<br/>输出 (70, 80, 128)"]
    D3["残差相加<br/>out = 解码结果 + 背景<br/>（只加前 bg_chans 个通道）<br/>再叠加两层增强残差"]
    D4["裁剪回 80×120<br/>输出 (70, 80, 120)"]

    E1["标签损失<br/>cos(纬度) 加权 MAE<br/>区域外权重 0"]
    E2["观测一致性损失<br/>可微 ZTD 算子 H<br/>站内 |H(分析场) − (obs − b_s)|<br/>λ_eff / σ_o"]
    E3["总损失 = 标签损失 + 一致性损失"]

    A1 --> B1
    A2 --> B2
    A4 --> B2
    A5 --> B2
    A3 --> B3

    B1 --> C1
    B2 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> D1
    D1 --> D2 --> D3 --> D4
    D4 --> E1
    D4 --> E2
    B3 --> E1
    E1 --> E3
    E2 --> E3

    classDef data fill:#e8f5e9,stroke:#4c8c4a,color:#000
    classDef proc fill:#eef3fb,stroke:#4a6fa5,color:#000
    classDef out fill:#fdeaea,stroke:#b04a4a,color:#000
    class A1,A2,A3,A4,A5,B1,B2,B3 data
    class C1,C2,C3,D1,D2,D3,D4 proc
    class E1,E2,E3 out
```

关键数字（当前配置）：

| 项 | 值 |
| --- | --- |
| 观测窗 | 73 帧 × 5 分钟 = 6 小时，结束于分析时刻 T |
| 观测通道 | 2（绝对 ZTD、创新）+ 掩膜 1 + 经度 1 + 纬度 1 = 5 |
| 背景通道 | 70（ERA5/FuXi 69 个状态量 + FuXi 自身降水） |
| 输出通道 | 70（69 个状态量 + 训练所用的降水） |
| 空间尺寸 | 输入 80×120 → 内部补齐 80×128 → 输出裁回 80×120 |
| 参数量 | 约 7370 万（embed_dim = 128，depth = (2,2,2)） |

---

## 3. 归一化与两条输入通道（2026-09-18 口径）

```mermaid
flowchart TB
    subgraph NORM["统一归一化：obs 与 H(FuXi) 共用 NGL 训练集的均值/标准差"]
        MU["μ_ngl = 2333.6450 毫米<br/>σ_ngl = 119.7633 毫米"]
        O1["观测绝对通道<br/>ztd_norm = (obs_mm − μ)/σ"]
        O2["H(FuXi) 内部量<br/>fuxi_norm = (H(FuXi)_mm − μ)/σ"]
        O3["创新通道<br/>innovation = ztd_norm − fuxi_norm<br/>= (obs_mm − H(FuXi)_mm)/σ"]
        O4["逐站去偏<br/>先把 b_s 加到 H(FuXi) 上<br/>等价于 (obs − b_s) − H(FuXi)"]
        MU --> O1
        MU --> O2
        O1 --> O3
        O2 --> O3
        O4 --> O3
    end

    subgraph OLD["旧口径（仅用于复现 2026-09-18 之前的实验）"]
        P1["innovation = (obs_mm − H(FuXi)_mm) / 15"]
        P2["实验标签带 _obsstd 表示新口径<br/>新口径用命令行覆盖回旧口径：<br/>--set obs_res_scale_mm=15"]
        P1 --> P2
    end

    subgraph SIZE["通道尺度实测"]
        Q1["绝对通道标准差 ≈ 1.0021"]
        Q2["统一创新标准差 ≈ 0.1304"]
        Q3["旧创新标准差 ≈ 1.0409<br/>比值正好 = 15 / 119.7633 = 0.125"]
    end

    classDef n fill:#e8f5e9,stroke:#4c8c4a,color:#000
    classDef o fill:#f2f2f2,stroke:#999999,color:#000
    class MU,O1,O2,O3,O4 n
    class P1,P2,Q1,Q2,Q3 o
```

---

## 4. 口径与输出对照（评估阶段）

```mermaid
flowchart TB
    IN["训练好的权重<br/>{model_id}_{arch_tag}.pth"]
    DS["评估数据集<br/>读全部 71 个通道<br/>（两个降水都能拿到）"]
    TRUTH["真值 = 0..68 ERA5 + 训练所用的降水"]
    BG["背景 = 0..68 FuXi + FuXi 降水（若有）"]
    OUT["模型输出 70 通道"]

    IN --> OUT
    DS --> TRUTH
    DS --> BG

    M1["区域口径<br/>loss_region_mask = 站点 + halo3<br/>共 5778 格<br/>→ metrics_region.csv<br/>→ 头条指标 improve_pct_70ch_region"]
    M2["站内口径<br/>站点格 1378 个<br/>→ metrics_station.csv"]
    M3["全格口径<br/>loss_domain='full' 时是全部 9600 格<br/>当前默认只用训练区域<br/>→ metrics.csv（列名没有 _region 后缀，注意）"]
    M4["逐通道排名<br/>→ channels_70ch.csv<br/>→ metrics.json 里的 channels_improved / channels_worse"]

    TRUTH --> M1
    TRUTH --> M2
    TRUTH --> M3
    OUT --> M1
    BG --> M1
    OUT --> M2
    BG --> M2
    OUT --> M3
    BG --> M3
    M1 --> M4
    M2 --> M4

    NOTE["长期口径政策：<br/>1. 只用 station_halo，头条指标看区域；<br/>2. 一律看 70 通道；<br/>3. 必须逐通道说明哪些提升、哪些下降。"]
    M4 --> NOTE

    classDef io fill:#eef3fb,stroke:#4a6fa5,color:#000
    classDef metric fill:#fdeaea,stroke:#b04a4a,color:#000
    IN,DS,TRUTH,BG,OUT io
    class M1,M2,M3,M4,NOTE metric
```

---

## 5. 步骤与脚本对照表

| 步骤 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- |
| [1] 站点映射 | `test/map_stations_to_grid.py` | NGL 归档目录、站点元数据 | `ngl_europe_0p25_80x120_station_grid_map.parquet` |
| [2] NGL 库 | `preprocessing/build_ngl_zarr.py` | `*.trop.zip`、站点映射 | `ngl_europe_0p25_5min.zarr` |
| [3] 标签库 | `preprocessing/build_label_zarr.py` | ERA5、IMERG、mean/std | `label_europe_0p25.zarr`（71 通道） |
| [4] 背景库 | `preprocessing/build_fuxi_zarr.py` | 三个 FuXi 源 | `fuxi_europe_0p25_24h_70ch.zarr` |
| [5] H(FuXi) | `preprocessing/build_ztd_fuxi_zarr.py` | 背景库、站点高度 | `ztd_fuxi_europe_0p25_24h.zarr` |
| [6] 逐站去偏 | `preprocessing/build_obs_debias.py` | NGL 库、H(FuXi) 库、训练样本 | `obs_debias_lead24h.npz` |
| [7] 数据集 | `main_code/main/utils/utils_data.py` | 上面全部 | 样本三元组（背景、观测、标签） |
| [8] 设备侧预处理 | `main_code/train_FSDP.py` | 样本张量 | GPU 上的 `(B,1,70,80,120)` 与 `(B,73,5,80,120)` |
| [9] 网络 | `main_code/main/model/assimilation.py` | 背景、观测 | 分析场 `(B,1,70,80,120)` |
| [10] 损失 | `build_optimizer.py` + `ztd_torch.py` | 分析场、标签、观测 | 标签损失 + 观测一致性损失 |
| [11] 训练 | `main_code/train_FSDP.py` | DataLoader | val 最优权重、`summary.json`、loss 曲线 |
| [12] 评估出图 | `main_code/plot_results.py` | 权重、三个库 | `metrics*.csv`、`channels_70ch.csv`、`metrics.json`、六类图 |

校验与辅助脚本：

| 脚本 | 用途 |
| --- | --- |
| `preprocessing/check_ztd_operator.py` | numpy 版 ZTD 算子 vs NGL 实测（总延迟 RMSE 14.3 毫米，距平 RMSE 12.1 毫米） |
| `preprocessing/check_ztd_torch.py` | torch 版算子 vs 已建 store（最大差 0.0007 毫米）；冻结静力项时 msl 梯度为 0 |
| `test/linear_da_baseline.py` / `_6h.py` | 线性卡尔曼增益基线（站内 69 通道 +0.376%，是网络要追平的目标） |
| `test/plot_channel_improvement.py` | 把单个实验的逐通道改善画成图 |
