# GNSS ZTD + FuXi → ERA5 同化训练代码

基于 `xuxiaoze/for_zrx/train_packet` 改写的版本，数据换成 `da_ngl/dataset` 里我们自己建的三个 zarr。

## 目录

```
main_code/
├── configs.py              全部超参 + 数据路径（默认配置）
├── configs_smoke.py        极小配置，用于验证环境能跑通
├── train.sh                CUDA_VISIBLE_DEVICES + 启动 FSDP 训练
├── train_FSDP.py           训练主循环（FSDP / fp16 / warmup+cosine / 存盘 / 评估）
└── main/
    ├── model/
    │   ├── assimilation.py      AssimilationNetv6（背景场 + 观测 → 分析场）
    │   └── build_optimizer.py   优化器、调度器、早停、纬向加权 MAE/MSE
    └── utils/
        ├── utils.py             checkpoint、logger
        └── utils_data.py        读三个 zarr 的 Dataset / DataLoader
```

## 一个样本长什么样

以 6 小时分析时刻 `T` 为单位：

| 角色 | 来源 | 形状 |
| --- | --- | --- |
| 背景场 `bg` | `fuxi_europe_0p25.zarr`，起报时刻 `init = T − fcst_step×6h` | `(1, 69, 80, 120)` |
| 观测 `obs` | `ngl_europe_0p25_5min.zarr`，`T` 前 2 小时的 25 个 5 分钟帧 | `(25, 1, 80, 120)` 原始 ZTD |
| 标签 `label` | `label_europe_0p25.zarr`，`T` 时刻 | 训练读前 `(70, 80, 120)` = ERA5 69 通道 + IMERG tp；store 里还有第 71 个通道 `era5_tp`（仅评估用，训练不用） |

`process_obs` 再把观测整理成模型输入：追加**逐帧有效性 mask**、`lat`、`lon` 通道（共 4 通道/帧），并按 mask 把无效格置零 —— 与原始包的掩膜约定一致。

样本量：train 3403 / val 980 / test 1092（6 小时一个）。

## 与原包的差异

| 项 | 原包 | 本版本 |
| --- | --- | --- |
| 数据读取 | 每个样本开 nc 文件 | 直接读 zarr（`AssimilationDataset`），惰性句柄、可用多 worker |
| 网格硬编码 | `linspace(90,-90,720)` / `linspace(0,359.75,1440)` | 改成区域真实的 `lat 36.50..56.25` / `lon −5.25..24.50` |
| 损失纬度权重 | 全局 720 行 | 区域 80 行，且**NaN 安全**（标签有约 0.012% 缺测） |
| `process_bg` | 默认插值到 720×1440 | 不再需要（数据已是 80×120），保留可选 resize |
| 通道数 | bg 100 / obs 40 帧 6 | bg 69 / obs 每帧 4，帧数可配（默认 25） |
| 输出通道 | = bg 通道 | `out_chans=70`，**残差只加在前 69 通道**，第 70（IMERG tp）从零学 |
| 空间尺寸 | 隐含要求能被 16 整除（720×1440 满足） | **入口自动 pad 到 16 的倍数、输出裁回**（120 → 128） |
| 观测标准化 | 用 `obs_stat_dir` 的 mean/std | 不用（三个 store 本身已标准化） |
| 其他 | `sudo cp`、过期 `__main__`、每样本 print | 已去掉 |

原始结构里 `out = decoder(...) + bg` 要求背景场和标签在同一标准化空间。FuXi 和 ERA5 都用 `mean_era5/std_era5`，前 69 通道没问题；IMERG tp 没有对应的 FuXi 背景，所以不能进残差，这就是把残差限制在前 `bg_chans` 通道的原因。

## 运行

```bash
conda activate gnss        # 需要 torch + zarr + einops
cd da_ngl/main_code
bash train.sh              # 或者: python train_FSDP.py --configs configs
```

先跑冒烟测试确认环境：

```bash
CUDA_VISIBLE_DEVICES=0 python train_FSDP.py --configs configs_smoke --master_port 22399
```

它会用 4 个样本训练 4 个 iteration，然后做 train/val/test 评估，几秒钟即可完成。

## 多卡训练

脚本用 `torch.cuda.device_count()` 取可见 GPU 数，按 `mp.spawn` 一卡一进程启动，所以
**只要设好 `CUDA_VISIBLE_DEVICES` 就行**：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 MASTER_PORT=22336 bash train.sh     # 三张卡
CUDA_VISIBLE_DEVICES=0,1   MASTER_PORT=22337 bash train.sh     # 两张卡
```

启动后看日志里第一行 `[World Size]` 确认进程数对不对。

### 换到另一台机器

1. 把整个 `main_code/` 拷过去（`cp -r` 即可，无额外依赖）。
2. 确认那台机器上 **能访问三个 zarr**：`configs.py` 里 `DATASET_DIR`、`work_dir` 都是绝对路径，
   挂在别的挂载点时改这两处即可。
3. 环境需要 `torch`（带 NCCL）+ `zarr` + `numpy/pandas` + `einops`。
4. 先验证：

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
   CUDA_VISIBLE_DEVICES=0 python train_FSDP.py --configs configs_smoke --master_port 22399
   ```

### 多卡时要调的参数

- **`num_iteration` 是每卡的优化步数**。全局 batch = `batch_size × 卡数`，所以在同一
  `num_iteration` 下三卡看到的数据量是一卡的三倍；想保持和一卡方案相同的数据量，
  把 `num_iteration` 除以卡数（或者把 `batch_size` 除以卡数）。
- 全局 batch 变大后学习率通常也相应调大（线性/平方根缩放都常见），
  `learning_rate` / `stop_lr` 在 `configs.py` 里。
- 每个 epoch 的步数 = `ceil(样本数 / (batch_size × 卡数))`，日志里会打印。

### 实现要点（多卡相关的改动）

- **保存**：改用 FSDP 的 `FULL_STATE_DICT` API（`save_checkpoint_fsdp`）。原来
  `summon_full_params` + `model.module.state_dict()` 在 `SHARD_GRAD_OP` 下拿到的是分片张量，
  多卡时存出来的 checkpoint 是坏的。现在存出来的是 CPU 上的完整权重 + 优化器/调度器状态。
- **加载**：`load_checkpoint` 会识别 FSDP 包装并在 `FULL_STATE_DICT` 上下文里加载。
- **修了一个死锁**：收尾的 background-loss 评估原来写在 `if rank == 0` 里，而
  `evaluate()` 内部有 `dist.all_reduce` —— 多卡时 rank 1/2 不进入就会永久挂住。
  现在所有 rank 都跑评估，只有 rank 0 负责写日志/存文件。
- `DistributedSampler(drop_last=False)` 会给各 rank 补齐成相同 batch 数，所以
  `all_reduce` 不会因为各卡步数不同而卡住。

> 注意：当前容器只有 1 张卡，上面这些多卡路径是按标准 FSDP 写法实现的，但**没法在这里实测**。
> 第一次上三卡时建议先跑几个 iteration，确认日志里 3 个 rank 都在推进、checkpoint 能正常存/续。

## 主要参数（`configs.py`）

```python
fcst_step = 1                # init = T - fcst_step*6h; step = fcst_step*6h (本 store 只有 lead 6h)
label_n_chans = 70           # 标签 store 有 71 通道，训练只读前 70（第 71 个 era5_tp 仅评估用）
obs_frames = 25              # 观测窗口长度（25 x 5min = 2h）
obs_end_offset_minutes = 0   # 窗口相对 T 的偏移
model_bg_chans = 69          # FuXi 通道
model_out_chans = 70         # ERA5 69 + IMERG tp
model_embed_dim = 128
model_depth = (2, 2, 2)
num_iteration = 20000        # 总优化步数
batch_size = 2
loss_fn = mae()              # 纬向加权、NaN 安全；也可用 mse()
```

单卡显存参考：`embed_dim=128, depth=(2,2,2)` 约 **73.5M** 参数；冒烟配置 `(32,(1,1,1))` 约 3.0M。

## 冒烟测试结果

`configs_smoke` 在 1 张卡上跑通：4 个 iteration 正常前向/反向/存盘，
随机初始化时 model loss ≈ 0.49、背景场 loss ≈ 0.10（训练前模型比背景场差是正常的）。

## 注意事项

- 模型要求 H、W 能被 **16** 整除；80 满足，120 会被 pad 到 128（自动，不影响结果）。
- `obs_frames` 增大时观测通道按 `obs_frames × 4` 增长，显存和 IO 都会上升。
- NGL 的缺测是**逐格逐帧**的，mask 按帧计算；某格某段时间完全无数据时会一直是 0。
- 数据集只保留 `init = T − fcst_step×6h` 存在的样本；训练/验证/测试区间是半开区间 `[start, end)`。
- 背景场口径与参考实现 `read_bg` 一致：`init = date - fcst_step*6h`、`step = fcst_step*6h`；数据集会按 `fcst_step` 去 zarr 的 `step` 轴上定位对应的 lead（当前 `step=[6]` → `fcst_step=1`）。
- 目前容器只有 1 张 GPU；多卡时 `train.sh` 会按 `CUDA_VISIBLE_DEVICES` 自动 spawn 相应进程数。
