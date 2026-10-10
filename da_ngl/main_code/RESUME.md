# 断点恢复与命令行覆盖

以下命令在 `main_code/` 中运行，GPU 和端口仍由 `train.sh` 控制。

## 新版 checkpoint

每个 epoch 验证结束后，所有 rank 共同收集完整 FSDP 状态，rank 0 原子写入：

- `model/last.pth`：最新完整状态，用于续训。
- `model/best.pth`：验证 loss 最佳的完整状态，用于评估，也可以从该点续训。

两者保存完整模型、完整优化器、学习率调度器、GradScaler、warmup、早停状态、
iteration、下一 epoch 编号、损失历史和各 rank 的随机数状态。会比旧版只保存
rank 0 优化器分片的文件更大；首次最佳更新时需要写入两份完整 checkpoint。

```bash
bash train.sh --model_id my_run \
  --set resume_model=/absolute/path/my_run/model/last.pth
```

请沿用原训练的模型结构、数据配置、优化器与总步数，优先在同一个实验目录续训。
`num_teration`（也接受 CLI 别名 `num_iteration`）表示累计目标步数，不是新增步数。
达到目标步数或已经早停的 checkpoint 不会再额外训练一步。
学习率调度器及早停策略按 checkpoint 保存的状态恢复；改学习率计划或重新选择
早停策略时，使用下面的“仅加载权重”或显式重置模式。

保存点是 epoch 验证结束处；若中途被终止，会从上一个保存点继续。
DataLoader worker 内部的随机替代样本状态未保存，故不承诺完整数据管道逐位复现。
变更 GPU 数时完整优化器可以重新分片，但随机流和全局 batch 将变化。
checkpoint 未嵌入数据集，不会自动替换当前数据路径。

## 旧版 checkpoint

旧版把 iteration 写成字典，新加载器兼容此格式。然而旧多卡文件只包含 rank 0 的
优化器分片，且没有 GradScaler、warmup、早停历史，无法还原原来的完整训练状态。
默认拒绝将这种文件当作完整续训点，避免静默使用错误的优化器状态。

保留旧权重和 iteration，**明确重置**优化器、调度器、warmup、早停与最佳指标：

```bash
bash train.sh --model_id resumed_legacy \
  --set resume_model=/absolute/path/old_run/model/best.pth \
  --set resume_reset_optimizer=true
```

仅加载权重，从 iteration 0 开始新实验：

```bash
bash train.sh --model_id finetune \
  --set pre_model=/absolute/path/old_run/model/best.pth
```

`resume_model` 与 `pre_model` 不能同时指定。模型参数按名称严格匹配，结构不一致会报错。

## 参数覆盖

支持布尔、数值、字符串、`None`/`null`、列表、元组和字典；复合值应加 shell 引号。
未知键、冲突派生字段和常见非法值在训练初始化前报错。

```bash
bash train.sh --model_id example \
  --set add_fuxi_ztd=true \
  --set obs_frames=73 \
  --set 'model_depth=(1,1,1)' \
  --set 'early_stop={"patience":8,"min_delta":0.00001}'
```

- `add_fuxi_ztd` 同步更新 `obs_channum` 和 `model_obs_chans`。
- `obs_frames` 同步更新 `model_obs_frames`。
- `DATASET_DIR` 重定位原数据根目录下的派生路径；显式覆盖的具体路径优先。
- `obs_dir` 同步更新 `obs_stat_dir`，除非后者也被显式指定。
- `num_workers=0` 自动关闭 `persistent_workers`；显式要求两者冲突会报错。
- `early_stop=None` 关闭早停；`loss_fn=mae` / `loss_fn=mse` 选择损失函数。
- 训练和 `eval_results.py` 使用同一覆盖解析器。

示例：`--set add_fuxi_ztd=true --set obs_channum=1` 会报错；只设置前者即可。
重新设置模型输入通道/帧数之后，原架构 checkpoint 可能无法加载，这是严格检查的预期行为。
