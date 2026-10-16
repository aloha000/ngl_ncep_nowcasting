#!/usr/bin/env python3
"""评估：模型分析场 vs 背景场（FuXi 原始）对 ERA5 的 MAE，逐通道 + 汇总。

和训练用的是同一套数据管道、同一套 process_bg / process_obs、同一个网格，所以
"背景场 MAE"就是 batch_fcst 直接对 ERA5 的 MAE，"分析场 MAE"是模型输出对 ERA5 的
MAE，两者在同样的样本上算，可以直接比。

输出：
    stdout      逐通道表格 + 汇总（全网格 / 站点格）
    CSV         eval_channels_<split>.csv     逐通道 bg/analysis MAE 与提升率
    PNG         eval_channels_<split>.png     逐通道提升率条形图 + 整体对比

用法（在 da_ngl/main_code 下）
----
    python eval_results.py --configs configs --split test \
        --checkpoint work_assi_ztd/model/iteration_XXXX.pth

不传 --checkpoint 时依次找 {work_dir}/{model_id}/model/ 下的 best.pth、最新的 .pth。

评估只遍历有效日期一次，不随机替换缺失样本，loss 按样本数加权平均。
默认 --precision fp16 对齐当前训练验证的输入/参数精度（buffer 保持 FP32）；
--precision fp32 可用于统一的全精度评估。旧日志的 batch 平均和 DDP 补齐
并未修改，因此本脚本不保证复现旧日志的数值。配置/预处理必须与 checkpoint 一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Select MKL's GNU OpenMP backend before NumPy/PyTorch load native libraries.
# The gnss environment's Intel backend fails during plotting with missing
# __kmpc_global_thread_num / omp_get_num_procs symbols.
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')

import numpy as np
import pandas as pd
import torch
import zarr

# train_FSDP 在模块级 wandb.init()，评估脚本只是借用它的 process_bg/process_obs，
# 不要因此新建 wandb run。
os.environ.setdefault('WANDB_MODE', 'disabled')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import importlib  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from main.model import AssimilationNetv6  # noqa: E402
from main.utils import (apply_overrides, run_dir,  # noqa: E402
                       station_halo_mask)
from main.utils.utils_data import LABEL_IDX, MyDataset  # noqa: E402


class EvaluationDataset(MyDataset):
    """Keep reader semantics, but never invoke MyDataset's random retry loop."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        requested = self.dates
        checks = {
            'era5': requested.isin(self.read_era5.dates),
            'fcst': (requested - pd.Timedelta(hours=self.read_fcst.fcst_step))
                    .isin(self.read_fcst.dates),
        }
        obs_reader = getattr(self, 'read_obs', None)
        if obs_reader is not None:
            checks['obs'] = requested.isin(obs_reader.dates)
            if obs_reader.add_fuxi_ztd:
                checks['fuxi_ztd'] = requested.isin(obs_reader.fx_times)
        valid = np.logical_and.reduce(list(checks.values()))
        self.excluded_dates = [
            {'date': date.isoformat(),
             'missing': [name for name, available in checks.items() if not available[i]]}
            for i, date in enumerate(requested) if not valid[i]
        ]
        self.requested_samples = len(requested)
        self.dates = requested[valid]
        if not len(self.dates):
            raise ValueError('指定日期范围内没有完整的评估样本，请检查数据时间轴')
        if not self.dates.is_unique:
            raise ValueError('评估日期存在重复')
        print(f'[Eval Data] requested={len(requested)}, valid={len(self.dates)}, '
              f'excluded={len(self.excluded_dates)}; no replacement / no padding')
        for item in self.excluded_dates[:10]:
            print(f'  excluded {item["date"]}: {", ".join(item["missing"])}')

    def __getitem__(self, idx):
        sample = self.prepare_data(idx)
        if any(value is None for value in sample):
            raise RuntimeError(f'评估样本 {self.dates[idx]} 读取失败；停止评估，不随机替换')
        return sample


class SampleMeanLoss:
    """Batch loss already averages over B; weight the last partial batch by B."""

    def __init__(self):
        self.bg_sum = self.analysis_sum = 0.0
        self.n_samples = self.n_batches = 0

    def update(self, background, analysis, batch_size):
        bg, an = float(background), float(analysis)
        if batch_size <= 0 or not np.isfinite([bg, an]).all():
            raise ValueError('评估 loss 非有限值或 batch_size 非正数，停止写出结果')
        self.bg_sum += bg * batch_size
        self.analysis_sum += an * batch_size
        self.n_samples += batch_size
        self.n_batches += 1

    def means(self):
        if not self.n_samples:
            raise ValueError('没有已评估的样本')
        return self.bg_sum / self.n_samples, self.analysis_sum / self.n_samples


def set_eval_precision(model, precision):
    # Match FSDP(param_dtype=fp16, buffer_dtype=fp32), not model.half(), which
    # would also cast floating-point buffers. No autocast is used in training.
    dtype = {'fp16': torch.float16, 'fp32': torch.float32}[precision]
    model.float()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
    return dtype


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def evaluation_metadata(cfg, args, ds, model, mean_std_dict, ckpt, n_samples):
    # Store resolved values, not just CLI overrides. Source hashes capture local
    # reader/preprocessing edits that are not represented by a config switch.
    keys = ('model_id', 'era5_dir', 'fcst_dir', 'fcst_step', 'obs_dir', 'obs_stat_dir',
            'obs_frames', 'obs_frame_minutes', 'obs_channum', 'add_fuxi_ztd',
            'ztd_fuxi_zarr', 'zero_obs', 'remove_obs_branch', 'use_ztd_innovation',
            'model_bg_chans', 'model_obs_chans', 'model_obs_frames',
            'model_embed_dim', 'model_depth', 'grid_hw', 'lat', 'lon',
            'loss_mask', 'loss_halo_cells', 'dates_train_range', 'dates_val_range',
            'dates_test_range', 'rand_seed')
    paths = [Path(__file__), HERE / 'train_FSDP.py',
             HERE / 'main/utils/utils_data.py', HERE / 'main/model/assimilation.py',
             HERE / 'main/model/build_optimizer.py', HERE / 'main/utils/configuration.py']
    if getattr(cfg, '__file__', None):
        paths.append(Path(cfg.__file__))
    stat = ckpt.stat()
    return {
        'schema_version': 2,
        'loss_reduction': 'sample_mean',
        'sampler': 'sequential_no_padding',
        'missing_sample_policy': 'filter_missing_times_then_fail_on_read_error',
        'observation_window_policy': 'unchanged_reader_semantics_nan_padding_allowed',
        'precision': args.precision,
        'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
        'buffer_dtypes': sorted({str(b.dtype) for b in model.buffers()}),
        'channel_mae_precision': 'float32_after_input_cast',
        'batch_size': args.batch_size if args.batch_size is not None else cfg.batch_size,
        'num_workers': args.num_workers if args.num_workers is not None else cfg.num_workers,
        'max_batches': args.max_batches,
        'partial_evaluation': n_samples != len(ds),
        'n_requested_samples': ds.requested_samples,
        'n_valid_samples': len(ds),
        'excluded_dates': ds.excluded_dates,
        'evaluated_dates': [date.isoformat() for date in ds.dates[:n_samples]],
        'configs_module': args.configs,
        'cli_overrides': ([f'model_id={args.model_id}'] if args.model_id else []) + args.set,
        'resolved_config': {key: json_value(getattr(cfg, key)) for key in keys if hasattr(cfg, key)},
        'obs_normalization': json_value(mean_std_dict),
        'loss_class': f'{type(cfg.loss_fn).__module__}.{type(cfg.loss_fn).__qualname__}',
        'loss_lat': json_value(getattr(cfg.loss_fn, 'lat', None)),
        'loss_obs_rect': json_value(getattr(cfg.loss_fn, 'obs_rect', None)),
        'loss_cell_mask': json_value(getattr(cfg.loss_fn, 'cell_mask', None)),
        'label_channel_indices': LABEL_IDX.tolist(),
        'checkpoint_size_bytes': stat.st_size,
        'checkpoint_mtime_ns': stat.st_mtime_ns,
        'source_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }


def pick_checkpoint(cfg, explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    mdir = Path(run_dir(cfg)) / 'model'
    if (mdir / 'best.pth').exists():          # 只存 val-best 时的固定文件名
        return mdir / 'best.pth'
    cks = sorted(mdir.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    if not cks:
        raise SystemExit(f'{mdir} 里没有 .pth，请用 --checkpoint 指定')
    return cks[-1]


def load_model(cfg, ckpt: Path, device):
    model = AssimilationNetv6(bg_chans=cfg.model_bg_chans,
                              obs_chans=cfg.model_obs_chans,
                              obs_frames=cfg.model_obs_frames,
                              embed_dim=cfg.model_embed_dim,
                              depth=cfg.model_depth).to(device)
    try:
        payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location='cpu')
    sd = payload['model'] if isinstance(payload, dict) and 'model' in payload else payload
    # FSDP / DDP 的前缀去掉
    sd = {k.replace('_fsdp_wrapped_module.', '').replace('module.', ''): v
          for k, v in sd.items()}
    # Incomplete loading would silently evaluate random/untrained parameters.
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--configs', default='configs')
    ap.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=None)
    ap.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16',
                    help='默认 fp16：对齐训练验证的输入/参数精度；fp32：全精度评估')
    ap.add_argument('--max-batches', type=int, default=0, help='>0 时只跑这么多 batch（调试用）')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--model_id', default=None,
                    help='等价于 --set model_id=xxx；决定去哪个 run 目录找 checkpoint')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='覆盖 configs 里的设置，例：--set zero_obs=true')
    args = ap.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        ap.error('--batch-size 必须 > 0')
    if args.num_workers is not None and args.num_workers < 0:
        ap.error('--num-workers 必须 >= 0')
    if args.max_batches < 0:
        ap.error('--max-batches 必须 >= 0')

    cfg = importlib.import_module(args.configs)
    # 必须在 run_dir() / 建 loader 之前生效
    apply_overrides(cfg, ([f'model_id={args.model_id}'] if args.model_id else []) + list(args.set))
    if not torch.cuda.is_available():
        raise SystemExit('数据管道（process_bg/process_obs）写在 CUDA 上，需要一张 GPU')
    device = torch.device('cuda:0')
    rank = 0
    # Delay the training import until actual evaluation: --help and CPU unit
    # tests should not initialize wandb even in disabled mode.
    from train_FSDP import process_bg, process_obs
    torch.manual_seed(int(getattr(cfg, 'rand_seed', 2000)))

    # halo 掩膜（站点格 + 周围 loss_halo_cells 格）：**总是**算出来用于报告；
    # 是否把它用进损失，取决于 cfg.loss_mask（要和训练时一致）。
    halo = station_halo_mask(cfg)
    print('[halo] loss_halo_cells=%d: %d/%d 格'
          % (int(getattr(cfg, 'loss_halo_cells', 3)), int(halo.sum()), halo.size))
    if str(getattr(cfg, 'loss_mask', 'none')).lower() == 'station_halo':
        cfg.loss_fn.set_cell_mask(halo)
        print('[Loss] 损失只在上面这个 halo 区域内计算（与训练一致）')
    else:
        cfg.loss_fn.set_cell_mask(None)
        print('[Loss] loss_mask=%r -> 损失在全网格计算' % getattr(cfg, 'loss_mask', None))

    ckpt = pick_checkpoint(cfg, args.checkpoint)
    print(f'[cfg] {args.configs} | split={args.split} | ckpt={ckpt}')

    dates_range = {'train': cfg.dates_train_range,
                   'val': cfg.dates_val_range,
                   'test': cfg.dates_test_range}[args.split]
    ds = EvaluationDataset(era5_dir=cfg.era5_dir, fcst_dir=cfg.fcst_dir,
                                  fcst_step=cfg.fcst_step, obs_dir=cfg.obs_dir,
                                  obs_frames=cfg.obs_frames, obs_stat_dir=cfg.obs_stat_dir,
                                  obs_channum=cfg.obs_channum, dates_range=dates_range,
                                  obs_frame_minutes=cfg.obs_frame_minutes,
                                  grid_hw=cfg.grid_hw,
                                  add_fuxi_ztd=cfg.add_fuxi_ztd,
                                  ztd_fuxi_zarr=cfg.ztd_fuxi_zarr)
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers
    worker_options = {}
    if num_workers:
        worker_options = {
            'prefetch_factor': getattr(cfg, 'prefetch_factor', 5),
            'multiprocessing_context': getattr(cfg, 'multiprocessing_context', 'forkserver'),
        }
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_size if args.batch_size is None else args.batch_size,
        shuffle=False, drop_last=False, num_workers=num_workers,
        persistent_workers=False, pin_memory=getattr(cfg, 'pin_memory', False),
        **worker_options)
    obs_reader = getattr(ds, 'read_obs', None)
    mean_std_dict = obs_reader.mean_std_dict if obs_reader is not None else {}
    n_chan = len(LABEL_IDX)
    all_ch = np.asarray([str(c) for c in zarr.open(str(cfg.era5_dir), mode='r')['channel'][:]])
    names = [str(c) for c in all_ch[LABEL_IDX]]

    station = ~np.asarray(zarr.open(str(cfg.obs_dir), mode='r')['mask'][:])      # True = 有站
    st_t = torch.as_tensor(station).to(device)
    n_st = int(station.sum())

    model = load_model(cfg, ckpt, device)
    dtype = set_eval_precision(model, args.precision)
    print(f'[Eval] precision={args.precision}, loss=sample_mean, sequential (no DDP padding)')
    print('[Eval] 使用当前配置及预处理代码；请确保它们与 checkpoint 训练时一致。')
    if list(cfg.dates_val_range) == list(cfg.dates_test_range):
        print('[warn] val/test 日期范围相同，不是独立测试集。')

    hl_t = torch.as_tensor(halo).to(device)
    n_hl = int(halo.sum())
    acc = {k: np.zeros(n_chan) for k in
           ('bg', 'an', 'bg_st', 'an_st', 'bg_hl', 'an_hl')}
    n_smp = n_cell = n_cell_st = n_cell_hl = 0
    losses = SampleMeanLoss()

    with torch.no_grad():
        for i, (bf, be, bo) in enumerate(dataloader):
            bf = process_bg(bf, rank, cfg.grid_hw)
            be = process_bg(be, rank, cfg.grid_hw)
            if cfg.model_obs_chans > 0:
                bo = process_obs(bo, mean_std_dict, rank, cfg).to(dtype=dtype)
            bf = bf.to(dtype=dtype)
            be = be.to(dtype=dtype)
            out = model(bf, bo) if cfg.model_obs_chans > 0 else model(bf)
            # Keep weighted loss arithmetic identical to training: subtraction
            # happens in FP16 in fp16 mode, before latitude weighting promotes it.
            losses.update(cfg.loss_fn(bf, be), cfg.loss_fn(out, be), int(bf.shape[0]))
            lab = be.float()

            # (B,1,C,H,W) -> (B,C,H,W)：T 维是 1，直接去掉
            d_bg = (bf.float()[:, 0] - lab[:, 0]).abs()
            d_an = (out.float()[:, 0] - lab[:, 0]).abs()
            acc['bg'] += d_bg.sum(dim=(0, 2, 3)).cpu().numpy()
            acc['an'] += d_an.sum(dim=(0, 2, 3)).cpu().numpy()
            acc['bg_st'] += (d_bg * st_t).sum(dim=(0, 2, 3)).cpu().numpy()
            acc['an_st'] += (d_an * st_t).sum(dim=(0, 2, 3)).cpu().numpy()
            acc['bg_hl'] += (d_bg * hl_t).sum(dim=(0, 2, 3)).cpu().numpy()
            acc['an_hl'] += (d_an * hl_t).sum(dim=(0, 2, 3)).cpu().numpy()

            n_smp += int(bf.shape[0])
            n_cell += int(bf.shape[0] * d_bg.shape[2] * d_bg.shape[3])   # B x H x W
            n_cell_st += int(bf.shape[0] * n_st)
            n_cell_hl += int(bf.shape[0] * n_hl)

            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(dataloader)} batch', flush=True)
            if args.max_batches and (i + 1) >= args.max_batches:
                break

    loss_bg, loss_an = losses.means()
    n_batch = losses.n_batches

    n_cell = max(n_cell, 1)
    n_cell_st = max(n_cell_st, 1)
    mae_bg = acc['bg'] / n_cell
    mae_an = acc['an'] / n_cell
    mae_bg_st = acc['bg_st'] / n_cell_st
    mae_an_st = acc['an_st'] / n_cell_st
    mae_bg_hl = acc['bg_hl'] / max(n_cell_hl, 1)
    mae_an_hl = acc['an_hl'] / max(n_cell_hl, 1)

    imp = 100.0 * (mae_bg - mae_an) / mae_bg
    imp_st = 100.0 * (mae_bg_st - mae_an_st) / mae_bg_st
    imp_hl = 100.0 * (mae_bg_hl - mae_an_hl) / mae_bg_hl

    print(f'\n样本 {n_smp}（{n_batch} batch），网格 {cfg.grid_hw}，站格 {n_st}')
    print(f'损失函数口径（纬度加权，{n_chan} 通道，按样本平均）:  背景 {loss_bg:.8f}   '
          f'分析 {loss_an:.8f}   '
          f'提升 {100 * (loss_bg - loss_an) / max(loss_bg, 1e-9):+.3f}%')
    print(f'逐通道平均 MAE（全网格）:            背景 {mae_bg.mean():.5f}   '
          f'分析 {mae_an.mean():.5f}   提升 {100 * (mae_bg.mean() - mae_an.mean()) / mae_bg.mean():+.3f}%')
    print(f'逐通道平均 MAE（站格）  :            背景 {mae_bg_st.mean():.5f}   '
          f'分析 {mae_an_st.mean():.5f}   提升 '
          f'{100 * (mae_bg_st.mean() - mae_an_st.mean()) / mae_bg_st.mean():+.3f}%')
    print(f'逐通道平均 MAE（halo{int(getattr(cfg, "loss_halo_cells", 3))}） :            '
          f'背景 {mae_bg_hl.mean():.5f}   分析 {mae_an_hl.mean():.5f}   提升 '
          f'{100 * (mae_bg_hl.mean() - mae_an_hl.mean()) / mae_bg_hl.mean():+.3f}%')

    print(f'\n{"通道":>9}{"bg_全域":>11}{"an_全域":>11}{"提升%":>9}'
          f'{"bg_halo":>11}{"an_halo":>11}{"提升%":>9}{"bg_站格":>11}{"提升%":>9}')
    for j, nm in enumerate(names):
        print(f'{nm:>9}{mae_bg[j]:>11.5f}{mae_an[j]:>11.5f}{imp[j]:>+9.3f}'
              f'{mae_bg_hl[j]:>11.5f}{mae_an_hl[j]:>11.5f}{imp_hl[j]:>+9.3f}'
              f'{mae_bg_st[j]:>11.5f}{imp_st[j]:>+9.3f}')

    out_dir = Path(args.out_dir) if args.out_dir else Path(run_dir(cfg))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f'eval_channels_{args.split}.csv'
    with open(csv, 'w') as fh:
        fh.write('channel,mae_bg,mae_analysis,improve_pct,'
                 'mae_bg_halo,mae_analysis_halo,improve_pct_halo,'
                 'mae_bg_station,mae_analysis_station,improve_pct_station\n')
        for j, nm in enumerate(names):
            fh.write(f'{nm},{mae_bg[j]:.6f},{mae_an[j]:.6f},{imp[j]:.4f},'
                     f'{mae_bg_hl[j]:.6f},{mae_an_hl[j]:.6f},{imp_hl[j]:.4f},'
                     f'{mae_bg_st[j]:.6f},{mae_an_st[j]:.6f},{imp_st[j]:.4f}\n')
    print(f'\n[out] {csv}')

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=130)
    ax = axes[0]
    ax.bar(np.arange(n_chan) - 0.25, imp, width=0.25, label='whole grid',
           color='tab:blue')
    ax.bar(np.arange(n_chan), imp_hl, width=0.25, label='station halo',
           color='tab:purple')
    ax.bar(np.arange(n_chan) + 0.25, imp_st, width=0.25, label='station cells',
           color='tab:red')
    ax.axhline(0, c='k', lw=.8)
    ax.set_xticks(np.arange(n_chan))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel('analysis vs background, MAE improvement (%)')
    ax.set_title(f'per-channel improvement ({args.split} split, {n_smp} samples)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis='y')

    ax = axes[1]
    x = np.arange(n_chan)
    ax.plot(x, mae_bg, 'o-', ms=3, lw=.8, label='background (FuXi)', color='tab:orange')
    ax.plot(x, mae_an, 's-', ms=3, lw=.8, label='analysis (model)', color='tab:green')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel('MAE vs ERA5')
    ax.set_title('per-channel MAE')
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    png = out_dir / f'eval_channels_{args.split}.png'
    fig.savefig(png)
    print(f'[out] {png}')

    with open(out_dir / f'eval_summary_{args.split}.json', 'w') as fh:
        json.dump({'split': args.split, 'checkpoint': str(ckpt), 'n_samples': n_smp,
                   'loss_bg': loss_bg, 'loss_analysis': loss_an,
                   'n_batches': n_batch,
                   'evaluation': evaluation_metadata(cfg, args, ds, model, mean_std_dict,
                                                     ckpt, n_smp),
                   'mae_bg_mean': float(mae_bg.mean()), 'mae_analysis_mean': float(mae_an.mean()),
                   'mae_bg_station_mean': float(mae_bg_st.mean()),
                   'mae_analysis_station_mean': float(mae_an_st.mean()),
                   'mae_bg_halo_mean': float(mae_bg_hl.mean()),
                   'mae_analysis_halo_mean': float(mae_an_hl.mean()),
                   'improve_pct_halo_mean': float(imp_hl.mean()),
                   'improve_pct_mean': float(imp.mean()),
                   'improve_pct_station_mean': float(imp_st.mean()),
                   'improve_pct_loss_weighted':
                       float(100 * (loss_bg - loss_an) / max(loss_bg, 1e-9))},
                  fh, indent=2)
    print(f'[out] {out_dir / f"eval_summary_{args.split}.json"}')


if __name__ == '__main__':
    main()
