#!/usr/bin/env python3
"""Decompose a saved eval_summary_test.json loss and evaluate physical-unit MAE.

Uses the saved resolved configuration, valid dates, mask, precision and checkpoint.
Linear variables are unstandardized using ERA5 std; precipitation is inverted
pointwise with expm1 before taking absolute errors. No clipping of precipitation.
Physical MAEs with different units must not be averaged into one scalar.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
os.environ['WANDB_MODE'] = 'disabled'
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval_results import EvaluationDataset, load_model, set_eval_precision
from main.model import mae
from main.utils.utils_data import LABEL_IDX


def unit_scale(channel):
    if channel.startswith('z'):
        return 'gpm', 1 / 9.80665
    if channel == 'msl':
        return 'hPa', .01
    if channel == 'era5_tp':
        return 'mm', 1.
    if channel.startswith('t'):
        return 'degC', 1.
    if channel.startswith('r'):
        return 'percentage_points', 1.
    if channel.startswith(('u', 'v')):
        return 'm/s', 1.
    raise ValueError(f'Unknown channel: {channel}')


class ChannelAccumulator:
    """Spatial weighted error sums; divide by actual samples, never batches."""
    def __init__(self, weights, channels, mean, std):
        self.weights = weights.float()
        self.mean = torch.as_tensor(mean, device=weights.device, dtype=torch.float64)
        self.std = torch.as_tensor(std, device=weights.device, dtype=torch.float64)
        self.scale = torch.tensor([unit_scale(c)[1] for c in channels],
                                  device=weights.device, dtype=torch.float64)
        self.tp = channels.index('era5_tp')
        self.norm_sum = torch.zeros(len(channels), device=weights.device, dtype=torch.float64)
        self.phys_sum = torch.zeros_like(self.norm_sum)
        self.n_samples = 0

    def update(self, prediction, target):
        # Loss decomposition preserves the training/eval FP16 subtraction.
        error_loss = (prediction - target).abs().float()
        self.norm_sum += (error_loss * self.weights).sum((0, 1, 3, 4), dtype=torch.float64)
        # Physical-unit MAE uses higher precision on the same quantized fields.
        # This avoids adding an extra FP16 subtraction rounding to unit conversion.
        error = (prediction.double() - target.double()).abs()
        physical_sum = (error * self.weights.double()).sum((0, 1, 3, 4)) * self.std * self.scale
        pred_tp = torch.expm1(prediction[:, :, self.tp].double() * self.std[self.tp] + self.mean[self.tp])
        true_tp = torch.expm1(target[:, :, self.tp].double() * self.std[self.tp] + self.mean[self.tp])
        physical_sum[self.tp] = ((pred_tp - true_tp).abs() * self.weights.double()).sum()
        if not torch.isfinite(physical_sum).all():
            raise ValueError('Non-finite physical errors; check normalization / predictions')
        self.phys_sum += physical_sum
        self.n_samples += int(prediction.shape[0] * prediction.shape[1])

    def results(self):
        if not self.n_samples:
            raise ValueError('No evaluated samples')
        # Match cfg.loss_fn's FP32 weight denominator for loss decomposition.
        norm = self.norm_sum / (self.n_samples * self.weights.sum().double())
        physical = self.phys_sum / (self.n_samples * self.weights.double().sum())
        return norm.cpu().numpy(), physical.cpu().numpy()


def plot_channels(table, out_dir):
    groups = [([f'{v}{p}' for p in (50,100,150,200,250,300,400,500,600,700,850,925,1000)], title)
              for v, title in (('z', 'Geopotential height'), ('t', 'Temperature'),
                               ('u', 'Zonal wind'), ('v', 'Meridional wind'), ('r', 'Relative humidity'))]
    groups += [(['t2m'], '2m temperature'), (['u10', 'v10'], '10m wind components'),
               (['msl'], 'Mean sea-level pressure'), (['era5_tp'], 'Precipitation (pointwise inverse log1p)')]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), layout='constrained')
    lookup = table.set_index('channel')
    for ax, (names, title) in zip(axes.ravel(), groups):
        block = lookup.loc[names]
        y = np.arange(len(names))
        ax.barh(y-.17, block.mae_bg_physical, height=.32, color='#e58b27', label='FuXi-24h')
        ax.barh(y+.17, block.mae_analysis_physical, height=.32, color='#1976c9', label='Analysis')
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('MAE (' + block.unit.iloc[0] + ')')
        ax.grid(axis='x', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Per-channel physical MAE vs ERA5 | best.pth | 1092 test times\n'
                 'Station halo + cosine-latitude weights, same spatial scope as loss_analysis', fontsize=14)
    for suffix in ('png', 'pdf'):
        fig.savefig(out_dir / f'channel_mae_physical.{suffix}', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-summary', type=Path, required=True)
    ap.add_argument('--stats-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()
    if args.num_workers < 0:
        ap.error('--num-workers must be >= 0')
    out_dir = args.out_dir or args.reference_summary.parent/'physical_channel_mae_test'
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty {out_dir}')
    ref = json.loads(args.reference_summary.read_text())
    ev = ref['evaluation']
    if ev['loss_class'] != 'main.model.build_optimizer.mae' or ev['loss_obs_rect'] is not None:
        raise ValueError('Only project MAE with no obs_rect is supported')
    if ev['partial_evaluation'] or ev['loss_reduction'] != 'sample_mean':
        raise ValueError('Reference must be a complete, sample-averaged evaluation')
    cfg = SimpleNamespace(**ev['resolved_config'])
    cfg.loss_fn = mae(lat=ev['loss_lat'])
    mask = ev['loss_cell_mask']
    cfg.loss_fn.set_cell_mask(mask)
    checkpoint = Path(ref['checkpoint'])
    if checkpoint.stat().st_size != ev['checkpoint_size_bytes'] or checkpoint.stat().st_mtime_ns != ev['checkpoint_mtime_ns']:
        raise ValueError('Checkpoint changed since the reference evaluation')
    for path, expected in ev['source_sha256'].items():
        if Path(path).name in ('train_FSDP.py', 'utils_data.py', 'assimilation.py', 'build_optimizer.py'):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Reference preprocessing/model code changed: {path}')
    mean = np.load(args.stats_dir/'mean_era5.npy').reshape(-1)
    std = np.load(args.stats_dir/'std_era5.npy').reshape(-1)
    if mean.shape != (70,) or std.shape != (70,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid 70-channel mean/std')
    torch.set_num_threads(4)
    torch.manual_seed(int(getattr(cfg, 'rand_seed', 2000)))
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    from train_FSDP import process_bg, process_obs
    torch.backends.cudnn.benchmark = ev['cudnn_benchmark']
    torch.backends.cudnn.deterministic = ev['cudnn_deterministic']
    torch.backends.cuda.matmul.allow_tf32 = ev['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = ev['cudnn_allow_tf32']
    ds = EvaluationDataset(
        era5_dir=cfg.era5_dir, fcst_dir=cfg.fcst_dir, fcst_step=cfg.fcst_step,
        obs_dir=cfg.obs_dir, obs_frames=cfg.obs_frames, obs_stat_dir=cfg.obs_stat_dir,
        obs_channum=cfg.obs_channum, dates_range=getattr(cfg, f'dates_{ref["split"]}_range'),
        obs_frame_minutes=cfg.obs_frame_minutes, grid_hw=cfg.grid_hw,
        add_fuxi_ztd=cfg.add_fuxi_ztd, ztd_fuxi_zarr=cfg.ztd_fuxi_zarr)
    if [d.isoformat() for d in ds.dates] != ev['evaluated_dates']:
        raise ValueError('Evaluated dates differ from reference')
    channels = [str(c) for c in ds.read_era5.store['channel'][:][LABEL_IDX]]
    expected_channels = [f'{v}{p}' for v in ('z','t','u','v','r')
                         for p in (50,100,150,200,250,300,400,500,600,700,850,925,1000)]
    if channels != expected_channels + ['t2m','u10','v10','msl','era5_tp']:
        raise ValueError('Unexpected channel order for ERA5 normalization')
    loader = torch.utils.data.DataLoader(ds, batch_size=ev['batch_size'], shuffle=False,
        num_workers=args.num_workers, **({'multiprocessing_context':'forkserver', 'prefetch_factor':2}
                                        if args.num_workers else {}))
    model = load_model(cfg, checkpoint, 'cuda:0')
    dtype = set_eval_precision(model, ev['precision'])
    lat = cfg.loss_fn.lat.to('cuda:0')
    wlat = torch.cos(torch.deg2rad(lat))[:, None]
    wlat /= wlat.mean()
    weight = wlat * (torch.ones(cfg.grid_hw, device='cuda:0') if mask is None
                     else torch.as_tensor(mask, device='cuda:0', dtype=torch.float32))
    bg_acc, an_acc = [ChannelAccumulator(weight, channels, mean, std) for _ in range(2)]
    direct_bg = direct_an = 0.
    start = time.monotonic()
    print(f'[reference] bg={ref["loss_bg"]:.10f}, analysis={ref["loss_analysis"]:.10f}', flush=True)
    with torch.inference_mode():
        for i, (bf, be, bo) in enumerate(loader):
            bg = process_bg(bf, 0, cfg.grid_hw).to(dtype)
            target = process_bg(be, 0, cfg.grid_hw).to(dtype)
            obs = process_obs(bo, ds.read_obs.mean_std_dict, 0, cfg).to(dtype)
            output = model(bg, obs)
            bg_acc.update(bg, target)
            an_acc.update(output, target)
            direct_bg += float(cfg.loss_fn(bg, target)) * len(bf)
            direct_an += float(cfg.loss_fn(output, target)) * len(bf)
            if i == 0 or (i+1) % 10 == 0 or an_acc.n_samples == len(ds):
                print(f'[eval] {an_acc.n_samples}/{len(ds)}, elapsed={time.monotonic()-start:.1f}s', flush=True)
    norm_bg, physical_bg = bg_acc.results()
    norm_an, physical_an = an_acc.results()
    direct_bg /= len(ds)
    direct_an /= len(ds)
    np.testing.assert_allclose([norm_bg.mean(), norm_an.mean()], [direct_bg, direct_an], atol=1e-7, rtol=0)
    table = pd.DataFrame({'channel':channels, 'unit':[unit_scale(c)[0] for c in channels],
        'mae_bg_normalized_loss':norm_bg, 'mae_analysis_normalized_loss':norm_an,
        'mae_bg_physical':physical_bg, 'mae_analysis_physical':physical_an,
        'delta_analysis_minus_bg':physical_an-physical_bg,
        'improve_pct':100*(physical_bg-physical_an)/physical_bg,
        'normalized_loss_improve_pct':100*(norm_bg-norm_an)/norm_bg,
        'normalization_std':std})
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty {out_dir}')
    table.to_csv(out_dir/'channel_mae_physical.csv', index=False)
    np.savez_compressed(out_dir/'channel_error_sums.npz', norm_bg_sum=bg_acc.norm_sum.cpu().numpy(),
                        norm_an_sum=an_acc.norm_sum.cpu().numpy(), phys_bg_sum=bg_acc.phys_sum.cpu().numpy(),
                        phys_an_sum=an_acc.phys_sum.cpu().numpy(), weights=weight.cpu().numpy(),
                        mean=mean, std=std, n_samples=len(ds), channels=np.array(channels))
    summary = {'reference_summary':str(args.reference_summary.resolve()),
        'reference_summary_sha256':hashlib.sha256(args.reference_summary.read_bytes()).hexdigest(),
        'reference_loss_bg':ref['loss_bg'], 'reference_loss_analysis':ref['loss_analysis'],
        'recomputed_loss_bg':direct_bg, 'recomputed_loss_analysis':direct_an,
        'decomposed_loss_bg':float(norm_bg.mean()), 'decomposed_loss_analysis':float(norm_an.mean()),
        'difference_from_reference':direct_an-ref['loss_analysis'],
        'reference_matches_atol_1e-6':bool(abs(direct_an-ref['loss_analysis']) < 1e-6),
        'n_samples':len(ds), 'n_cells':int((weight>0).sum()), 'precision':ev['precision'],
        'scope':'same spatial mask and cosine-latitude weights as reference loss',
        'stats_dir':str(args.stats_dir.resolve()),
        'stats_sha256':{n:hashlib.sha256((args.stats_dir/n).read_bytes()).hexdigest()
                        for n in ('mean_era5.npy','std_era5.npy')},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'precipitation':'expm1(z*std+mean), mm; pointwise inversion; no clipping',
        'arithmetic':'normalized loss preserves FP16 subtraction; physical MAE uses FP64 differences on the same quantized fields',
        'notes':'Different physical units cannot be averaged. Test dates overlap validation.',
        'resolved_config':ev['resolved_config']}
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines = ['# 各通道物理单位 MAE（与 loss_analysis 相同空间权重）', '',
             f'有效样本 {len(ds)} 个，mask 内 {summary["n_cells"]} 个格点，精度 {ev["precision"]}。', '',
             f'重算标准化 loss：FuXi {direct_bg:.10f}，Analysis {direct_an:.10f}。',
             f'原 JSON 的 Analysis loss：{ref["loss_analysis"]:.10f}。', '',
             '正改善率表示模型 MAE 降低；负值表示退化。不同单位不能再合成一个物理 MAE。',
             '温度误差的 K 与 ℃ 数值相同；r 为百分点；z 除以 g0 后为位势高度 gpm。',
             '降水先逐点 expm1 反变换再计算毫米 MAE，不是把对数 MAE 直接乘标准差。',
             'FP16 loss 的减法舍入与高精度物理 MAE 换算存在微小数值区别。',
             '测试集与验证集日期重叠，非独立留出检验。', '',
             '| 通道 | 单位 | FuXi MAE | Analysis MAE | 变化（模型−背景） | 改善率 |',
             '|---|---|---:|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {r.channel} | {r.unit} | {r.mae_bg_physical:.5f} | {r.mae_analysis_physical:.5f} | {r.delta_analysis_minus_bg:+.5f} | {r.improve_pct:+.2f}% |')
    (out_dir/'channel_mae_physical.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot_channels(table, out_dir)
    print(table.to_string(index=False), flush=True)
    print('[out]', out_dir.resolve(), flush=True)
    print('[loss agreement]', summary['difference_from_reference'], flush=True)


if __name__ == '__main__':
    main()
