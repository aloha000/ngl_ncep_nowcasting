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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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
from main.utils import (apply_overrides, build_dataloader, run_dir,  # noqa: E402
                       station_halo_mask)
from main.utils.utils_data import LABEL_IDX  # noqa: E402
from train_FSDP import process_bg, process_obs  # noqa: E402


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
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'[warn] state_dict 不完全匹配：缺 {len(missing)} 个、多 {len(unexpected)} 个'
              f'（缺的前 3 个：{list(missing)[:3]}）')
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
    ap.add_argument('--max-batches', type=int, default=0, help='>0 时只跑这么多 batch（调试用）')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--model_id', default=None,
                    help='等价于 --set model_id=xxx；决定去哪个 run 目录找 checkpoint')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='覆盖 configs 里的设置，例：--set zero_obs=true')
    args = ap.parse_args()

    cfg = importlib.import_module(args.configs)
    # 必须在 run_dir() / 建 loader 之前生效
    apply_overrides(cfg, ([f'model_id={args.model_id}'] if args.model_id else []) + list(args.set))
    if not torch.cuda.is_available():
        raise SystemExit('数据管道（process_bg/process_obs）写在 CUDA 上，需要一张 GPU')
    device = torch.device('cuda:0')
    rank = 0

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
    dataloader = build_dataloader(world_size=1, rank=0,
                                  era5_dir=cfg.era5_dir, fcst_dir=cfg.fcst_dir,
                                  fcst_step=cfg.fcst_step, obs_dir=cfg.obs_dir,
                                  obs_frames=cfg.obs_frames, obs_stat_dir=cfg.obs_stat_dir,
                                  obs_channum=cfg.obs_channum, dates_range=dates_range,
                                  batch_size=args.batch_size or cfg.batch_size,
                                  num_workers=cfg.num_workers if args.num_workers is None else args.num_workers,
                                  persistent_workers=False, shuffle=False,
                                  obs_frame_minutes=cfg.obs_frame_minutes,
                                  grid_hw=cfg.grid_hw)
    ds = dataloader.dataset
    mean_std_dict = ds.read_obs.mean_std_dict
    n_chan = len(LABEL_IDX)
    all_ch = np.asarray([str(c) for c in zarr.open(str(cfg.era5_dir), 'r')['channel'][:]])
    names = [str(c) for c in all_ch[LABEL_IDX]]

    station = ~np.asarray(zarr.open(str(cfg.obs_dir), 'r')['mask'][:])      # True = 有站
    st_t = torch.as_tensor(station).to(device)
    n_st = int(station.sum())

    model = load_model(cfg, ckpt, device)
    lut = torch.cos(torch.deg2rad(torch.as_tensor(np.asarray(cfg.lat, dtype='float32'),
                                                  device=device), ))
    lut = (lut / lut.mean()).view(1, 1, -1, 1)                              # 纬度权重

    hl_t = torch.as_tensor(halo).to(device)
    n_hl = int(halo.sum())
    acc = {k: np.zeros(n_chan) for k in
           ('bg', 'an', 'bg_st', 'an_st', 'bg_hl', 'an_hl')}
    n_smp = n_cell = n_cell_st = n_cell_hl = 0
    loss_bg, loss_an, n_batch = 0.0, 0.0, 0

    with torch.no_grad():
        for i, (bf, be, bo) in enumerate(dataloader):
            bf = process_bg(bf, rank, cfg.grid_hw)
            be = process_bg(be, rank, cfg.grid_hw)
            bo = process_obs(bo, mean_std_dict, rank, cfg)
            out = model(bf.float(), bo.float())
            lab = be.float()

            # (B,1,C,H,W) -> (B,C,H,W)：T 维是 1，直接去掉
            d_bg = (bf.float()[:, 0] - lab[:, 0]).abs()
            d_an = (out[:, 0] - lab[:, 0]).abs()
            acc['bg'] += d_bg.sum(dim=(0, 2, 3)).cpu().numpy()
            acc['an'] += d_an.sum(dim=(0, 2, 3)).cpu().numpy()
            acc['bg_st'] += (d_bg * st_t).sum(dim=(0, 2, 3)).cpu().numpy()
            acc['an_st'] += (d_an * st_t).sum(dim=(0, 2, 3)).cpu().numpy()
            acc['bg_hl'] += (d_bg * hl_t).sum(dim=(0, 2, 3)).cpu().numpy()
            acc['an_hl'] += (d_an * hl_t).sum(dim=(0, 2, 3)).cpu().numpy()

            loss_bg += float(cfg.loss_fn(bf.float(), lab))
            loss_an += float(cfg.loss_fn(out, lab))
            n_batch += 1
            n_smp += int(bf.shape[0])
            n_cell += int(bf.shape[0] * d_bg.shape[2] * d_bg.shape[3])   # B x H x W
            n_cell_st += int(bf.shape[0] * n_st)
            n_cell_hl += int(bf.shape[0] * n_hl)

            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(dataloader)} batch', flush=True)
            if args.max_batches and (i + 1) >= args.max_batches:
                break

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
    print(f'损失函数口径（纬度加权，70 通道）:  背景 {loss_bg / max(n_batch,1):.5f}   '
          f'分析 {loss_an / max(n_batch,1):.5f}   '
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
                   'loss_bg': loss_bg / max(n_batch, 1), 'loss_analysis': loss_an / max(n_batch, 1),
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
