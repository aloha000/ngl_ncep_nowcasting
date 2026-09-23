"""FSDP training entry point for the GNSS/FuXi -> ERA5 assimilation task.

Adapted from ``xuxiaoze/for_zrx/train_packet/train_FSDP.py``.  The FSDP setup,
mixed precision, warm-up + cosine schedule, checkpointing and evaluation loop
are unchanged; the data layer, grid handling and channel handling were
rewritten for the ``da_ngl`` stores.
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
import json
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from main.model import (AssimilationNetv6, EarlyStopping, IncrementPenalty,
                        ObsConsistencyLoss, StationZTD, WarmupScheduler,
                        build_optimizer, get_parameter_number)
from main.utils import (arch_tag, bg_chans, build_dataloader, checkpoint_file,
                        create_logger, experiment_dir, exp_tag, load_checkpoint,
                        load_obs_debias, log_file, loss_domain_weight,
                        loss_domain_weight_share, loss_region_mask, model_id,
                        obs_chans, station_cell_weight)
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="linan_nilan-shanghai-academy-of-ai4s",
    # Set the wandb project where this run will be logged.
    project="da_ngl",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 'active',
        "architecture": "CNN",
        "dataset": "fuxi",
        "epochs": 36,
    },
)

def init_dist(rank, configs, master_port, world_size, overrides=None):
    xconfig = importlib.import_module(configs)
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(xconfig, key, value)
    if 'model_obs_chans' not in (overrides or {}):
        # obs_mode may have just been overridden -- keep the channel count in sync
        xconfig.model_obs_chans = obs_chans(xconfig)
    if 'model_bg_chans' not in (overrides or {}):
        # include_fuxi_tp may have just been overridden -- keep it in sync too
        xconfig.model_bg_chans = bg_chans(xconfig)
    # 纬度加权开关：--set lat_weight=false 是在 configs import 之后才生效的，
    # 光改 cfg.lat_weight 不会重建已经建好的 loss_fn 实例，所以这里同步一次。
    if hasattr(xconfig, 'loss_fn') and hasattr(xconfig.loss_fn, 'set_lat_weight'):
        xconfig.loss_fn.set_lat_weight(bool(getattr(xconfig, 'lat_weight', True)))

    # per-cell weighting of the label loss: the historical station / no-station
    # split, and -- with loss_domain='station_halo' -- "only inside the station
    # mask dilated by loss_halo_cells".  No-op when every cell ends up at 1.0.
    loss_note = ''
    ws = float(getattr(xconfig, 'loss_station_weight', 1.0))
    wn = float(getattr(xconfig, 'loss_nostation_weight', 1.0))
    domain = str(getattr(xconfig, 'loss_domain', 'full')).lower()
    region = loss_region_mask(xconfig)
    if (ws, wn) != (1.0, 1.0) or domain != 'full':
        cw = loss_domain_weight(xconfig)
        xconfig.loss_fn.set_cell_weight(cw)
        _lat = np.asarray(getattr(xconfig, 'lat', []), dtype=float)
        _use_latw = bool(getattr(xconfig, 'lat_weight', True))
        if _lat.size and _use_latw:         # latitude-weighted share of the loss
            _wl = np.cos(np.deg2rad(_lat))[:, None] * np.ones((1, cw.shape[1]))
            _st = cw == ws
            share = float((cw[_st] * _wl[_st]).sum() / (cw * _wl).sum())
        else:
            share = float(cw[cw == ws].sum() / cw.sum())
        loss_note = (f'\n[Loss] domain={domain}'
                     + (f' halo={int(getattr(xconfig, "loss_halo_cells", 0) or 0)}'
                        if domain == 'station_halo' else '') +
                     f': region {int(region.sum())}/{region.size} cells '
                     f'({100 * region.sum() / region.size:.1f}%), weights '
                     f'in={ws:g} out={wn:g} -> region holds {100 * share:.1f}% '
                     f'of the loss weight'
                     + ('' if _use_latw else '  [lat_weight=off -> uniform cells]'))

    if rank == 0:
        os.makedirs(experiment_dir(xconfig), exist_ok=True)
        os.makedirs(os.path.dirname(log_file(xconfig)), exist_ok=True)

    xconfig.logger = create_logger(
        xconfig.work_dir, 'Train',
        log_dir=os.path.dirname(log_file(xconfig)),
        file_name=os.path.basename(log_file(xconfig))[:-len('.log')])

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    seed = xconfig.rand_seed + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if rank == 0:
        xconfig.logger.info(
            f'[Configs]: {configs} | world size {world_size} | seed {seed}\n'
            f'[Run]: model_id={model_id(xconfig)} 实验配置={exp_tag(xconfig)} '
            f'模型配置={arch_tag(xconfig)}\n'
            f'[Results]: {experiment_dir(xconfig)}\n'
            f'[Checkpoint]: {checkpoint_file(xconfig)}\n'
            f'[Log]: {log_file(xconfig)}{loss_note}')
    torch.cuda.set_device(rank)
    xconfig.rank = rank
    return xconfig


def process_bg(data, rank, hw=None):
    """Move the (B,T,C,H,W) field to the GPU (optionally resize in space)."""
    data = data.to(rank).float()
    if hw is not None and tuple(data.shape[-2:]) != tuple(hw):
        B, T, C, H, W = data.shape
        data = torch.nn.functional.interpolate(
            data.reshape(B, T * C, H, W), size=hw, mode="bilinear", align_corners=False)
        data = data.reshape(B, T, C, hw[0], hw[1])
    return data


def process_obs(obs, cfg, rank):
    """Build the observation tensor from the raw 5-minute ZTD window.

    ``obs`` is (B, T, 1, H, W) with NaN where a cell has no station.  The
    validity mask (per frame), the optional ``lat``/``lon`` channels are added
    and everything is zeroed where the frame is invalid -- exactly the masking
    convention of the original packet.

    With ``cfg.zero_obs`` the ZTD values are replaced by 0 -- the training-mean
    ZTD, i.e. the same value a station-less cell carries -- *after* the
    validity mask is derived.  The station geometry/availability survives, the
    ZTD content does not.  This is the "no GNSS" ablation; doing it here means
    training and ``plot_results.py`` (which imports this function) see the same
    observations.
    """
    obs = obs.to(rank).float()
    finite = torch.isfinite(obs)
    data = torch.nan_to_num(obs)
    mask = finite.any(dim=2, keepdim=True).float()      # (B,T,1,H,W)

    if getattr(cfg, 'zero_obs', False):
        data = torch.zeros_like(data)                   # no-GNSS ablation

    B, T, _, H, W = data.shape
    extra = []
    if getattr(cfg, 'obs_add_mask', True):
        extra.append(mask)
    if getattr(cfg, 'obs_add_latlon', True):
        lat = torch.as_tensor(np.asarray(cfg.lat, dtype=np.float32), device=rank)
        lon = torch.as_tensor(np.asarray(cfg.lon, dtype=np.float32), device=rank)
        extra.append(lat.view(1, 1, 1, H, 1).expand(B, T, 1, H, W))
        extra.append(lon.view(1, 1, 1, 1, W).expand(B, T, 1, H, W))
    if extra:
        data = torch.cat([data] + extra, dim=2)

    return data * mask


def obs_at_valid_time(obs_raw, cfg, rank):
    """(B, H, W) observed GNSS ZTD [mm] at the analysis time ``T``.

    ``obs_raw`` is the dataset's standardised window ``(B, T, 1, H, W)`` with NaN
    outside the station cells; frame ``-1`` is the valid time because the window
    is ``[T - (frames-1) * 5 min, T]``.  NaN is preserved so the consistency loss
    can mask it instead of counting it as a zero error.
    """
    z = obs_raw[:, -1, 0].to(rank).float()
    return z * float(cfg.ztd_train_std) + float(cfg.ztd_train_mean)


def build_obs_loss(cfg, rank):
    """Observation-consistency loss (HANDOFF 11.8 item 1); None when disabled.

    Also loads the NGL train-split mean/std (needed to turn the standardised
    observation window back into mm) and the per-station de-bias map when
    ``obs_debias`` is on (item 2).
    """
    lam = float(getattr(cfg, 'lambda_obs', 0.0) or 0.0)
    if lam <= 0:
        return None
    # Restricting the label loss to a sub-domain makes the label term count more
    # per surviving cell, which dilutes lambda_obs by the domain's share of the
    # loss weight (HANDOFF 13.4 item 4: halo3 -> 0.587, i.e. lambda 0.2 behaved
    # like 0.117).  Compensate here so lambda_obs keeps its full-grid-equivalent
    # meaning and stays comparable across loss domains.
    share = 1.0
    if bool(getattr(cfg, 'lambda_obs_domain_compensation', True)):
        share = loss_domain_weight_share(cfg)
        if share > 0:
            lam = lam / share
    cfg.lambda_obs_effective = lam
    cfg.lambda_obs_domain_share = share
    import zarr

    gn = zarr.open(str(cfg.ngl_zarr), 'r')
    cfg.ztd_train_mean = float(np.asarray(gn['ztd_train_mean'][:]).reshape(-1)[0])
    cfg.ztd_train_std = float(np.asarray(gn['ztd_train_std'][:]).reshape(-1)[0])

    operator = StationZTD(cfg)
    bias = None
    if bool(getattr(cfg, 'obs_debias', False)):
        bias, _ = load_obs_debias(cfg)
    loss = ObsConsistencyLoss(
        operator, weight=lam,
        sigma_o_mm=float(getattr(cfg, 'obs_sigma_o_mm', 11.0)), bias=bias)
    return loss.to(rank)


def build_increment_penalty(cfg, rank):
    """逐通道增量惩罚（软约束）；``increment_penalty_mu <= 0`` 时返回 None。

    分母 sigma_b,c 来自 ``dataset/bg_err_std.npz``（见 preprocessing/build_bg_err_std.py）。
    作用格点默认是站点格：ZTD 观测约束只在那里起作用，而 msl/z 这两个"廉价出口"
    的破坏也集中在那里。
    """
    mu = float(getattr(cfg, 'increment_penalty_mu', 0.0) or 0.0)
    if mu <= 0:
        return None
    from main.utils import station_cell_mask
    path = getattr(cfg, 'increment_penalty_file', None) or os.path.join(
        os.path.dirname(str(cfg.ngl_zarr)), 'bg_err_std.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'increment_penalty_mu={mu} 需要 {path}；'
            f'先用 preprocessing/build_bg_err_std.py 生成')
    d = np.load(path, allow_pickle=False)
    sigma = np.asarray(d['sigma_station'], dtype=np.float32)
    n_ch = int(getattr(cfg, 'model_out_chans', 70))
    if sigma.size < n_ch:                      # 兜底：通道数不够时用中位数补齐
        sigma = np.pad(sigma, (0, n_ch - sigma.size),
                       constant_values=float(np.nanmedian(sigma)))
    cfg.increment_penalty_sigma_file = path
    mode = str(getattr(cfg, 'increment_penalty_mode', 'station')).lower()
    mask = station_cell_mask(cfg) if mode == 'station' else None
    pen = IncrementPenalty(sigma[:n_ch].copy(), weight=mu,
                           channel_slice=slice(0, int(getattr(cfg, 'label_n_chans', 70)) - 1),
                           mask=mask)
    return pen.to(rank)


def train_one_epoch(cfg, model, rank, dataloader, optimizer, grad_scaler,
                    warmup_scheduler=None, scheduler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    ddp_oc = torch.zeros(3).to(rank)          # sum(term), sum(mae mm), count
    ddp_pen = torch.zeros(2).to(rank)         # sum(term), count
    obs_loss = getattr(cfg, 'obs_loss', None)
    inc_pen = getattr(cfg, 'inc_penalty', None)
    hw = getattr(cfg, 'grid_hw', None)

    time_start = time.perf_counter()
    log_interval = int(getattr(cfg, 'log_interval', 100) or 0)
    log_batch = bool(getattr(cfg, 'log_batch_loss', False))
    run_sum, run_n = 0.0, 0
    t_log = time_start
    if rank == 0:
        cfg.logger.info(f'[Epoch {getattr(cfg, "epoch", 0) + 1}] {len(dataloader)} iterations')
    for batch_fcst, batch_obs, batch_era5 in dataloader:
        cfg.iteration += 1

        optimizer.zero_grad()
        time_data = time.perf_counter()

        obs_raw = batch_obs                      # standardised window, pre-masking
        batch_fcst = process_bg(batch_fcst, rank, hw)
        batch_era5 = process_bg(batch_era5.unsqueeze(1), rank, hw)
        batch_obs = process_obs(batch_obs, cfg, rank)
        obs_valid = (obs_at_valid_time(obs_raw, cfg, rank)
                     if obs_loss is not None else None)

        if cfg.amp:
            batch_fcst = batch_fcst.half()
            batch_era5 = batch_era5.half()
            batch_obs = batch_obs.half()

        time_data_process = time.perf_counter()
        batch_out = model(batch_fcst, batch_obs)

        loss = cfg.loss_fn(batch_out, batch_era5)
        oc = None
        if obs_loss is not None:
            oc = obs_loss(batch_out, obs_valid, batch_fcst)
            loss = loss + oc
        if inc_pen is not None:
            pen = inc_pen(batch_out, batch_fcst)
            loss = loss + pen
            ddp_pen[0] += float(pen.detach())
            ddp_pen[1] += 1
        if cfg.amp:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=32)
        if cfg.amp:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        if oc is not None:
            ddp_oc[0] += float(oc.detach())
            ddp_oc[1] += float(obs_loss.last_mae_mm)
            ddp_oc[2] += 1
        cfg.lr.append(optimizer.param_groups[0]['lr'])

        batch_loss = float(loss.item())
        if rank == 0 and log_batch:
            cfg.iter_loss.append(batch_loss)
            run_sum += batch_loss
            run_n += 1
        if rank == 0 and (log_batch or (log_interval and cfg.iteration % log_interval == 0)):
            now = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            if log_batch:
                # every batch: its own loss, the running mean of the epoch so far,
                # the learning rate and the wall time of that single step
                cfg.logger.info(f'[iter {cfg.iteration}/{cfg.num_iteration}] '
                                f'epoch {getattr(cfg, "epoch", 0) + 1} '
                                f'loss={batch_loss:.5f} '
                                f'mean={run_sum / max(run_n, 1):.5f} '
                                f'lr={lr:.3e} {now - t_log:.3f}s')
            else:
                cfg.logger.info(f'[iter {cfg.iteration}/{cfg.num_iteration}] '
                                f'loss={batch_loss:.4f} lr={lr:.3e} '
                                f'{(now - t_log) / log_interval:.2f}s/it')
            t_log = now

        if cfg.warmup:
            cfg.warmup = warmup_scheduler()
        elif scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        time_start = time.perf_counter()
        if cfg.iteration >= cfg.num_iteration:
            break

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    epoch_loss = (ddp_loss[0] / ddp_loss[1].clamp_min(1)).item()
    oc_note = ''
    if obs_loss is not None:
        dist.all_reduce(ddp_oc, op=dist.ReduceOp.SUM)
        oc_note = (f'  [obs-consistency: term={ddp_oc[0] / ddp_oc[2].clamp_min(1):.4f}'
                   f' |H(xa)-obs|={ddp_oc[1] / ddp_oc[2].clamp_min(1):.2f} mm]')
    if inc_pen is not None:
        dist.all_reduce(ddp_pen, op=dist.ReduceOp.SUM)
        oc_note += (f'  [increment penalty: term='
                    f'{ddp_pen[0] / ddp_pen[1].clamp_min(1):.5f}]')
    if rank == 0:
        cfg.logger.info(f'[Epoch {getattr(cfg, "epoch", 0) + 1}] '
                        f'train loss={epoch_loss:.4f} (iteration {cfg.iteration}){oc_note}')
    return model, epoch_loss


def evaluate(cfg, model, rank, dataloader, get_fcst_loss=False):
    """Label-weighted ERA5 MAE of the model (or of the background).

    The observation-consistency term is reported on the side but is deliberately
    *not* part of the returned number, so the val-best checkpoint keeps being
    selected on the ERA5 fit and ``best_val`` stays comparable across runs.
    """
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)
    ddp_oc = torch.zeros(3).to(rank)
    obs_loss = getattr(cfg, 'obs_loss', None)
    hw = getattr(cfg, 'grid_hw', None)

    for batch_fcst, batch_obs, batch_era5 in dataloader:
        obs_raw = batch_obs                      # standardised window, pre-masking
        batch_fcst = process_bg(batch_fcst, rank, hw)
        batch_era5 = process_bg(batch_era5.unsqueeze(1), rank, hw)
        batch_obs = process_obs(batch_obs, cfg, rank)
        obs_valid = (obs_at_valid_time(obs_raw, cfg, rank)
                     if obs_loss is not None else None)

        if cfg.amp:
            batch_fcst = batch_fcst.half()
            batch_era5 = batch_era5.half()
            batch_obs = batch_obs.half()

        if get_fcst_loss:
            # the background only carries the bg_chans state channels
            batch_out = batch_fcst[..., :batch_era5.shape[2], :, :]
        else:
            with torch.no_grad():
                batch_out = model(batch_fcst, batch_obs)

        target = batch_era5
        if get_fcst_loss:
            target = batch_era5[..., :batch_out.shape[2], :, :]
        loss = cfg.loss_fn(batch_out, target).item()
        ddp_loss[0] += loss
        ddp_loss[1] += 1
        if obs_loss is not None:
            with torch.no_grad():
                oc = obs_loss(batch_out, obs_valid, batch_fcst)
            ddp_oc[0] += float(oc)
            ddp_oc[1] += float(obs_loss.last_mae_mm)
            ddp_oc[2] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    epoch_loss = (ddp_loss[0] / ddp_loss[1].clamp_min(1)).item()
    oc_note = ''
    if obs_loss is not None:
        dist.all_reduce(ddp_oc, op=dist.ReduceOp.SUM)
        oc_note = (f'  [obs-consistency: term={ddp_oc[0] / ddp_oc[2].clamp_min(1):.4f}'
                   f' |H(x)-obs|={ddp_oc[1] / ddp_oc[2].clamp_min(1):.2f} mm]')
    if rank == 0:
        kind = 'background-only' if get_fcst_loss else 'model'
        cfg.logger.info(f'[Eval] {kind} loss={epoch_loss:.4f} '
                        f'(iteration {cfg.iteration}){oc_note}')
    return epoch_loss


def save_checkpoint_fsdp(cfg, model, rank, optimizer=None, scheduler=None, val_loss=None):
    """Save a FULL (unsharded) checkpoint; only rank 0 writes the file.

    ``FSDP.summon_full_params`` + ``model.module.state_dict()`` returns sharded
    tensors with SHARD_GRAD_OP, so the FULL_STATE_DICT API is used instead.
    All ranks must enter the context (it is collective); only rank 0 writes.
    """
    save_file = checkpoint_file(cfg)
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer) if optimizer is not None else None
        sched_state = scheduler.state_dict() if scheduler is not None else None
    if rank == 0:
        payload = {'model': model_state, 'iteration': {'iteration': cfg.iteration},
                   'val_loss': val_loss,
                   'model_id': model_id(cfg), 'exp_tag': exp_tag(cfg),
                   'arch_tag': arch_tag(cfg),
                   'zero_obs': bool(getattr(cfg, 'zero_obs', False)),
                   'include_fuxi_tp': bool(getattr(cfg, 'include_fuxi_tp', False)),
                   'obs_mode': str(getattr(cfg, 'obs_mode', 'absolute')),
                   'loss_station_weight': float(getattr(cfg, 'loss_station_weight', 1.0)),
                   'loss_nostation_weight': float(getattr(cfg, 'loss_nostation_weight', 1.0)),
                   'lambda_obs': float(getattr(cfg, 'lambda_obs', 0.0) or 0.0),
                   'lambda_obs_effective': float(getattr(cfg, 'lambda_obs_effective', 0.0) or 0.0),
                   'lambda_obs_domain_share': float(getattr(cfg, 'lambda_obs_domain_share', 1.0)),
                   'obs_sigma_o_mm': float(getattr(cfg, 'obs_sigma_o_mm', 11.0)),
                   'obs_debias': bool(getattr(cfg, 'obs_debias', False)),
                   'obs_res_scale_mm': float(getattr(cfg, 'obs_res_scale_mm', 0.0) or 0.0),
                   'freeze_msl': bool(getattr(cfg, 'freeze_msl', False)),
                   'increment_penalty_mu': float(getattr(cfg, 'increment_penalty_mu', 0.0) or 0.0)}
        if optim_state is not None:
            payload['optimizer'] = optim_state
        if sched_state is not None:
            payload['scheduler'] = sched_state
        torch.save(payload, save_file)
        cfg.logger.info(f'[Save Model]: {save_file}')
    return save_file


def make_loaders(cfg, world_size, rank):
    common = dict(num_workers=cfg.num_workers, world_size=world_size, rank=rank,
                  persistent_workers=cfg.persistent_workers,
                  prefetch_factor=cfg.prefetch_factor,
                  multiprocessing_context=cfg.multiprocessing_context,
                  pin_memory=cfg.pin_memory)
    train_loader = build_dataloader(cfg, cfg.dates_train_range, cfg.batch_size,
                                    shuffle=True, **common)
    val_loader = build_dataloader(cfg, cfg.dates_val_range, cfg.batch_size,
                                  shuffle=False, **common)
    test_loader = None
    if getattr(cfg, 'dates_test_range', None):
        test_loader = build_dataloader(cfg, cfg.dates_test_range, cfg.batch_size,
                                       shuffle=False, **common)
    return train_loader, val_loader, test_loader


def main(rank, configs, master_port, world_size, overrides=None):
    cfg = init_dist(rank, configs, master_port, world_size, overrides)
    cfg.amp = bool(getattr(cfg, 'amp', True))

    cfg.obs_loss = build_obs_loss(cfg, rank)          # None unless lambda_obs > 0
    cfg.inc_penalty = build_increment_penalty(cfg, rank)   # None unless mu > 0
    if rank == 0 and cfg.obs_loss is not None:
        cfg.logger.info(
            f'[ObsLoss] lambda={getattr(cfg, "lambda_obs", 0.0):g} '
            f'-> effective {cfg.obs_loss.weight:g} '
            f'(domain share {getattr(cfg, "lambda_obs_domain_share", 1.0):.3f}); '
            f'sigma_o={cfg.obs_loss.sigma_o_mm:g} mm '
            f'-> {cfg.obs_loss.weight / cfg.obs_loss.sigma_o_mm:.4f} per mm; '
            f'debias={cfg.obs_loss.bias is not None} '
            f'stations={cfg.obs_loss.operator.n_cell}')
    if rank == 0 and cfg.inc_penalty is not None:
        _s = cfg.inc_penalty.sigma
        cfg.logger.info(
            f'[IncPenalty] mu={cfg.inc_penalty.weight:g} '
            f'channels=[0,{cfg.inc_penalty.ch1}) mask='
            f'{"station" if cfg.inc_penalty.mask is not None else "all cells"} '
            f'sigma median={float(_s.median()):.3f} '
            f'(msl={float(_s[68]):.3f}, r700={float(_s[61]):.3f}, z500={float(_s[7]):.3f})')

    train_loader, val_loader, test_loader = make_loaders(cfg, world_size, rank)
    if rank == 0:
        sizes = f'train={len(train_loader.dataset)} val={len(val_loader.dataset)}'
        if test_loader is not None:
            sizes += f' test={len(test_loader.dataset)}'
        cfg.logger.info(f'[Data] {sizes} samples/rank, batch_size={cfg.batch_size}, '
                        f'world_size={world_size}')

    model = AssimilationNetv6(bg_chans=cfg.model_bg_chans,
                              obs_chans=cfg.model_obs_chans,
                              obs_frames=cfg.model_obs_frames,
                              out_chans=cfg.model_out_chans,
                              embed_dim=cfg.model_embed_dim,
                              depth=cfg.model_depth,
                              freeze_msl=bool(getattr(cfg, 'freeze_msl', False))).cuda(rank)

    if rank == 0:
        if getattr(cfg, 'freeze_msl', False):
            cfg.logger.info('[Model] freeze_msl: analysis msl (channel 68) = '
                            'background msl (no residual)')
        cfg.logger.info(f'[Model Parameters]: {get_parameter_number(model)}')

    if cfg.amp:
        mp_precision = MixedPrecision(param_dtype=torch.float16,
                                      reduce_dtype=torch.float32,
                                      buffer_dtype=torch.float32)
        model = FSDP(model, mixed_precision=mp_precision,
                     device_id=torch.cuda.current_device(),
                     sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
        grad_scaler = ShardedGradScaler()
    else:
        model = FSDP(model, device_id=torch.cuda.current_device(),
                     sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
        grad_scaler = None

    if cfg.warmup:
        warmup_steps = int(cfg.num_iteration * cfg.warmup_rate)
        t_max = int(cfg.num_iteration - warmup_steps)
        optimizer, scheduler = build_optimizer(
            opt_type=cfg.opt_type, learning_rate=cfg.start_lr,
            weight_decay=cfg.weight_decay, model=model, scheduler=cfg.scheduler,
            step_size=getattr(cfg, 'step_size', None),
            T_max=t_max if cfg.scheduler == 'CosineAnnealingLR' else None)
        warmup_scheduler = WarmupScheduler(optimizer=optimizer, start_lr=cfg.start_lr,
                                           stop_lr=cfg.stop_lr, warmup_steps=warmup_steps)
        if rank == 0:
            cfg.logger.info(f'[Schedule] warmup={warmup_steps} steps, cosine T_max={t_max}')
    else:
        optimizer, scheduler = build_optimizer(
            opt_type=cfg.opt_type, learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay, model=model, scheduler=cfg.scheduler,
            step_size=getattr(cfg, 'step_size', None),
            T_max=cfg.num_iteration if cfg.scheduler == 'CosineAnnealingLR' else None)
        warmup_scheduler = None

    start_iteration = 0
    if getattr(cfg, 'resume_model', None) is not None:
        model, optimizer, scheduler, iteration = load_checkpoint(
            cfg.resume_model, model, optimizer, scheduler)
        start_iteration = iteration or 0
        if rank == 0:
            cfg.logger.info(f'[Resume]: {cfg.resume_model}')
    elif getattr(cfg, 'pre_model', None) is not None:
        model, _, _, _ = load_checkpoint(cfg.pre_model, model)
        start_iteration = getattr(cfg, 'start_iteration', 0)
        if rank == 0:
            cfg.logger.info(f'[Pre Model]: {cfg.pre_model}')

    cfg.lr, cfg.loss_train, cfg.loss_val = [], [], []
    cfg.iter_loss = []                              # per-batch losses (rank 0)
    cfg.iteration = start_iteration
    best_val, best_iter, best_epoch = float('inf'), None, None

    for epoch in range(cfg.num_epochs):
        cfg.epoch = epoch
        model, loss_train = train_one_epoch(cfg, model, rank, train_loader,
                                            optimizer, grad_scaler,
                                            warmup_scheduler, scheduler)
        loss_val = evaluate(cfg, model, rank, val_loader)
        cfg.loss_train.append(loss_train)
        cfg.loss_val.append(loss_val)
        run.log({'train_loss':loss_train,'valloss':loss_val})

        # Only the best-val model is kept on disk.  Every rank evaluates the
        # same (all-reduced) val loss, and ``state_dict_type`` is collective,
        # so all ranks enter the saving context together.
        if loss_val < best_val - float(getattr(cfg, 'min_delta', 0.0)):
            best_val, best_iter, best_epoch = loss_val, cfg.iteration, epoch + 1
            save_checkpoint_fsdp(cfg, model, rank, optimizer, scheduler, loss_val)
            if rank == 0:
                cfg.logger.info(f'[Best] val={loss_val:.4f} @ epoch {epoch + 1} '
                                f'(iter {cfg.iteration}) -> saved')
        elif rank == 0:
            cfg.logger.info(f'[Best] val={loss_val:.4f} @ epoch {epoch + 1} '
                            f'-- keeping the earlier best {best_val:.4f}')

        if getattr(cfg, 'early_stop', None) is not None:
            if not hasattr(cfg, 'early_stopper'):
                cfg.early_stopper = EarlyStopping(**cfg.early_stop)
            cfg.early_stopper(loss_val)
            if cfg.early_stopper.early_stop:
                if rank == 0:
                    cfg.logger.info(f'[Early Stop] at epoch {epoch + 1}, '
                                    f'best val {cfg.early_stopper.best_loss:.4f}')
                break

        if cfg.iteration >= cfg.num_iteration:
            break

    run.finish()
    # 训练循环一结束就先把曲线与最小摘要落盘：收尾评估（载回 best + background-only
    # + test）偶尔会失败/被杀，之前会导致 30+ 分钟的训练连 loss 曲线都没有。
    # 这里先写一份 summary_train.json 与三个 npy，收尾评估若成功，末尾再覆盖成完整版。
    if rank == 0:
        np.save(os.path.join(experiment_dir(cfg), 'train_loss.npy'), np.array(cfg.loss_train))
        np.save(os.path.join(experiment_dir(cfg), 'val_loss.npy'), np.array(cfg.loss_val))
        np.save(os.path.join(experiment_dir(cfg), 'lr.npy'), np.array(cfg.lr))
        if cfg.iter_loss:
            np.save(os.path.join(experiment_dir(cfg), 'iter_loss.npy'),
                    np.array(cfg.iter_loss, dtype=np.float32))
        with open(os.path.join(experiment_dir(cfg), 'summary_train.json'), 'w') as fh:
            json.dump({'model_id': model_id(cfg), 'exp_tag': exp_tag(cfg),
                       'arch_tag': arch_tag(cfg),
                       'best_val_loss': best_val, 'best_iteration': best_iter,
                       'best_epoch': best_epoch,
                       'train_loss': cfg.loss_train, 'val_loss': cfg.loss_val,
                       'checkpoint': checkpoint_file(cfg), 'log': log_file(cfg),
                       'note': '训练结束时写入；收尾评估失败时这是唯一的记录'},
                      fh, indent=2, default=float)

    # Evaluate the *best* model, not whatever the last epoch left behind.
    ckpt = checkpoint_file(cfg)
    if best_iter is not None and os.path.exists(ckpt):
        if rank == 0:
            cfg.logger.info(f'[Restore] val-best checkpoint '
                            f'(val={best_val:.4f} @ iter {best_iter})')
        model, _, _, _ = load_checkpoint(ckpt, model)

    # NOTE: evaluate() calls dist.all_reduce internally, so *every* rank must
    # run it -- only logging/saving is gated on rank 0.
    bg_train = evaluate(cfg, model, rank, train_loader, get_fcst_loss=True)
    bg_val = evaluate(cfg, model, rank, val_loader, get_fcst_loss=True)
    result = {
        'model_id': model_id(cfg), 'exp_tag': exp_tag(cfg), 'arch_tag': arch_tag(cfg),
        'configs': configs, 'world_size': world_size,
        'train_samples': len(train_loader.dataset), 'val_samples': len(val_loader.dataset),
        'num_iteration': cfg.num_iteration, 'batch_size': cfg.batch_size,
        'obs_frames': cfg.obs_frames, 'training_tp': getattr(cfg, 'tp_label_source', 'imerg'),
        'zero_obs': bool(getattr(cfg, 'zero_obs', False)),
        'include_fuxi_tp': bool(getattr(cfg, 'include_fuxi_tp', False)),
        'obs_mode': str(getattr(cfg, 'obs_mode', 'absolute')),
        'loss_station_weight': float(getattr(cfg, 'loss_station_weight', 1.0)),
        'loss_nostation_weight': float(getattr(cfg, 'loss_nostation_weight', 1.0)),
        'lambda_obs': float(getattr(cfg, 'lambda_obs', 0.0) or 0.0),
        'lambda_obs_effective': float(getattr(cfg, 'lambda_obs_effective', 0.0) or 0.0),
        'lambda_obs_domain_share': float(getattr(cfg, 'lambda_obs_domain_share', 1.0)),
        'obs_sigma_o_mm': float(getattr(cfg, 'obs_sigma_o_mm', 11.0)),
        'obs_debias': bool(getattr(cfg, 'obs_debias', False)),
        'obs_res_scale_mm': float(getattr(cfg, 'obs_res_scale_mm', 0.0) or 0.0),
        'freeze_msl': bool(getattr(cfg, 'freeze_msl', False)),
        'increment_penalty_mu': float(getattr(cfg, 'increment_penalty_mu', 0.0) or 0.0),
        'obs_freeze_geometry': bool(getattr(cfg, 'obs_freeze_geometry', False)),
        'best_val_loss': best_val, 'best_iteration': best_iter, 'best_epoch': best_epoch,
        'train_loss': cfg.loss_train, 'val_loss': cfg.loss_val,
        'background_only_train': bg_train, 'background_only_val': bg_val,
        'checkpoint': ckpt, 'log': log_file(cfg),
    }
    if rank == 0:
        cfg.logger.info(f'[Background-only loss] train={bg_train:.4f} val={bg_val:.4f}')
        np.save(os.path.join(experiment_dir(cfg), 'train_loss.npy'), np.array(cfg.loss_train))
        np.save(os.path.join(experiment_dir(cfg), 'val_loss.npy'), np.array(cfg.loss_val))
        np.save(os.path.join(experiment_dir(cfg), 'lr.npy'), np.array(cfg.lr))
        if cfg.iter_loss:
            np.save(os.path.join(experiment_dir(cfg), 'iter_loss.npy'),
                    np.array(cfg.iter_loss, dtype=np.float32))

    if test_loader is not None:
        loss_test = evaluate(cfg, model, rank, test_loader, get_fcst_loss=True)
        loss_test_model = evaluate(cfg, model, rank, test_loader)
        result['test_samples'] = len(test_loader.dataset)
        result['test_model_loss'] = loss_test_model
        result['test_background_loss'] = loss_test
        if rank == 0:
            note = ('(NOTE: model includes tp, background does not -- '
                    'use plot_results.py for the 69-channel comparison)'
                    if int(cfg.model_bg_chans) < 70 else
                    '(70 channels on both sides: FuXi tp is the tp background)')
            cfg.logger.info(f'[Test] model={loss_test_model:.4f} '
                            f'background={loss_test:.4f} {note}')

    if rank == 0:
        with open(os.path.join(experiment_dir(cfg), 'summary.json'), 'w') as fh:
            json.dump(result, fh, indent=2, default=float)
        cfg.logger.info(f'[Done] results -> {experiment_dir(cfg)}')

    dist.barrier()
    dist.destroy_process_group()


def _coerce(value):
    """CLI string -> bool / int / float / plain string."""
    low = value.lower()
    if low in ('true', 'yes', 'on'):
        return True
    if low in ('false', 'no', 'off'):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs')
    parser.add_argument('--master_port', type=str, default='22346')
    parser.add_argument('--model_id', type=str, default=None,
                        help='run prefix; configs.model_id by default')
    parser.add_argument('--exp_tag', type=str, default=None,
                        help='实验配置 tag; auto-derived from the setup by default')
    parser.add_argument('--arch_tag', type=str, default=None,
                        help='模型配置 tag; auto-derived from the network by default')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override any configs.py knob, e.g. --set zero_obs=true')
    args = parser.parse_args()
    world_size = max(torch.cuda.device_count(), 1)
    overrides = {'model_id': args.model_id, 'exp_tag': args.exp_tag,
                 'arch_tag': args.arch_tag}
    for item in args.set:
        if '=' not in item:
            parser.error(f'--set expects KEY=VALUE, got {item!r}')
        key, value = item.split('=', 1)
        overrides[key.strip()] = _coerce(value.strip())
    mp.spawn(main, args=(args.configs, args.master_port, world_size, overrides),
             nprocs=world_size, join=True)
