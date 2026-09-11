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

from main.model import (AssimilationNetv6, EarlyStopping, WarmupScheduler,
                        build_optimizer, get_parameter_number)
from main.utils import build_dataloader, create_logger, load_checkpoint


def init_dist(rank, configs, master_port, world_size):
    xconfig = importlib.import_module(configs)

    if rank == 0:
        os.makedirs(os.path.join(xconfig.work_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(xconfig.work_dir, 'logs'), exist_ok=True)

    xconfig.logger = create_logger(xconfig.work_dir, 'Train')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    seed = xconfig.rand_seed + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    xconfig.logger.info(f'[Work Dir]: {xconfig.work_dir}\n'
                        f'[Configs]: {configs}\n'
                        f'[World Size]: {world_size}\n'
                        f'[Rank]: {rank}\n'
                        f'[Device]: {torch.cuda.current_device()}\n'
                        f'[Seed]: {seed}\n')
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
    """
    obs = obs.to(rank).float()
    finite = torch.isfinite(obs)
    data = torch.nan_to_num(obs)
    mask = finite.any(dim=2, keepdim=True).float()      # (B,T,1,H,W)

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


def train_one_epoch(cfg, model, rank, dataloader, optimizer, grad_scaler,
                    warmup_scheduler=None, scheduler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    hw = getattr(cfg, 'grid_hw', None)

    time_start = time.perf_counter()
    cfg.logger.info(f'[Epoch Iteration Number] ||| {len(dataloader)}')
    for batch_fcst, batch_obs, batch_era5 in dataloader:
        cfg.iteration += 1

        optimizer.zero_grad()
        time_data = time.perf_counter()

        batch_fcst = process_bg(batch_fcst, rank, hw)
        batch_era5 = process_bg(batch_era5.unsqueeze(1), rank, hw)
        batch_obs = process_obs(batch_obs, cfg, rank)

        if cfg.amp:
            batch_fcst = batch_fcst.half()
            batch_era5 = batch_era5.half()
            batch_obs = batch_obs.half()

        time_data_process = time.perf_counter()
        batch_out = model(batch_fcst, batch_obs)

        loss = cfg.loss_fn(batch_out, batch_era5)
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
        cfg.lr.append(optimizer.param_groups[0]['lr'])

        time_model = time.perf_counter()
        cfg.logger.info(f"[Rank]: {rank} ||| [Iteration]: {cfg.iteration} ||| "
                        f"[LR]: {optimizer.param_groups[0]['lr']:.3e} ||| "
                        f"[Loss]: {loss.item():.4f} ||| "
                        f"[Data]: {time_data - time_start:.2f}s ||| "
                        f"[Process]: {time_data_process - time_data:.2f}s ||| "
                        f"[Model]: {time_model - time_data_process:.2f}s ||| "
                        f"[MaxMem]: {torch.cuda.max_memory_allocated(rank) / 1024 ** 2:.0f}MB")

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
    cfg.logger.info(f"[Rank]: {rank} ||| [Iteration]: {cfg.iteration} ||| "
                    f"[Epoch loss]: {epoch_loss:.4f}")
    return model, epoch_loss


def evaluate(cfg, model, rank, dataloader, get_fcst_loss=False):
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)
    hw = getattr(cfg, 'grid_hw', None)

    for batch_fcst, batch_obs, batch_era5 in dataloader:
        batch_fcst = process_bg(batch_fcst, rank, hw)
        batch_era5 = process_bg(batch_era5.unsqueeze(1), rank, hw)
        batch_obs = process_obs(batch_obs, cfg, rank)

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

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    epoch_loss = (ddp_loss[0] / ddp_loss[1].clamp_min(1)).item()
    cfg.logger.info('{:#^75}'.format('Evaluation'))
    cfg.logger.info(f"[Rank]: {rank} ||| [Iteration]: {cfg.iteration} ||| "
                    f"[Loss]: {epoch_loss:.4f} ||| [Background loss]: {get_fcst_loss}")
    cfg.logger.info('{:#^75}'.format('Evaluation'))
    return epoch_loss


def save_checkpoint_fsdp(cfg, model, rank, optimizer=None, scheduler=None):
    """Save a FULL (unsharded) checkpoint; only rank 0 writes the file.

    ``FSDP.summon_full_params`` + ``model.module.state_dict()`` returns sharded
    tensors with SHARD_GRAD_OP, so the FULL_STATE_DICT API is used instead.
    All ranks must enter the context (it is collective); only rank 0 writes.
    """
    save_file = os.path.join(cfg.work_dir, 'model', f'iteration_{cfg.iteration}.pth')
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer) if optimizer is not None else None
        sched_state = scheduler.state_dict() if scheduler is not None else None
    if rank == 0:
        payload = {'model': model_state, 'iteration': {'iteration': cfg.iteration}}
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


def main(rank, configs, master_port, world_size):
    cfg = init_dist(rank, configs, master_port, world_size)
    cfg.amp = bool(getattr(cfg, 'amp', True))

    train_loader, val_loader, test_loader = make_loaders(cfg, world_size, rank)
    cfg.logger.info('{:#^75}'.format('Data Information'))
    cfg.logger.info(f'[Train]: {len(train_loader.dataset)} samples, '
                    f'{len(train_loader)} batches, batch_size={cfg.batch_size}')
    cfg.logger.info(f'[Val]: {len(val_loader.dataset)} samples, '
                    f'{len(val_loader)} batches')
    if test_loader is not None:
        cfg.logger.info(f'[Test]: {len(test_loader.dataset)} samples, '
                        f'{len(test_loader)} batches')
    cfg.logger.info('{:#^75}'.format('Data Information'))

    model = AssimilationNetv6(bg_chans=cfg.model_bg_chans,
                              obs_chans=cfg.model_obs_chans,
                              obs_frames=cfg.model_obs_frames,
                              out_chans=cfg.model_out_chans,
                              embed_dim=cfg.model_embed_dim,
                              depth=cfg.model_depth).cuda(rank)
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
        cfg.logger.info(f'[Warm Up]: {warmup_steps} ||| [T Max]: {t_max}')
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
        cfg.logger.info(f'[Resume]: {cfg.resume_model}')
    elif getattr(cfg, 'pre_model', None) is not None:
        model, _, _, _ = load_checkpoint(cfg.pre_model, model)
        start_iteration = getattr(cfg, 'start_iteration', 0)
        cfg.logger.info(f'[Pre Model]: {cfg.pre_model}')

    cfg.lr, cfg.loss_train, cfg.loss_val = [], [], []
    cfg.iteration = start_iteration

    for epoch in range(cfg.num_epochs):
        model, loss_train = train_one_epoch(cfg, model, rank, train_loader,
                                            optimizer, grad_scaler,
                                            warmup_scheduler, scheduler)
        save_checkpoint_fsdp(cfg, model, rank, optimizer, scheduler)

        loss_val = evaluate(cfg, model, rank, val_loader)
        cfg.loss_train.append(loss_train)
        cfg.loss_val.append(loss_val)

        if getattr(cfg, 'early_stop', None) is not None:
            if not hasattr(cfg, 'early_stopper'):
                cfg.early_stopper = EarlyStopping(**cfg.early_stop)
            cfg.early_stopper(loss_val)
            if cfg.early_stopper.early_stop:
                cfg.logger.info(f'[Early Stop] at epoch {epoch + 1}, '
                                f'best val {cfg.early_stopper.best_loss:.4f}')
                break

        if cfg.iteration >= cfg.num_iteration:
            break

    # NOTE: evaluate() calls dist.all_reduce internally, so *every* rank must
    # run it -- only logging/saving is gated on rank 0.
    bg_train = evaluate(cfg, model, rank, train_loader, get_fcst_loss=True)
    bg_val = evaluate(cfg, model, rank, val_loader, get_fcst_loss=True)
    if rank == 0:
        cfg.logger.info(f'[Background-only loss] train={bg_train:.4f} val={bg_val:.4f}')
        cfg.logger.info(f'[All Train loss]: {cfg.loss_train}')
        cfg.logger.info(f'[All Val loss]: {cfg.loss_val}')
        np.save(f'{cfg.work_dir}/train_loss.npy', np.array(cfg.loss_train))
        np.save(f'{cfg.work_dir}/val_loss.npy', np.array(cfg.loss_val))
        np.save(f'{cfg.work_dir}/lr.npy', np.array(cfg.lr))

    if test_loader is not None:
        loss_test = evaluate(cfg, model, rank, test_loader, get_fcst_loss=True)
        loss_test_model = evaluate(cfg, model, rank, test_loader)
        if rank == 0:
            cfg.logger.info(f'[Test] model={loss_test_model:.4f} background={loss_test:.4f}')

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs')
    parser.add_argument('--master_port', type=str, default='22346')
    args = parser.parse_args()
    world_size = max(torch.cuda.device_count(), 1)
    mp.spawn(main, args=(args.configs, args.master_port, world_size),
             nprocs=world_size, join=True)
