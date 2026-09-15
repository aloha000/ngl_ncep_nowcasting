"""Checkpoint / logging helpers (same as the train_packet, lightly cleaned)."""

from __future__ import annotations

import datetime
import logging
import os
from collections import OrderedDict

import torch
import torch.distributed as dist


def create_logger(save_path, type='Train', log_dir=None, file_name=None):
    """File logger at INFO; the console handler only shows WARNING and above.

    The console threshold matters: training is usually launched as
    ``nohup bash train.sh > somewhere.log &``, and a full INFO console stream
    would duplicate the (large) file log into that redirect.  Only the file
    gets the progress lines.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers when the module is imported twice
    if not logger.handlers:
        now = datetime.datetime.now().strftime("%Y%m%d%H")
        log_path = log_dir or f'{save_path}/logs'
        os.makedirs(log_path, exist_ok=True)
        log_file = f'{log_path}/{file_name or f"{type}_log_{now}"}.log'
        fileinfo = logging.FileHandler(log_file)
        controshow = logging.StreamHandler()
        controshow.setLevel(logging.WARNING)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fileinfo.setFormatter(formatter)
        controshow.setFormatter(formatter)
        logger.addHandler(fileinfo)
        logger.addHandler(controshow)
    return logger


# ---------------------------------------------------------------------------
# run identity: {model_id}_{实验配置} folders, {model_id}_{模型配置} files
# ---------------------------------------------------------------------------
def exp_tag(cfg):
    """实验配置 tag -- explicit ``cfg.exp_tag`` or one derived from the setup."""
    tag = getattr(cfg, 'exp_tag', None)
    if tag:
        return str(tag)
    minutes = int(getattr(cfg, 'obs_frames', 0)) * int(getattr(cfg, 'obs_frame_minutes', 5))
    lead = int(getattr(cfg, 'fcst_step', 1)) * 6
    tp = str(getattr(cfg, 'tp_label_source', 'imerg'))
    tag = f'lead{lead}h_obs{minutes // 60}h_{tp}tp'
    if getattr(cfg, 'zero_obs', False):
        tag += '_zeroobs'
    mode = str(getattr(cfg, 'obs_mode', 'absolute')).lower()
    if mode != 'absolute':
        tag += f'_{mode}'
    return tag


def obs_chans(cfg):
    """Observation input channels implied by ``cfg.obs_mode`` (+ mask/lat/lon).

    ``model_obs_chans`` in configs.py is evaluated at import time, which is
    *before* ``--set obs_mode=...`` is applied, so anything that builds the model
    must come back through here (train_FSDP.init_dist, plot_results).
    """
    mode = str(getattr(cfg, 'obs_mode', 'absolute')).lower()
    head = {'absolute': 1, 'residual': 1, 'both': 2}.get(mode)
    if head is None:
        raise ValueError(f"obs_mode must be absolute|residual|both, got {mode!r}")
    return (head + int(getattr(cfg, 'obs_add_mask', True))
            + 2 * int(getattr(cfg, 'obs_add_latlon', True)))


def arch_tag(cfg):
    """模型配置 tag -- explicit ``cfg.arch_tag`` or one derived from the net shape."""
    tag = getattr(cfg, 'arch_tag', None)
    if tag:
        return str(tag)
    depth = ''.join(str(d) for d in getattr(cfg, 'model_depth', (2, 2, 2)))
    return f'ed{getattr(cfg, "model_embed_dim", 0)}_d{depth}'


def model_id(cfg):
    return str(getattr(cfg, 'model_id', 'model') or 'model')


def experiment_name(cfg):
    """``{model_id}_{实验配置}`` -- the per-run folder name."""
    return f'{model_id(cfg)}_{exp_tag(cfg)}'


def results_root(cfg):
    return getattr(cfg, 'results_dir', None) or os.path.join(cfg.work_dir, 'results')


def experiment_dir(cfg):
    """Where one run keeps its checkpoint, plots and summary."""
    return os.path.join(results_root(cfg), experiment_name(cfg))


def checkpoint_file(cfg):
    """``{model_id}_{模型配置}.pth`` -- the single val-best checkpoint of a run."""
    return os.path.join(experiment_dir(cfg), f'{model_id(cfg)}_{arch_tag(cfg)}.pth')


def log_file(cfg):
    log_dir = getattr(cfg, 'log_dir', None) or os.path.join(cfg.work_dir, 'logs')
    return os.path.join(log_dir, f'{model_id(cfg)}_{arch_tag(cfg)}.log')


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _clean_state_dict(state):
    """Strip FSDP/DDP wrappers added to the checkpoint keys."""
    cleaned = OrderedDict()
    for key, value in state.items():
        for prefix in ('_fsdp_wrapped_module.', 'module.'):
            while key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def fsdp_state_dict_type(model, rank0_only):
    """Return the (context, FSDP) pair for a FULL_STATE_DICT round trip."""
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    return FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only)), FSDP


def is_fsdp(model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except Exception:
        return False


def _load_state(model, state):
    """Load a full (unsharded) state dict into a plain or FSDP-wrapped model."""
    state = _clean_state_dict(state)
    ref = model.state_dict()
    if set(state) != set(ref) and len(state) == len(ref):
        # old checkpoints had reordered keys
        state = OrderedDict(zip(ref.keys(), state.values()))

    if is_fsdp(model):
        ctx, _ = fsdp_state_dict_type(model, rank0_only=False)
        with ctx:
            missing, unexpected = model.load_state_dict(state, strict=False)
    else:
        target = model.module if hasattr(model, 'module') else model
        missing, unexpected = target.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[load_checkpoint] missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    _load_state(model, checkpoint['model'])
    if optimizer is not None and checkpoint.get('optimizer') is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as exc:  # sharded vs full optim state
            print(f'[load_checkpoint] optimizer state not restored: {exc}')
    if scheduler is not None and checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint.get('iteration')


def save_checkpoint(file_name, model, iteration, optimizer=None, scheduler=None):
    if get_rank() != 0:
        return None
    save_dict = OrderedDict()
    if hasattr(model, 'module'):
        model = model.module
    save_dict['model'] = model.state_dict()
    save_dict['iteration'] = dict(iteration=iteration)
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    torch.save(save_dict, file_name)
    return None
