"""Checkpoint / logging helpers (same as the train_packet, lightly cleaned)."""

from __future__ import annotations

import datetime
import logging
import os
from collections import OrderedDict

import torch
import torch.distributed as dist


def create_logger(save_path, type='Train'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers when the module is imported twice
    if not logger.handlers:
        now = datetime.datetime.now().strftime("%Y%m%d%H")
        log_path = f'{save_path}/logs'
        os.makedirs(log_path, exist_ok=True)
        log_file = f'{log_path}/{type}_log_{now}.log'
        fileinfo = logging.FileHandler(log_file)
        controshow = logging.StreamHandler()
        controshow.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fileinfo.setFormatter(formatter)
        controshow.setFormatter(formatter)
        logger.addHandler(fileinfo)
        logger.addHandler(controshow)
    return logger


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
