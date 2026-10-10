"""Collective FSDP checkpoints; model weights remain usable by eval_results.py."""
from contextlib import nullcontext
from pathlib import Path
import os
import random
import tempfile
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig)


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _context(model, saving=False):
    if not isinstance(model, FSDP):
        return nullcontext()
    return FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=saving),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=saving))


def _iteration(value):
    if isinstance(value, dict):  # old save_checkpoint stored {'iteration': N}
        value = value.get('iteration')
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'Invalid checkpoint iteration: {value!r}')
    return value


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['cuda'])


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(file_name, model, iteration, optimizer=None, scheduler=None,
                    *, grad_scaler=None, warmup_scheduler=None, early_stop=None,
                    training_state=None, also_save=None):
    """All FSDP ranks MUST call. Rank 0 writes full weights and optimizer state."""
    with _context(model, saving=True):
        model_state = model.state_dict()
        optimizer_state = (FSDP.optim_state_dict(model, optimizer) if isinstance(model, FSDP)
                           else optimizer.state_dict()) if optimizer is not None else None
    local_rng = _rng_state()
    rng_by_rank = [local_rng]
    if dist.is_initialized():
        rng_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(rng_by_rank, local_rng)
    error = [None]
    if _rank() == 0:
        try:
            payload = dict(checkpoint_version=2, model=model_state, iteration=_iteration(iteration),
                           optimizer=optimizer_state,
                           optimizer_class=type(optimizer).__name__ if optimizer is not None else None,
                           optimizer_format='fsdp_full' if isinstance(model, FSDP) else 'torch',
                           scheduler=scheduler.state_dict() if scheduler is not None else None,
                           scheduler_class=type(scheduler).__name__ if scheduler is not None else None,
                           grad_scaler=grad_scaler.state_dict() if grad_scaler is not None else None,
                           warmup_scheduler=warmup_scheduler.state_dict() if warmup_scheduler is not None else None,
                           early_stop=early_stop.state_dict() if early_stop is not None else None,
                           training_state=dict(training_state or {}), rng_by_rank=rng_by_rank)
            _atomic_save(payload, file_name)
            if also_save is not None and Path(also_save) != Path(file_name):
                _atomic_save(payload, also_save)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f'Checkpoint write failed: {error[0]}')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None,
                    *, grad_scaler=None, warmup_scheduler=None, early_stop=None,
                    reset_optimizer=False, return_state=False, restore_rng=True):
    """Strict name-based load; keep the FSDP wrapper. All FSDP ranks must call.

    Old multi-GPU files contain only rank 0's optimizer shard. They can be
    loaded as pre_model, or explicitly resumed with reset_optimizer=True.
    Missing optimizer/scaler/early-stop state is never silently fabricated.
    """
    # Checkpoints are trusted local torch.save files; RNG includes NumPy objects.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    version = checkpoint.get('checkpoint_version', 1)
    iteration = _iteration(checkpoint.get('iteration', 0))
    state = dict(checkpoint.get('training_state', {}))
    resume = optimizer is not None
    if resume and not reset_optimizer:
        if isinstance(model, FSDP) and checkpoint.get('optimizer_format') != 'fsdp_full':
            raise ValueError('Legacy/non-FSDP optimizer state cannot restore FSDP training. '
                             'Use --set resume_reset_optimizer=true to resume weights/iteration '
                             'with fresh optimizer, or --set pre_model=PATH to start from weights.')
        if checkpoint.get('optimizer') is None:
            raise ValueError('Checkpoint has no optimizer; use resume_reset_optimizer=true or pre_model')
        if version >= 2:
            if checkpoint.get('optimizer_class') != type(optimizer).__name__:
                raise ValueError('Optimizer type changed; use resume_reset_optimizer=true or pre_model')
            if checkpoint.get('scheduler_class') != (type(scheduler).__name__ if scheduler is not None else None):
                raise ValueError('Scheduler type changed; use resume_reset_optimizer=true or pre_model')
            if (checkpoint.get('warmup_scheduler') is None) != (warmup_scheduler is None):
                raise ValueError('Warmup configuration changed; use resume_reset_optimizer=true or pre_model')
            for name, component in (('scheduler', scheduler), ('grad_scaler', grad_scaler),
                                    ('warmup_scheduler', warmup_scheduler), ('early_stop', early_stop)):
                if component is not None and checkpoint.get(name) is None:
                    raise ValueError(f'Checkpoint has no {name} state; reset optimizer or use pre_model')
    weights = dict(checkpoint['model'])
    # Strip wrapper prefixes only, never remap parameters by insertion order.
    for prefix in ('module.', '_fsdp_wrapped_module.'):
        if weights and all(key.startswith(prefix) for key in weights):
            weights = {key[len(prefix):]: value for key, value in weights.items()}
    with _context(model):
        model.load_state_dict(weights, strict=True)
        if resume and not reset_optimizer:
            optim_state = checkpoint['optimizer']
            if isinstance(model, FSDP):
                optim_state = FSDP.optim_state_dict_to_load(model, optimizer, optim_state)
            optimizer.load_state_dict(optim_state)
    if resume and not reset_optimizer:
        for name, component in (('scheduler', scheduler), ('grad_scaler', grad_scaler),
                                ('warmup_scheduler', warmup_scheduler), ('early_stop', early_stop)):
            if component is not None and checkpoint.get(name) is not None:
                component.load_state_dict(checkpoint[name])
        if version < 2:
            warnings.warn('Legacy checkpoint: scaler, warmup, early-stop and RNG state are unavailable')
    elif resume:
        warnings.warn('Explicit resume_reset_optimizer: keeping weights/iteration, resetting '
                      'optimizer, LR/warmup/scaler, best-loss and early-stop history')
        state = {}
    if resume and restore_rng and not reset_optimizer and checkpoint.get('rng_by_rank'):
        states = checkpoint['rng_by_rank']
        world = dist.get_world_size() if dist.is_initialized() else 1
        if len(states) == world:
            _restore_rng(states[_rank()])
        else:
            warnings.warn('World size changed: full optimizer reshards, but RNG streams restart')
    state['legacy_checkpoint'] = version < 2
    result = (model, optimizer, scheduler, iteration)
    return (*result, state) if return_state else result
