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
    ws = float(getattr(cfg, 'loss_station_weight', 1.0))
    wn = float(getattr(cfg, 'loss_nostation_weight', 1.0))
    if (ws, wn) != (1.0, 1.0):
        tag += f'_w{ws:g}-{wn:g}'
    if getattr(cfg, 'include_fuxi_tp', False):
        tag += '_bgtp'
    if getattr(cfg, 'zero_obs', False):
        tag += '_zeroobs'
    mode = str(getattr(cfg, 'obs_mode', 'absolute')).lower()
    if mode != 'absolute':
        tag += f'_{mode}'
    lam = float(getattr(cfg, 'lambda_obs', 0.0) or 0.0)
    if lam:
        tag += f'_oc{lam:g}'
        if getattr(cfg, 'obs_freeze_zhd', False):
            tag += '_frzzhd'
    if getattr(cfg, 'obs_debias', False):
        tag += '_debias'
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


def station_cell_mask(cfg):
    """(H, W) bool mask, True on the 1378 cells that hold a GNSS station.

    The NGL store's static ``mask`` is True where a cell is *masked out* (no
    station) -- polarity is checked against the store's ``station`` array so a
    flipped mask cannot slip through silently.
    """
    import numpy as np
    import zarr

    g = zarr.open(str(cfg.ngl_zarr), 'r')
    mask = np.asarray(g['mask'][:]).astype(bool)          # True = no station
    station = np.asarray(g['station'][:])
    has = np.array([[str(v).strip() not in ('', 'nan', 'None') for v in row]
                    for row in station])
    if has.shape != mask.shape or not np.array_equal(has, ~mask):
        raise ValueError(f'station mask polarity check failed on {cfg.ngl_zarr}')
    return ~mask


def station_cell_weight(cfg):
    """(H, W) float32 loss weights: ``loss_station_weight`` on station cells,
    ``loss_nostation_weight`` everywhere else."""
    import numpy as np

    m = station_cell_mask(cfg)
    return np.where(m, float(getattr(cfg, 'loss_station_weight', 1.0)),
                    float(getattr(cfg, 'loss_nostation_weight', 1.0))).astype('float32')


def bg_chans(cfg):
    """Background channels implied by ``cfg.include_fuxi_tp``.

    69 ERA5/FuXi state channels, plus FuXi's own ``tp`` when the switch is on
    (the store then needs 70 channels, i.e. it must be built with
    ``build_fuxi_zarr.py --with-tp``).  ``model_bg_chans`` in configs.py is
    evaluated at import time, before ``--set include_fuxi_tp=true`` is applied,
    so the dataset and anything building the model must come back through here.
    """
    return 69 + int(bool(getattr(cfg, 'include_fuxi_tp', False)))


def station_geometry(cfg):
    """``(iy, ix, height_m, station_id)`` for the station cells, in map order.

    Exactly the construction ``preprocessing/build_ztd_fuxi_zarr.py`` uses: the
    NGL grid map gives one station per cell, and the height comes from
    ``ngl_europe_stations.parquet`` (ETOPO is *not* used -- a 100 m height error
    is already ~12 hPa of surface pressure).  Kept here so the torch ZTD
    operator, the de-bias builder and the evaluation index the same cells in the
    same order.
    """
    import numpy as np
    import pandas as pd

    dsit = os.path.dirname(str(cfg.ngl_zarr))
    grid_map = os.path.join(dsit, 'ngl_europe_0p25_80x120_station_grid_map.parquet')
    stations = os.path.join(dsit, 'ngl_europe_stations.parquet')
    mp = pd.read_parquet(grid_map)
    st = pd.read_parquet(stations)
    cells = mp[~mp['mask'].astype(bool) & mp['station_id'].notna()].reset_index(drop=True)
    cells = cells.merge(st[['gnss_station_id', 'height_m']],
                        left_on='station_id', right_on='gnss_station_id', how='left')
    lat_axis = np.asarray(cfg.lat, dtype=np.float64)
    lon_axis = np.asarray(cfg.lon, dtype=np.float64)
    iy = np.array([int(np.abs(lat_axis - v).argmin()) for v in cells['lat'].values])
    ix = np.array([int(np.abs(lon_axis - v).argmin()) for v in cells['lon'].values])
    if (not np.allclose(lat_axis[iy], cells['lat'].values, atol=1e-6)
            or not np.allclose(lon_axis[ix], cells['lon'].values, atol=1e-6)):
        raise ValueError(f'{grid_map}: station coordinates are not on the configured grid')
    height = cells['height_m'].values.astype(np.float64)
    if not np.isfinite(height).all():
        raise ValueError(f'{stations}: some station cells have no height_m')
    if len(set(zip(iy.tolist(), ix.tolist()))) != iy.size:
        raise ValueError('two stations landed on the same grid cell')
    return iy, ix, height, cells['station_id'].astype(str).values


def obs_debias_path(cfg):
    """Where the per-station static innovation bias is cached.

    Built by ``preprocessing/build_obs_debias.py``; one file per lead time
    because ``H(bg)`` -- and therefore the bias -- depends on the lead.
    """
    explicit = getattr(cfg, 'obs_debias_file', None)
    if explicit:
        return str(explicit)
    lead = int(getattr(cfg, 'fcst_step', 1)) * 6
    return os.path.join(os.path.dirname(str(cfg.ngl_zarr)),
                        f'obs_debias_lead{lead}h.npz')


def load_obs_debias(cfg):
    """``(bias_mm, station_id)`` for this lead, validated against the grid map.

    ``bias_mm`` is the train-split mean of ``obs - H(bg)`` per station [mm];
    subtracting it removes the 1378 constant offsets the network would otherwise
    have to learn before it can see the time-varying part of the innovation.
    """
    import numpy as np

    path = obs_debias_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'obs_debias=True needs {path}; build it with '
            f'preprocessing/build_obs_debias.py using the same --set fcst_step / '
            f'ztd_fuxi_zarr as the run')
    data = np.load(path, allow_pickle=False)
    bias = np.asarray(data['bias_mm'], dtype=np.float32)
    _, _, _, station_id = station_geometry(cfg)
    stored = np.asarray(data['station_id']).astype(str)
    if bias.shape != station_id.shape or not np.array_equal(stored, station_id):
        raise ValueError(f'{path}: station order does not match the grid map; rebuild it')
    return bias, station_id


def obs_debias_grid(cfg):
    """``(H, W)`` float32 map of the per-station bias, 0 outside the station cells.

    This is what the dataset adds to ``H(bg)``: ``obs - (H(bg) + b_s)`` is the
    de-biased innovation (HANDOFF 11.8 item 2).
    """
    import numpy as np
    import zarr

    bias, _ = load_obs_debias(cfg)
    iy, ix, _, _ = station_geometry(cfg)
    shape = tuple(int(v) for v in zarr.open(str(cfg.ngl_zarr), 'r')['mask'].shape)
    grid = np.zeros(shape, dtype=np.float32)
    grid[iy, ix] = bias
    return grid


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
