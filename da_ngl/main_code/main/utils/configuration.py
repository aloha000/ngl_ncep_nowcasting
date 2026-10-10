"""Parse CLI overrides and resolve the training configuration dependencies."""
import ast
import json
import types
from pathlib import Path

import numpy as np


def _coerce(text):
    text = text.strip()
    low = text.lower()
    if low in ('true', 'yes', 'on'):
        return True
    if low in ('false', 'no', 'off'):
        return False
    if low in ('none', 'null'):
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError):
            pass
    return text


def apply_overrides(cfg, items):
    """Apply atomically: explicit paths win; inconsistent derived fields fail."""
    values = dict(vars(cfg))
    explicit = {}
    optional = {'start_iteration': 0, 'resume_reset_optimizer': False,
                'start_lr': 1e-8, 'stop_lr': 1e-4, 'warmup_rate': .05,
                'step_size': 2}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f'--set requires KEY=VALUE, got {item!r}')
        key, raw = item.split('=', 1)
        key = key.strip()
        if key == 'num_iteration':  # public spelling; keep the legacy config name
            key = 'num_teration'
        template = values.get(key, optional.get(key))
        if (key.startswith('_') or key not in values and key not in optional
                or isinstance(template, types.ModuleType)
                or callable(template) and key != 'loss_fn'):
            raise SystemExit(f'Unknown or non-overridable config key: {key}')
        value = _coerce(raw)
        # Preserve numeric-looking experiment names and other string settings.
        if isinstance(template, str):
            if value is None and key != 'scheduler':
                value = raw.strip()
            elif value is not None:
                value = str(value) if not isinstance(value, (dict, list, tuple)) else value
        if isinstance(template, bool) and not isinstance(value, bool):
            raise SystemExit(f'{key} requires true/false')
        explicit[key] = value
    values.update(explicit)
    for key in ('DATASET_DIR', 'work_dir', 'model_id', 'era5_dir', 'fcst_dir', 'obs_dir', 'obs_stat_dir', 'ztd_fuxi_zarr'):
        if not isinstance(values.get(key), (str, Path)) or not str(values[key]):
            raise SystemExit(f'{key} must be a nonempty path/string')

    def derive(key, value):
        actual = tuple(explicit[key]) if key in explicit and isinstance(value, tuple) and isinstance(explicit[key], list) else explicit.get(key)
        if key in explicit and actual != value:
            raise SystemExit(f'{key}={explicit[key]!r} conflicts with derived value {value!r}')
        values[key] = value

    if 'DATASET_DIR' in explicit:
        old_root, new_root = Path(cfg.DATASET_DIR), Path(values['DATASET_DIR'])
        for key in ('era5_dir', 'fcst_dir', 'obs_dir', 'obs_stat_dir', 'ztd_fuxi_zarr'):
            if key not in explicit and hasattr(cfg, key):
                try:
                    values[key] = str(new_root / Path(getattr(cfg, key)).relative_to(old_root))
                except ValueError:
                    pass  # a deliberately external/custom path
    if 'obs_dir' in explicit and 'obs_stat_dir' not in explicit:
        values['obs_stat_dir'] = values['obs_dir']
    derive('obs_channum', 1 + int(values['add_fuxi_ztd']))
    derive('model_obs_chans', values['obs_channum'] + 2)
    derive('model_obs_frames', values['obs_frames'])
    for key in ('lat', 'lon'):
        axis = np.asarray(values[key], dtype=float)
        if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all():
            raise SystemExit(f'{key} must be a finite, nonempty 1D coordinate array')
        values[key] = axis
    derive('grid_hw', (len(values['lat']), len(values['lon'])))

    for key in ('obs_frames', 'batch_size', 'num_teration', 'model_bg_chans',
                'model_embed_dim', 'prefetch_factor', 'fcst_step'):
        if not isinstance(values[key], int) or isinstance(values[key], bool) or values[key] <= 0:
            raise SystemExit(f'{key} must be a positive integer')
    for key in ('num_workers', 'loss_halo_cells', 'start_iteration'):
        if key in values and (not isinstance(values[key], int) or isinstance(values[key], bool) or values[key] < 0):
            raise SystemExit(f'{key} must be a nonnegative integer')
    depth = values['model_depth']
    if not isinstance(depth, (list, tuple)) or len(depth) != 3 or any(type(n) is not int or n < 1 for n in depth):
        raise SystemExit('model_depth requires three positive integers, e.g. --set "model_depth=(1,1,1)"')
    values['model_depth'] = tuple(depth)
    if values['num_workers'] == 0:
        if explicit.get('persistent_workers') is True:
            raise SystemExit('persistent_workers=true requires num_workers>0')
        values['persistent_workers'] = False
    if values.get('resume_model') and values.get('pre_model'):
        raise SystemExit('resume_model and pre_model are mutually exclusive')
    if values['loss_mask'] not in ('none', 'station_halo'):
        raise SystemExit('loss_mask must be none or station_halo')
    for key in ('resume_model', 'pre_model'):
        if values.get(key) is not None and not isinstance(values[key], str):
            raise SystemExit(f'{key} must be a checkpoint path or None')
    for key in ('dates_train_range', 'dates_val_range', 'dates_test_range'):
        dates = values[key]
        if not isinstance(dates, (list, tuple)) or len(dates) != 2:
            raise SystemExit(f'{key} requires a two-element list of YYYYMMDDHH strings')
        import datetime
        try:
            start, end = [datetime.datetime.strptime(str(d), '%Y%m%d%H') for d in dates]
        except ValueError as exc:
            raise SystemExit(f'Invalid {key}: {exc}') from exc
        if start >= end:
            raise SystemExit(f'{key}: start must precede end')
        values[key] = [str(d) for d in dates]
    early = values.get('early_stop')
    if early is not None:
        if (not isinstance(early, dict) or set(early) != {'patience', 'min_delta'}
                or type(early['patience']) is not int or early['patience'] <= 0
                or not isinstance(early['min_delta'], (int, float)) or early['min_delta'] < 0):
            raise SystemExit('early_stop requires {"patience": positive int, "min_delta": nonnegative number}, or None')
    for key in ('learning_rate', 'weight_decay'):
        if not isinstance(values[key], (int, float)) or not np.isfinite(values[key]) or values[key] < 0:
            raise SystemExit(f'{key} must be a finite nonnegative number')
    if values['warmup']:
        for key in ('start_lr', 'stop_lr', 'warmup_rate'):
            values.setdefault(key, optional[key])
        if not 0 < values['warmup_rate'] < 1 or not 0 <= values['start_lr'] <= values['stop_lr']:
            raise SystemExit('warmup requires 0<warmup_rate<1 and 0<=start_lr<=stop_lr')
    if values['scheduler'] in ('StepLR', 'MultiStepLR'):
        values.setdefault('step_size', optional['step_size'])
        step = values['step_size']
        if values['scheduler'] == 'StepLR' and (type(step) is not int or step <= 0):
            raise SystemExit('StepLR needs a positive integer step_size')
        if values['scheduler'] == 'MultiStepLR' and (not isinstance(step, (list, tuple)) or any(type(n) is not int or n <= 0 for n in step)):
            raise SystemExit('MultiStepLR needs a list of positive integer milestones in step_size')
    elif values['scheduler'] not in ('CosineAnnealingLR', None):
        raise SystemExit('Unsupported scheduler')
    if values['opt_type'] not in ('Adam', 'AdamW', 'SGD'):
        raise SystemExit('opt_type must be Adam, AdamW, or SGD')
    if 'loss_fn' in explicit or 'lat' in explicit:
        from ..model import mae, mse
        kind = explicit.get('loss_fn', type(cfg.loss_fn).__name__)
        if kind not in ('mae', 'mse'):
            raise SystemExit('loss_fn override must be mae or mse')
        values['loss_fn'] = {'mae': mae, 'mse': mse}[kind](lat=values['lat'])
    for key, value in values.items():
        if not key.startswith('__'):
            setattr(cfg, key, value)
    for key in explicit:
        print(f'[override] {key} = {getattr(cfg, key)!r}')
    return cfg
