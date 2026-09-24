import os
import torch
import torch.distributed as dist
from collections import OrderedDict
import datetime
import logging


def create_logger(save_path, type='Train'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    now = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d%H")
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


def _coerce(value):
    """CLI 字符串 -> bool / int / float / str。"""
    low = str(value).strip().lower()
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


def station_halo_mask(cfg, halo_cells=None):
    """(H, W) bool：有 GNSS ZTD 站的格点 + 周围 ``loss_halo_cells`` 格（方形核）。

    NGL store 里的 ``mask`` 是"无站为 True"，这里取反再膨胀；halo<=0 时只留站格。
    """
    import numpy as np
    import zarr

    g = zarr.open(str(cfg.obs_dir), 'r')
    m = ~np.asarray(g['mask'][:]).astype(bool)
    h = int(getattr(cfg, 'loss_halo_cells', 3) if halo_cells is None else halo_cells)
    if h <= 0:
        return m
    from scipy import ndimage as ndi
    return np.asarray(ndi.binary_dilation(m, structure=np.ones((2 * h + 1, 2 * h + 1), bool)))


def apply_overrides(cfg, items):
    """把 ``--set KEY=VALUE`` 作用到 config 模块上（就地修改）。

    例：``--set zero_obs=true --set model_id=obszero``。train_FSDP 与 eval_results
    共用，保证训练和评测用的是同一套设置。
    """
    for item in (items or []):
        if '=' not in item:
            raise SystemExit(f'--set 需要 KEY=VALUE，收到 {item!r}')
        key, value = item.split('=', 1)
        key, value = key.strip(), _coerce(value.strip())
        setattr(cfg, key, value)
        print(f'[override] {key} = {value!r}')
    return cfg


def run_dir(cfg):
    """一次运行的所有产物都放在 ``{work_dir}/{model_id}/`` 下面。

    目录结构：
        {work_dir}/{model_id}/logs/         训练日志
        {work_dir}/{model_id}/model/        val-best checkpoint
        {work_dir}/{model_id}/*.npy         train/val loss、lr 曲线
        {work_dir}/{model_id}/eval_*        评估脚本的产物
    """
    return os.path.join(str(getattr(cfg, 'work_dir', '.')),
                        str(getattr(cfg, 'model_id', 'model')))


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def distributed():
    num_gpus = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
    return num_gpus > 1


def get_local_rank():
    if 'LOCAL_RANK' not in os.environ:
        return get_rank()
    else:
        return int(os.environ['LOCAL_RANK'])


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    # 加载checkpoint
    if hasattr(model, 'module'):
        model = model.module
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 获取模型的参数
    checkpoint_model_keys = [ikey for ikey in checkpoint['model'].keys()]
    # check = checkpoint['model']
    model_dict = model.state_dict()
    check_ = OrderedDict()
    for num, (k, v) in enumerate(model_dict.items()):
        check_[k] = checkpoint['model'][checkpoint_model_keys[num]]
    model.load_state_dict(check_)
    # 获取优化器的参数
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # 获取调度器的参数
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'iteration' in checkpoint:
        iteration = checkpoint['iteration']
    else:
        iteration = None
    return model, optimizer, scheduler, iteration


def save_checkpoint(file_name, model, iteration, optimizer=None, scheduler=None):
    save_dict = OrderedDict()
    if get_rank() == 0:
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


def times_step_gene(beg_time: str, end_time: str, step: int, test=False):
    beg_time = datetime.datetime.strptime(beg_time, '%Y%m%d%H')
    end_time = datetime.datetime.strptime(end_time, '%Y%m%d%H')
    time_interval = datetime.timedelta(hours=step)
    time_ls = []
    current_date = beg_time
    while current_date <= end_time:
        time_ls.append(current_date)
        current_date += time_interval
    return time_ls


def check_times_list(beg_time: str, end_time: str, step: int, x1_dir: str, x2_dir: str, y_dir: str):
    beg_time = datetime.datetime.strptime(beg_time, '%Y%m%d%H')
    end_time = datetime.datetime.strptime(end_time, '%Y%m%d%H')
    time_interval = datetime.timedelta(hours=step)
    time_ls = []
    current_date = beg_time
    while current_date <= end_time:
        current_str = datetime.datetime.strftime(current_date, '%Y%m%d%H')
        file_x1 = os.path.join(x1_dir, f"{current_str}.npy")
        file_x2 = os.path.join(x2_dir, f"{current_str}.npy")
        file_y = os.path.join(y_dir, f"{current_str}.npy")
        if os.path.exists(file_x1) & os.path.exists(file_x2) & os.path.exists(file_y):
            time_ls.append(current_date)
        current_date += time_interval


def build_InOut_time_ls(input_time_step, lead_time, time_list):
    input_time_ls = []
    output_time_ls = []
    inp_out_time_ls = []

    input_step = input_time_step
    lead_time = lead_time
    time_list = sorted(time_list)
    time_ls_list = [time_list[i: i + input_step] for i in range(len(time_list))]
    if len(time_ls_list[-1]) < input_step:
        time_ls_list = time_ls_list[:-1]
    for itime in time_ls_list:
        out_time = itime[-1] + datetime.timedelta(hours=lead_time)
        if out_time < time_list[-1]:
            input_time_ls.append(itime)
            output_time_ls.append([out_time])
            inp_out_time_ls.append([itime, [out_time]])
    return input_time_ls, output_time_ls, inp_out_time_ls
