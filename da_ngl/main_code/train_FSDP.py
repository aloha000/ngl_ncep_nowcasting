import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import importlib
from main.utils import (load_checkpoint, save_checkpoint, create_logger, run_dir,
                        apply_overrides, station_halo_mask, build_dataloader)
from main.model import AssimilationNetv6, get_parameter_number, build_optimizer, EarlyStopping, WarmupScheduler
import time
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, 
    MixedPrecision, 
    ShardingStrategy,    
)
import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="linan_nilan-shanghai-academy-of-ai4s",
    # Set the wandb project where this run will be logged.
    project="my-awesome-project",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 'warmup',
        "architecture": "CNN",
        "dataset": "mydata",
        "epochs": 18,
    },
)


def __init__(rank, configs, master_port, world_size, overrides=None):
    # 加载config
    xconfig = importlib.import_module(configs)
    apply_overrides(xconfig, overrides)      # --model_id / --set 在这里生效
    # 创建log目录
    
    if rank == 0:
        os.makedirs(os.path.join(run_dir(xconfig), 'model'), exist_ok=True)
        os.makedirs(os.path.join(run_dir(xconfig), 'logs'), exist_ok=True)

    # 创建log文件
    xconfig.logger = create_logger(run_dir(xconfig), 'Train')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    
    seed = xconfig.rand_seed + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    xconfig.logger.info(f'[Work Dir]: {run_dir(xconfig)} \n'
                        f'[Configs]: {configs} \n'
                        f"[Word Size is]: {world_size} \n"
                        f'[Rank is]: {rank} \n'
                        f'[Device is]: {torch.cuda.current_device()} \n'
                        f'[Seed is]: {seed} \n')
    
    torch.cuda.set_device(rank)
    
    xconfig.rank = rank
    return xconfig


def process_obs(data, mean_std_dict, rank, cfg):
    # 参考版这里写死了 720x1440 和 6 帧；本版从数据和 cfg 取
    B, T, _, H, W = data.shape
    lat = torch.from_numpy(np.asarray(cfg.lat, dtype='float32')).cuda(rank)
    lat = torch.sin(lat / 180. * 3.1415).reshape(1, 1, H, 1).repeat(B, T, 1, 1, W).float()
    lon = torch.from_numpy(np.asarray(cfg.lon, dtype='float32')).cuda(rank)
    lon = torch.cos(lon / 180. * 3.1415).reshape(1, 1, 1, W).repeat(B, T, 1, H, 1).float()
    lat_lon = torch.concat([lat, lon], dim=2)
    
    data = data.cuda(rank)
    mask = 1 - torch.isnan(data).int()
    mask = torch.sum(mask, axis=2, keepdim=True) > 0
    data = torch.nan_to_num(data)
    
    instrument_mean = mean_std_dict['instrument'][0].cuda(rank)
    instrument_std = mean_std_dict['instrument'][1].cuda(rank)
    
    data = (data - instrument_mean) / instrument_std
    if getattr(cfg, 'zero_obs', False):
        # 消融：ZTD 置零（标准化空间里的 0 就是训练均值）。
        # 站点几何、帧掩膜、lat/lon 侧通道都保留，只是没有观测信号。
        data = torch.zeros_like(data)
    data = torch.concat([data, lat_lon], dim=2)
    data = data * mask
    return data


def process_bg(data, rank, hw=(80, 120)):
    data = data.cuda(rank)
    B, T, C, H, W = data.shape
    data = data.reshape(B, T * C, H, W)
    output = F.interpolate(
        data,
        size=hw,
        mode="bilinear",
        align_corners=False
    )
    output = output.reshape(B, T, C, hw[0], hw[1])
    return output


def train_one_epoch(cfg, model, rank, dataloader, optimizer, grad_scaler, warmup_scheduler=None, scheduler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    time_start = time.perf_counter()
    cfg.logger.info(f'[Epoch Iteration Number] ||| {len(dataloader)} ')
    
    mean_std_dict = dataloader.dataset.read_obs.mean_std_dict
    for _, (batch_fcst, batch_era5, batch_obs) in enumerate(dataloader):
        cfg.iteration += 1
        
        optimizer.zero_grad()
        # dist.barrier()
        
        time_data = time.perf_counter()
        
        batch_fcst = process_bg(batch_fcst, rank)
        batch_era5 = process_bg(batch_era5, rank)
        batch_obs = process_obs(batch_obs, mean_std_dict, rank, cfg)
        
        batch_fcst = batch_fcst.half()
        batch_era5 = batch_era5.half()
        batch_obs = batch_obs.half()
        
        time_data_proccess = time.perf_counter()
        batch_out = model(batch_fcst, batch_obs)
        
        loss = cfg.loss_fn(batch_out, batch_era5)
        grad_scaler.scale(loss).backward()
        # dist.barrier()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=32)
        # dist.barrier()
        grad_scaler.step(optimizer)
        # dist.barrier()
        grad_scaler.update()
        # dist.barrier()
        
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1
        cfg.lr.append(optimizer.param_groups[0]['lr'])
        
        time_model = time.perf_counter()
        cfg.logger.info(f"[LOCAL_RANK]: {rank} \n"
                        f"[Iteration]: {cfg.iteration} \n"
                        f"[Learning Rate]: {optimizer.param_groups[0]['lr']} \n"
                        f"[Batch loss]: {loss.item():.4f} \n"
                        f"[MaxMemory]: {torch.cuda.max_memory_allocated(batch_fcst.device) / 1024 ** 2}")
        
        if cfg.warmup:
            cfg.warmup = warmup_scheduler()
        elif scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        cfg.logger.info(f"[Data Load Time]: {time_data - time_start:.2f} \n"
                        f"[Data Proccess Time]: {time_data_proccess - time_data:.2f} \n"
                        f"[Run Model Time]: {time_model - time_data_proccess:.2f} \n")

        time_start = time.perf_counter()
        
        if cfg.iteration >= cfg.num_teration:
            break
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    epoch_loss = ddp_loss[0] / ddp_loss[1]
    epoch_loss = epoch_loss.item()
    cfg.logger.info(f"[LOCAL_RANK]: {rank}] \n"
                    f"[Iteration]: {cfg.iteration} \n"
                    f"[Learning Rate]: {optimizer.param_groups[0]['lr']}] \n"
                    f"[Epoch loss]: {epoch_loss:.4f}] \n"
                    f"[MaxMemory]: {torch.cuda.max_memory_allocated(batch_fcst.device) / 1024 ** 2}]")
    return model, epoch_loss


def evaluate(cfg, model, rank, dataloader, get_fcst_loss=False):
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)

    mean_std_dict = dataloader.dataset.read_obs.mean_std_dict
    for _, (batch_fcst, batch_era5, batch_obs) in enumerate(dataloader):

        batch_fcst = process_bg(batch_fcst, rank)
        batch_era5 = process_bg(batch_era5, rank)
        batch_obs = process_obs(batch_obs, mean_std_dict, rank, cfg)
        
        batch_fcst = batch_fcst.half()
        batch_era5 = batch_era5.half()
        batch_obs = batch_obs.half()

        if get_fcst_loss:
            batch_out = batch_fcst
        else:
            with torch.no_grad():
                batch_out = model(batch_fcst, batch_obs)
        loss = cfg.loss_fn(batch_out, batch_era5)
        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    epoch_loss = ddp_loss[0] / ddp_loss[1]
    epoch_loss = epoch_loss.item()
    cfg.logger.info('{:#^75}'.format('Evaluation'))
    cfg.logger.info(f"[LOCAL_RANK]: {rank}] \n"
                    f"[Iteration]: {cfg.iteration} \n"
                    f"[Epoch loss]: {epoch_loss:.4f}] \n"
                    f"[MaxMemory]: {torch.cuda.max_memory_allocated(batch_fcst.device) / 1024 ** 2}] \n"
                    f"[Get Fcst Loss]: {get_fcst_loss}]")
    cfg.logger.info('{:#^75}'.format('Evaluation'))
    return epoch_loss


def main(rank, configs, master_port, world_size, overrides=None):
    # 加载数据
    cfg = __init__(rank, configs, master_port, world_size, overrides)
    dataloader = build_dataloader(world_size=world_size,
                                  rank=rank,
                                  era5_dir=cfg.era5_dir, 
                                  fcst_dir=cfg.fcst_dir, 
                                  fcst_step=cfg.fcst_step, 
                                  obs_dir=cfg.obs_dir, 
                                  obs_frames=cfg.obs_frames, 
                                  obs_stat_dir=cfg.obs_stat_dir,
                                  obs_channum=cfg.obs_channum, 
                                  dates_range=cfg.dates_train_range,
                                  obs_frame_minutes=cfg.obs_frame_minutes,
                                  grid_hw=cfg.grid_hw,
                                  batch_size=cfg.batch_size, 
                                  num_workers=cfg.num_workers, 
                                  persistent_workers=cfg.persistent_workers, 
                                  prefetch_factor=cfg.prefetch_factor, 
                                  multiprocessing_context=cfg.multiprocessing_context, 
                                  pin_memory=cfg.pin_memory, 
                                  shuffle=True)
    dataloader_val = build_dataloader(world_size=world_size,
                                      rank=rank,
                                      era5_dir=cfg.era5_dir, 
                                      fcst_dir=cfg.fcst_dir, 
                                      fcst_step=cfg.fcst_step, 
                                      obs_dir=cfg.obs_dir, 
                                      obs_frames=cfg.obs_frames, 
                                      obs_stat_dir=cfg.obs_stat_dir,
                                      obs_channum=cfg.obs_channum, 
                                      dates_range=cfg.dates_val_range,
                                      obs_frame_minutes=cfg.obs_frame_minutes,
                                      grid_hw=cfg.grid_hw,
                                      batch_size=cfg.batch_size, 
                                      num_workers=cfg.num_workers, 
                                      persistent_workers=cfg.persistent_workers, 
                                      prefetch_factor=cfg.prefetch_factor, 
                                      multiprocessing_context=cfg.multiprocessing_context, 
                                      pin_memory=cfg.pin_memory, 
                                      shuffle=False)
    if getattr(cfg, 'zero_obs', False):
        cfg.logger.info('[Ablation] zero_obs=True: ZTD 通道置零（只保留站点几何/掩膜）')
    cfg.logger.info('{:#^75}'.format('Data Information'))
    cfg.logger.info(f'[Train Data Number]: {len(dataloader.dataset)} \n'
                    f'[Train Batch Number]: {len(dataloader)} \n'
                    f'[Train Batch Size]: {cfg.batch_size}')
    cfg.logger.info(f'[Val Data Number]: {len(dataloader_val.dataset)} \n'
                    f'[Val Batch Number]: {len(dataloader_val)} \n'
                    f'[Val Batch Size]: {cfg.batch_size}')
    cfg.logger.info('{:#^75}'.format('Data Information'))

    # 创建模型
    model = AssimilationNetv6(bg_chans=cfg.model_bg_chans,
                              obs_chans=cfg.model_obs_chans,
                              obs_frames=cfg.model_obs_frames,
                              embed_dim=cfg.model_embed_dim,
                              depth=cfg.model_depth).cuda(rank)
    cfg.logger.info(f'[Model Parameter Numbe]: {get_parameter_number(model)}')
    
    # 模型混合精度训练
    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    mp = MixedPrecision(
        param_dtype=dtype_dict['fp16'],
        reduce_dtype=dtype_dict['fp32'],
        buffer_dtype=dtype_dict['fp32'],
    )
    cfg.logger.info(f'[Cuda Current Device]: {torch.cuda.current_device()}')
    model = FSDP(model, 
                 mixed_precision=mp, 
                 device_id=torch.cuda.current_device(), 
                 sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
    grad_scaler = ShardedGradScaler()
    
    if cfg.warmup:
        warmup_steps = int(cfg.num_teration * cfg.warmup_rate)
        T_max = int(cfg.num_teration - warmup_steps)
        optimizer, scheduler = build_optimizer(opt_type=cfg.opt_type, 
                                               learning_rate=cfg.start_lr, 
                                               weight_decay=cfg.weight_decay, 
                                               model=model, 
                                               scheduler=cfg.scheduler,
                                               step_size=cfg.step_size if hasattr(cfg, 'step_size') else None,
                                               T_max=T_max if cfg.scheduler == 'CosineAnnealingLR' else None)
        warmup_scheduler = WarmupScheduler(optimizer=optimizer,
                                           start_lr=cfg.start_lr,
                                           stop_lr=cfg.stop_lr,
                                           warmup_steps=warmup_steps)
        cfg.logger.info(f'[Warm Up]: {warmup_steps}')
        cfg.logger.info(f'[T Max]: {T_max}')
    else:
        optimizer, scheduler = build_optimizer(opt_type=cfg.opt_type, 
                                               learning_rate=cfg.learning_rate, 
                                               weight_decay=cfg.weight_decay, 
                                               model=model, 
                                               scheduler=cfg.scheduler,
                                               step_size=cfg.step_size if hasattr(cfg, 'step_size') else None,
                                               T_max=cfg.num_teration if cfg.scheduler == 'CosineAnnealingLR' else None)
        warmup_scheduler = None

    # 加载模型参数
    start_iteration = 0
    if cfg.resume_model is not None:
        model, optimizer, scheduler, iteration = load_checkpoint(cfg.resume_model, model, optimizer, scheduler)
        start_iteration = iteration
        cfg.logger.info(f'[Resume Model] ||| {cfg.resume_model} ')
    elif cfg.pre_model is not None:
        if hasattr(cfg, 'start_iteration'):
            start_iteration = cfg.start_iteration
        model, _, _, _ = load_checkpoint(cfg.pre_model, model)
        cfg.logger.info(f'[Pre Model] ||| {cfg.pre_model} ')

    if hasattr(cfg, 'early_stop'):
        early_stop = EarlyStopping(**cfg.early_stop)

    # 训练
    cfg.lr = []
    cfg.loss_train = []
    cfg.loss_val = []
    cfg.iteration = start_iteration
    # 只保留 val 最好的那一个 checkpoint：先评估、只有 val 变好才覆盖同一个文件。
    # 文件名固定，所以磁盘上永远只有 1 个（含 optimizer 状态，可续训）。
    best_val = float('inf')
    best_iter = -1
    best_file = os.path.join(run_dir(cfg), 'model', 'best.pth')

    # 标签损失的格点掩膜：只监督 ZTD 站点周围的格点
    if str(getattr(cfg, 'loss_mask', 'none')).lower() == 'station_halo':
        halo = station_halo_mask(cfg)
        cfg.loss_fn.set_cell_mask(halo)
        if rank == 0:
            cfg.logger.info('[Loss] station_halo=%d: %d/%d 格参与损失（其余不计入）'
                            % (int(getattr(cfg, 'loss_halo_cells', 3)), int(halo.sum()), halo.size))
    else:
        cfg.loss_fn.set_cell_mask(None)

    # 背景场基线：batch_out = batch_fcst，跟模型权重无关，所以只算一次就够，
    # 这里算完既写 wandb（每个 epoch 都带上，画出来就是一条水平参考线），
    # 也复用为训练结束时的 [All Train/Val loss] 末位，省掉最后那两遍评估。
    bg_train = evaluate(cfg, model, rank, dataloader, get_fcst_loss=True)
    bg_val = evaluate(cfg, model, rank, dataloader_val, get_fcst_loss=True)
    if rank == 0:
        cfg.logger.info(f'[Background-only loss] train={bg_train:.4f} val={bg_val:.4f}')

    for epoch in range(1000):
        model, loss_train = train_one_epoch(cfg, model, rank, dataloader, optimizer, grad_scaler, warmup_scheduler, scheduler)

        loss_val = evaluate(cfg, model, rank, dataloader_val)

        if loss_val < best_val:
            best_val, best_iter = loss_val, cfg.iteration
            # summon_full_params 是集合操作，所有 rank 都要进这个上下文；
            # 只有 rank 0 真正写盘（save_checkpoint 内部也只让 rank 0 写）。
            with FSDP.summon_full_params(model):
                if rank == 0:
                    save_checkpoint(best_file, model, cfg.iteration, optimizer, scheduler)
                    cfg.logger.info(f'[Save Model] best val={loss_val:.4f} @ iter {cfg.iteration}'
                                    f' -> {best_file}')
        elif rank == 0:
            cfg.logger.info(f'[Best] val={loss_val:.4f} @ iter {cfg.iteration} '
                            f'-- keeping the earlier best {best_val:.4f} @ iter {best_iter}')
        
        if hasattr(cfg, 'early_stop'):
            early_stop(loss_val)
            if early_stop.early_stop:
                cfg.logger.info('{:#^75}'.format('Early Stop'))
                cfg.logger.info(f"Epoch {epoch + 1}: Loss did not improve from {early_stop.best_loss}, stop training")
                cfg.logger.info('{:#^75}'.format('Early Stop'))
                break

        cfg.loss_train.append(loss_train)
        cfg.loss_val.append(loss_val)
        
        run.log({"train_loss": loss_train, "val_loss": loss_val,
                 "background_train_loss": bg_train, "background_val_loss": bg_val})

        if cfg.iteration >= cfg.num_teration:
            break
        
    run.finish()

    # 复用开头算好的背景基线（模型无关），数值与原来结尾重算的一致
    cfg.loss_train.append(bg_train)
    cfg.loss_val.append(bg_val)
    
    if rank == 0:
        cfg.logger.info(f'[Best Model] val={best_val:.4f} @ iter {best_iter} -> {best_file}')
        cfg.logger.info(f"[All Train loss]: {cfg.loss_train}")
        cfg.logger.info(f"[All Val loss]: {cfg.loss_val}")
        np.save(os.path.join(run_dir(cfg), 'train_loss.npy'), np.array(cfg.loss_train))
        np.save(os.path.join(run_dir(cfg), 'val_loss.npy'), np.array(cfg.loss_val))
        np.save(os.path.join(run_dir(cfg), 'lr.npy'), np.array(cfg.lr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='configs')
    parser.add_argument('--master_port', type=str, default='22346', help='Local host addres')
    parser.add_argument('--model_id', type=str, default=None,
                        help='实验名；产物落在 {work_dir}/{model_id}/ 下')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='覆盖 configs 里的设置，例：--set zero_obs=true')
    args = parser.parse_args()
    world_size = max(torch.cuda.device_count(), 1)
    mp.spawn(
        main,
        args=(args.configs, args.master_port, world_size,
              ([f'model_id={args.model_id}'] if args.model_id else []) + list(args.set)),
        nprocs=world_size,
        join=True)