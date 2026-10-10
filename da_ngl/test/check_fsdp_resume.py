"""Tiny two-GPU FSDP save/resume test, no training dataset or wandb required."""
import os
from pathlib import Path
import socket
import sys
import tempfile

os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                   StateDictType, FullStateDictConfig, ShardingStrategy)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'main_code'))
from main.utils import load_checkpoint,save_checkpoint


def worker(rank,world,port,directory):
    os.environ['MASTER_ADDR']='127.0.0.1'
    os.environ['MASTER_PORT']=str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=world)
    try:
        def build():
            torch.manual_seed(33)
            model=FSDP(torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.Linear(8,2)).cuda(rank),
                       device_id=rank,sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)
            opt=torch.optim.AdamW(model.parameters(),lr=.001)
            sch=torch.optim.lr_scheduler.StepLR(opt,step_size=2)
            scaler=ShardedGradScaler(init_scale=16)
            return model,opt,sch,scaler

        def step(model,opt,sch,scaler):
            opt.zero_grad()
            x=torch.arange(6,dtype=torch.float32,device=rank).reshape(2,3)/10+rank*.1
            scaler.scale(model(x).square().mean()).backward()
            scaler.step(opt);scaler.update();sch.step()

        a,opt,sch,scaler=build()
        step(a,opt,sch,scaler)
        path=Path(directory)/'last.pth'
        save_checkpoint(path,a,1,opt,sch,grad_scaler=scaler,training_state={'next_epoch':1})
        step(a,opt,sch,scaler)
        b,opt2,sch2,scaler2=build()
        result=load_checkpoint(path,b,opt2,sch2,grad_scaler=scaler2,return_state=True)
        assert result[0] is b and isinstance(b,FSDP) and result[3]==1
        step(b,opt2,sch2,scaler2)
        with FSDP.state_dict_type(a,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False)):
            expected=a.state_dict()
        with FSDP.state_dict_type(b,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False)):
            actual=b.state_dict()
        for key in expected:
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=1e-7)
        assert sch.state_dict()==sch2.state_dict()
        if rank==0:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['optimizer_format']=='fsdp_full'
            assert len(saved['optimizer']['state'])==4
            print('PASS: two-GPU FSDP full optimizer, wrapper retained, next update matches',flush=True)
            torch.save({'model':saved['model'],'iteration':{'iteration':1},
                        'optimizer':opt.state_dict(),'scheduler':sch.state_dict()},
                       Path(directory)/'legacy.pth')
        dist.barrier()
        c,opt3,sch3,scaler3=build()
        legacy=Path(directory)/'legacy.pth'
        try:
            load_checkpoint(legacy,c,opt3,sch3,grad_scaler=scaler3)
            raise AssertionError('Legacy optimizer must be rejected')
        except ValueError as exc:
            assert 'resume_reset_optimizer' in str(exc)
        result=load_checkpoint(legacy,c,opt3,sch3,grad_scaler=scaler3,
                               reset_optimizer=True,return_state=True)
        assert result[0] is c and result[3]==1 and not opt3.state
        if rank==0:
            print('PASS: legacy FSDP state rejected by default; explicit weights/iteration reset works',flush=True)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    if torch.cuda.device_count()<2:
        raise SystemExit('Need two visible GPUs')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='fsdp_resume_test_') as directory:
        mp.spawn(worker,args=(2,port,directory),nprocs=2,join=True)
