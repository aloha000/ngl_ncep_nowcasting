"""CPU regression checks: python -m unittest discover -s test -p test_training_resume.py."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'main_code'))
from main.utils import apply_overrides, save_checkpoint, load_checkpoint
from main.model.build_optimizer import EarlyStopping, WarmupScheduler


def config():
    spec = importlib.util.spec_from_file_location('test_config', ROOT/'main_code/configs.py')
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


class ConfigTests(unittest.TestCase):
    def test_dependencies_and_types(self):
        cfg = config()
        apply_overrides(cfg, ['add_fuxi_ztd=true', 'obs_frames=73', 'model_depth=(1,1,1)',
                              'resume_model=None', 'num_workers=0', 'loss_mask=none',
                              'num_iteration=123', 'early_stop={"patience":3,"min_delta":0.0001}'])
        self.assertEqual((cfg.obs_channum, cfg.model_obs_chans, cfg.model_obs_frames), (2, 4, 73))
        self.assertEqual(cfg.model_depth, (1, 1, 1))
        self.assertIsNone(cfg.resume_model)
        self.assertFalse(cfg.persistent_workers)
        self.assertEqual(cfg.num_teration, 123)
        self.assertEqual(cfg.early_stop['patience'], 3)

    def test_root_and_explicit_path_precedence(self):
        for items in (['DATASET_DIR=/tmp/new', 'obs_dir=/tmp/special.zarr'],
                      ['obs_dir=/tmp/special.zarr', 'DATASET_DIR=/tmp/new']):
            cfg = config()
            apply_overrides(cfg, items)
            self.assertEqual(cfg.era5_dir, '/tmp/new/label_europe_0p25.zarr')
            self.assertEqual(cfg.obs_dir, '/tmp/special.zarr')
            self.assertEqual(cfg.obs_stat_dir, cfg.obs_dir)

    def test_fail_fast_and_atomic(self):
        for items in (['batch_szie=4'], ['batch_size=0'], ['zero_obs=maybe'],
                      ['add_fuxi_ztd=true', 'obs_channum=1'], ['obs_frames=73','model_obs_frames=25'],
                      ['num_workers=0','persistent_workers=true'], ['early_stop=false'],
                      ['resume_model=a','pre_model=b']):
            cfg = config()
            old = cfg.add_fuxi_ztd
            with self.assertRaises(SystemExit):
                apply_overrides(cfg, items)
            self.assertEqual(cfg.add_fuxi_ztd, old)

    def test_early_stop_off_and_scheduler_override(self):
        cfg = config()
        apply_overrides(cfg, ['early_stop=None','scheduler=MultiStepLR','step_size=[2,4]',
                              'loss_fn=mse'])
        self.assertIsNone(cfg.early_stop)
        self.assertEqual(cfg.step_size, [2,4])
        self.assertEqual(type(cfg.loss_fn).__name__, 'mse')


class CheckpointTests(unittest.TestCase):
    def components(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        warmup = WarmupScheduler(optimizer, start_lr=.001, stop_lr=.01, warmup_steps=4)
        early = EarlyStopping(patience=5, min_delta=.001)
        scaler = torch.amp.GradScaler('cpu', init_scale=16)
        return model, optimizer, scheduler, warmup, early, scaler

    @staticmethod
    def step(model, opt, scheduler, scaler):
        opt.zero_grad()
        x=torch.arange(6,dtype=torch.float32).reshape(2,3)/10
        scaler.scale(model(x).square().mean()).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

    def test_round_trip_and_next_update(self):
        a, opt, sch, warm, early, scaler = self.components()
        self.step(a,opt,sch,scaler); warm(); early(1.); early(1.1)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pth'
            save_checkpoint(path,a,7,opt,sch,grad_scaler=scaler,warmup_scheduler=warm,
                            early_stop=early,training_state={'next_epoch':3,'best_val':1.})
            next_random=torch.rand(4)
            self.step(a,opt,sch,scaler)
            b, opt2, sch2, warm2, early2, scaler2 = self.components()
            result=load_checkpoint(path,b,opt2,sch2,grad_scaler=scaler2,
                                   warmup_scheduler=warm2,early_stop=early2,return_state=True)
            self.assertIs(result[0],b)
            self.assertEqual(result[3],7)
            self.assertEqual(result[4]['next_epoch'],3)
            self.assertEqual(early2.counter,1)
            self.assertEqual(warm2.count,1)
            torch.testing.assert_close(torch.rand(4),next_random)
            self.step(b,opt2,sch2,scaler2)
            for p,q in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            self.assertEqual(sch.state_dict(),sch2.state_dict())

    def test_legacy_iteration_and_parameter_name_matching(self):
        model,*_=self.components()
        weights=dict(reversed(list(model.state_dict().items())))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.pth'
            torch.save({'model':weights,'iteration':{'iteration':123}},path)
            target,*_=self.components()
            result=load_checkpoint(path,target)
            self.assertEqual(result[3],123)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,target.state_dict()[key])
            with self.assertRaises(ValueError):
                load_checkpoint(path,target,torch.optim.Adam(target.parameters()))

    def test_strict_missing_weights(self):
        model,*_=self.components()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.pth'
            torch.save({'model':{},'iteration':0},path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path,model)

    def test_resume_rejects_changed_scheduler(self):
        model,opt,sch,*_=self.components()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pth'
            save_checkpoint(path,model,0,opt,sch)
            with self.assertRaisesRegex(ValueError,'Scheduler type changed'):
                load_checkpoint(path,model,opt,None)


if __name__ == '__main__':
    unittest.main()
