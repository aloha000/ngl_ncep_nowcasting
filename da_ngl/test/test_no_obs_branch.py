"""CPU regression tests for the structural observation ablation.

Run: python da_ngl/test/test_no_obs_branch.py
"""
import ast
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

CODE = Path(__file__).resolve().parents[1] / 'main_code'
sys.path.insert(0, str(CODE))
from main.model.assimilation import AssimilationNetv6
from main.utils.configuration import apply_overrides
from main.utils.utils_data import MyDataset


class NoObsBranchTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2000)

    def test_structure_side_input_and_backward(self):
        model = AssimilationNetv6(bg_chans=70, obs_chans=0, obs_frames=10,
                                  embed_dim=8, depth=(2, 2, 2))
        self.assertFalse(any('obs' in name for name, _ in model.named_parameters()))
        for block in (model.block_0, model.block_1, model.block_2):
            self.assertIsNone(block.obs_encoder)
            for fusion in block.fussion.module_list:
                self.assertFalse(hasattr(fusion, 'upsample_obs_0'))
                self.assertFalse(hasattr(fusion, 'upsample_obs_1'))
        side_inputs = []
        hook = model.block_0.side_encoder.register_forward_pre_hook(
            lambda module, args: side_inputs.append(args[0].detach().clone()))
        for h, w in ((32, 32), (80, 120)):
            bg = torch.randn(1, 1, 70, h, w)
            output = model(bg)
            self.assertEqual(output.shape, bg.shape)
            torch.testing.assert_close(side_inputs[-1][..., :h, :w], bg[:, 0])
            self.assertTrue(torch.isfinite(output).all())
            model.zero_grad(set_to_none=True)
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
        hook.remove()
        with self.assertRaises(ValueError):
            model(bg, torch.randn(1, 10, 4, 80, 120))

    def test_original_observation_mode(self):
        model = AssimilationNetv6(bg_chans=70, obs_chans=4, obs_frames=10,
                                  embed_dim=8, depth=(1, 1, 1))
        bg = torch.randn(1, 1, 70, 80, 120)
        self.assertEqual(model(bg, torch.randn(1, 10, 4, 80, 120)).shape, bg.shape)
        self.assertTrue(any('obs_encoder' in name for name, _ in model.named_parameters()))

    def test_config_switch(self):
        cfg = importlib.reload(importlib.import_module('configs'))
        apply_overrides(cfg, [])
        self.assertEqual((cfg.obs_channum, cfg.model_obs_chans), (0, 0))
        apply_overrides(cfg, ['remove_obs_branch=false'])
        self.assertEqual((cfg.obs_channum, cfg.model_obs_chans), (2, 4))
        apply_overrides(cfg, ['remove_obs_branch=true'])
        self.assertEqual((cfg.obs_channum, cfg.model_obs_chans), (0, 0))

    def test_dataset_does_not_open_observations(self):
        with patch('main.utils.utils_data.read_era5') as era, \
             patch('main.utils.utils_data.read_fcst') as fcst, \
             patch('main.utils.utils_data.read_obs', side_effect=AssertionError('obs opened')):
            era.return_value.prepare_data.return_value = torch.zeros(1, 70, 16, 16)
            fcst.return_value.prepare_data.return_value = torch.ones(1, 70, 16, 16)
            ds = MyDataset('unused', 'unused', 24, 'missing_obs', 10,
                           'missing_stats', 0, ['2022010100', '2022010118'],
                           grid_hw=(16, 16), add_fuxi_ztd=True,
                           ztd_fuxi_zarr='missing_fuxi_ztd')
            bg, label, obs = ds[0]
            self.assertIsNone(ds.read_obs)
            self.assertEqual(obs.shape, (10, 0, 16, 16))
            self.assertEqual(bg.shape, label.shape)

    def test_preprocessing_skips_coordinates_and_statistics(self):
        # Extract only this function: importing the entrypoint would start wandb.
        tree = ast.parse((CODE / 'train_FSDP.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'process_obs')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'process_obs', 'exec'), namespace)
        from types import SimpleNamespace
        empty = torch.empty(2, 10, 0, 80, 120)
        self.assertIs(namespace['process_obs'](empty, None, 0,
                                              SimpleNamespace(model_obs_chans=0)), empty)


if __name__ == '__main__':
    unittest.main()
