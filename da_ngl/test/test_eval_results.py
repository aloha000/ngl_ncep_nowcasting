"""CPU regression tests: python -m unittest discover -s da_ngl/test -p test_eval_results.py."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main_code'))
from eval_results import (EvaluationDataset, MyDataset, SampleMeanLoss,
                          json_value, load_model, set_eval_precision)
from main.model import mae


def fake_init(ds, **kwargs):
    dates = pd.date_range('2025-09-30', periods=5, freq='6h')
    ds.dates = dates
    ds.read_era5 = SimpleNamespace(dates=dates[:-1])
    ds.read_fcst = SimpleNamespace(dates=dates - pd.Timedelta(hours=24), fcst_step=24)
    ds.read_obs = SimpleNamespace(dates=dates, add_fuxi_ztd=True, fx_times=dates[:-1])


class DatasetTests(unittest.TestCase):
    def dataset(self, initializer=fake_init):
        with patch.object(MyDataset, '__init__', initializer):
            return EvaluationDataset()

    def test_filter_missing_endpoint_and_no_padding(self):
        ds = self.dataset()
        self.assertEqual(ds.requested_samples, 5)
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds.excluded_dates, [
            {'date': '2025-10-01T00:00:00', 'missing': ['era5', 'fuxi_ztd']}])
        with patch.object(MyDataset, 'prepare_data', lambda self, idx: (idx, idx, idx)):
            loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=False)
            self.assertEqual([v for batch in loader for v in batch[0].tolist()], [0, 1, 2, 3])

    def test_each_required_time_axis_is_checked(self):
        def initialize(ds, **kwargs):
            fake_init(ds)
            ds.read_fcst.dates = ds.read_fcst.dates.delete(1)
            ds.read_obs.dates = ds.read_obs.dates.delete(2)
        ds = self.dataset(initialize)
        self.assertEqual(len(ds), 2)
        self.assertEqual([v['missing'] for v in ds.excluded_dates],
                         [['fcst'], ['obs'], ['era5', 'fuxi_ztd']])

    def test_missing_read_raises_without_random_retry(self):
        ds = self.dataset()
        with patch.object(MyDataset, 'prepare_data', return_value=(None, None, None)), \
                patch.object(MyDataset, '_rand_another', side_effect=AssertionError('random retry')):
            with self.assertRaisesRegex(RuntimeError, '不随机替换'):
                ds[0]

    def test_empty_valid_set_fails_early(self):
        def initialize(ds, **kwargs):
            fake_init(ds)
            ds.read_era5.dates = pd.DatetimeIndex([])
        with self.assertRaisesRegex(ValueError, '没有完整'):
            self.dataset(initialize)

    def test_no_obs_reader_and_disabled_extra_channel(self):
        def no_obs(ds, **kwargs):
            fake_init(ds)
            ds.read_obs = None
        self.assertEqual(len(self.dataset(no_obs)), 4)
        def no_extra(ds, **kwargs):
            fake_init(ds)
            ds.read_era5.dates = ds.dates
            ds.read_obs.add_fuxi_ztd = False
        self.assertEqual(len(self.dataset(no_extra)), 5)


class LossTests(unittest.TestCase):
    def test_partial_batch_sample_weighting(self):
        losses = SampleMeanLoss()
        losses.update(1, 2, 8)
        losses.update(3, 4, 2)
        np.testing.assert_allclose(losses.means(), [1.4, 2.4])
        self.assertEqual(losses.n_samples, 10)
        self.assertEqual(losses.n_batches, 2)

    def test_actual_masked_loss_invariant_to_batch_partition(self):
        torch.manual_seed(42)
        target = torch.randn(11, 1, 2, 3, 4)
        background = torch.randn_like(target)
        analysis = torch.randn_like(target)
        loss_fn = mae(lat=np.array([37., 45., 55.]))
        loss_fn.set_cell_mask(np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]]))
        for dtype in (torch.float16, torch.float32):
            bg, an, lab = [x.to(dtype) for x in (background, analysis, target)]
            expected = [float(loss_fn(bg, lab)), float(loss_fn(an, lab))]
            for size in (1, 3, 8, 11):
                losses = SampleMeanLoss()
                for start in range(0, 11, size):
                    b, a, t = [x[start:start + size] for x in (bg, an, lab)]
                    losses.update(loss_fn(b, t), loss_fn(a, t), len(b))
                np.testing.assert_allclose(losses.means(), expected, rtol=2e-6)

    def test_invalid_loss_is_not_reported(self):
        losses = SampleMeanLoss()
        with self.assertRaises(ValueError):
            losses.means()
        for bg, an, size in ((np.nan, 1, 2), (1, np.inf, 2), (1, 2, 0)):
            with self.assertRaises(ValueError):
                losses.update(bg, an, size)


class PrecisionAndCheckpointTests(unittest.TestCase):
    def test_parameter_dtype_changes_but_buffers_stay_fp32(self):
        model = torch.nn.Linear(2, 2)
        model.register_buffer('float_buffer', torch.ones(2))
        model.register_buffer('int_buffer', torch.ones(2, dtype=torch.long))
        self.assertEqual(set_eval_precision(model, 'fp16'), torch.float16)
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertEqual(model.float_buffer.dtype, torch.float32)
        self.assertEqual(model.int_buffer.dtype, torch.long)
        set_eval_precision(model, 'fp32')
        self.assertEqual(model.weight.dtype, torch.float32)

    def test_checkpoint_mismatch_fails(self):
        cfg = SimpleNamespace(model_bg_chans=2, model_obs_chans=1,
                              model_obs_frames=1, model_embed_dim=2, model_depth=(1, 1, 1))
        with patch('eval_results.AssimilationNetv6', return_value=torch.nn.Linear(2, 2)), \
                patch('eval_results.torch.load', return_value={'model': {}}):
            with self.assertRaises(RuntimeError):
                load_model(cfg, Path('unused.pth'), 'cpu')

    def test_json_tensor_and_config_values(self):
        self.assertEqual(json_value({'a': torch.tensor([1.]), 'b': (np.int64(2), Path('x'))}),
                         {'a': [1.], 'b': [2, 'x']})


if __name__ == '__main__':
    unittest.main()
