import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'main_code'))
from eval_channel_physical_mae import ChannelAccumulator, unit_scale
from main.model import mae


class PhysicalChannelTests(unittest.TestCase):
    def test_linear_units_and_nonlinear_precipitation(self):
        channels = ['msl','t850','z500','era5_tp']
        target = torch.zeros(2,1,4,2,2)
        prediction = torch.ones_like(target)
        weights = torch.tensor([[1.,2.],[0.,1.]])
        acc = ChannelAccumulator(weights,channels,[100000,273.15,50000,0],[100,2,9.80665,1])
        acc.update(prediction,target)
        norm, physical = acc.results()
        np.testing.assert_allclose(norm,1)
        np.testing.assert_allclose(physical,[1,2,1,np.e-1])
        self.assertEqual(unit_scale('r850'),('percentage_points',1))

    def test_batch_partition_and_reconstruction_of_loss(self):
        torch.manual_seed(42)
        labels = torch.randn(5,1,3,2,3).half()
        prediction = torch.randn_like(labels)
        latitude = np.array([35.,55.])
        mask = np.array([[1,0,1],[0,1,1]])
        loss_fn = mae(lat=latitude).set_cell_mask(mask)
        w = torch.cos(torch.deg2rad(torch.tensor(latitude).float()))[:,None]
        w = w/w.mean() * torch.tensor(mask).float()
        expected = float(loss_fn(prediction,labels))
        outputs = []
        for batch_size in (1,2,5):
            acc = ChannelAccumulator(w,['u850','t850','era5_tp'],[0,273,0],[2,3,.1])
            for start in range(0,5,batch_size):
                acc.update(prediction[start:start+batch_size],labels[start:start+batch_size])
            normal, physical = acc.results()
            self.assertAlmostEqual(normal.mean(),expected,places=6)
            outputs.append(physical)
        np.testing.assert_allclose(outputs[0],outputs[1],rtol=1e-12)
        np.testing.assert_allclose(outputs[0],outputs[2],rtol=1e-12)

    def test_log_mae_does_not_determine_millimeter_mae(self):
        w = torch.ones(1,1)
        low = ChannelAccumulator(w,['era5_tp'],[0],[1])
        high = ChannelAccumulator(w,['era5_tp'],[0],[1])
        low.update(torch.ones(1,1,1,1,1),torch.zeros(1,1,1,1,1))
        high.update(torch.full((1,1,1,1,1),3.),torch.full((1,1,1,1,1),2.))
        np.testing.assert_allclose(low.results()[0], high.results()[0])
        self.assertGreater(high.results()[1][0],low.results()[1][0]*7)


if __name__ == '__main__':
    unittest.main()
