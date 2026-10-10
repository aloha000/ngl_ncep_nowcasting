"""CPU regression tests for the actual gradient-update block in train_FSDP.py.

Extract the block to avoid importing the training entry point (wandb/datasets).
The model double checks the FSDP clipping API; this is not a multi-GPU test.
Run: python -m unittest discover -s test -p test_gradient_clipping.py -v
"""
import ast
import os
from pathlib import Path
import unittest

os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')
import torch


def training_update_block():
    path = Path(__file__).resolve().parents[1] / 'main_code/train_FSDP.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'train_one_epoch')
    loop = next(node for node in function.body if isinstance(node, ast.For))
    start = next(i for i, node in enumerate(loop.body)
                 if ast.unparse(node) == 'grad_scaler.scale(loss).backward()')
    stop = next(i for i, node in enumerate(loop.body)
                if ast.unparse(node) == 'grad_scaler.update()')
    return compile(ast.Module(body=loop.body[start:stop + 1], type_ignores=[]),
                   str(path), 'exec')


class ClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.norm_before_clip = None

    def clip_grad_norm_(self, max_norm):
        self.norm_before_clip = torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=max_norm)
        return self.norm_before_clip


class GradientClippingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.update_block = training_update_block()

    def run_update(self, gradient, scale):
        model = ClipModel()
        # SGD makes the applied gradient directly observable in the update.
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        scaler = torch.amp.GradScaler('cpu', init_scale=scale)
        loss = (model.weight * torch.tensor(gradient)).sum()
        exec(self.update_block, dict(model=model, optimizer=optimizer,
                                     grad_scaler=scaler, loss=loss, torch=torch))
        return model, scaler

    def test_small_gradient_is_not_clipped(self):
        for scale in (1.0, 1024.0, 65536.0):
            with self.subTest(scale=scale):
                model, _ = self.run_update([3.0, 4.0], scale)
                torch.testing.assert_close(model.norm_before_clip, torch.tensor(5.0))
                torch.testing.assert_close(model.weight, torch.tensor([-3.0, -4.0]))

    def test_large_gradient_clips_at_true_threshold(self):
        for scale in (1.0, 1024.0, 65536.0):
            with self.subTest(scale=scale):
                model, _ = self.run_update([60.0, 80.0], scale)
                torch.testing.assert_close(model.norm_before_clip, torch.tensor(100.0))
                torch.testing.assert_close(model.weight, torch.tensor([-19.2, -25.6]))
                torch.testing.assert_close(model.weight.norm(), torch.tensor(32.0))

    def test_nonfinite_gradient_skips_update(self):
        for nonfinite in (float('inf'), float('nan')):
            with self.subTest(nonfinite=nonfinite):
                model, scaler = self.run_update([nonfinite, 1.0], 65536.0)
                torch.testing.assert_close(model.weight, torch.zeros(2))
                self.assertEqual(scaler.get_scale(), 32768.0)


if __name__ == '__main__':
    unittest.main()
