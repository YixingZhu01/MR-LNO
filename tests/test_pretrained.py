import importlib.util
import pathlib
import sys
import unittest
import os
import subprocess
import tempfile

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestPretrained(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('pretrained'),
                             'pretrained inference entry point is missing')
        import pretrained
        return pretrained

    def test_physical_input_is_encoded_without_mutation(self):
        module = self.module()
        x = np.ones((1, 4, 128, 128), dtype=np.float32)
        x[:, 2:] = np.e
        encoded = module.encode_input(x)
        np.testing.assert_allclose(encoded[:, 2:], 1, rtol=1e-6)
        self.assertAlmostEqual(float(x[0, 2, 0, 0]), np.e, places=6)

    def test_invalid_inputs_are_rejected(self):
        module = self.module()
        for x in (np.ones((4, 128, 128)), np.zeros((1, 4, 128, 128)),
                  np.full((1, 4, 128, 128), np.nan)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                module.encode_input(x)

    def test_rollout_is_cumulative_and_decodes_physical_fields(self):
        module = self.module()
        class Increment(torch.nn.Module):
            def forward(self, x, stages=None):
                return x + stages
        x = np.ones((1, 4, 128, 128), dtype=np.float32)
        result = module.rollout(Increment(), x, steps=2, stages=2)
        self.assertEqual(result.shape, (2, 1, 4, 128, 128))
        self.assertEqual(float(result[1, 0, 0, 0, 0]), 5)
        self.assertAlmostEqual(float(result[1, 0, 2, 0, 0]), np.exp(4), places=4)

    def test_invalid_rollout_counts_are_rejected(self):
        module = self.module()
        for steps, stages in ((0, 3), (1, 0), (1, 4)):
            with self.subTest(steps=steps, stages=stages), self.assertRaises(ValueError):
                module.rollout(torch.nn.Identity(), np.ones((1, 4, 128, 128)), steps, stages)

    def test_published_checkpoint_cli_round_trip_and_overwrite_guard(self):
        module = self.module()
        self.assertTrue(module.DEFAULT_CHECKPOINT.is_file())
        with tempfile.TemporaryDirectory() as directory:
            initial = pathlib.Path(directory) / 'initial.npz'
            output = pathlib.Path(directory) / 'prediction.npz'
            np.savez(initial, input=np.ones((1, 4, 128, 128), dtype=np.float32))
            command = [sys.executable, '-B', str(ROOT / 'pretrained.py'),
                       '--input', str(initial), '--output', str(output),
                       '--device', 'cpu', '--stages', '2', '--steps', '2']
            env = dict(os.environ, OMP_NUM_THREADS='2')
            run = subprocess.run(command, cwd=directory, env=env,
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(data['prediction'].shape, (2, 1, 4, 128, 128))
                self.assertTrue(np.isfinite(data['prediction']).all())
            original = output.read_bytes()
            repeated = subprocess.run(command, cwd=directory, env=env,
                                      capture_output=True, text=True, timeout=60)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn('Output already exists', repeated.stderr)
            self.assertEqual(output.read_bytes(), original)
