import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from Data.DatasetNS import ComNS_Dataset2D


class TestCacheSplits(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        (self.root / 'demo').mkdir()
        for order in (1, 2, 3):
            savemat(self.root / 'demo' / f'demo_{order}.mat', {
                name: np.full((31, 4), float(order), dtype=np.float32)
                for name in ('u', 'v', 'rho', 'T')
            })
        # Keep real MAT loading, augmentation, cache I/O and batch generation.
        self.grid = patch('Data.DatasetNS.NG', 2)
        self.grid.start()
        self.addCleanup(self.grid.stop)

    def dataset(self, train):
        return ComNS_Dataset2D(str(self.root) + '/', 'demo', train, [3], 1)

    def assert_train_batch(self, dataset, order):
        self.assertEqual(dataset.orders_train, [order])
        inputs, targets = next(dataset.data_generator_series(1, 1, 1))
        self.assertEqual(tuple(inputs.shape), (1, 4, 2, 2))
        self.assertTrue((inputs[:, 2] == order).all())
        self.assertTrue((targets[0][:, 2] == order).all())

    def test_eval_then_train_preserves_eval_cache_and_produces_batch(self):
        evaluation = self.dataset([])
        original = pathlib.Path(evaluation.param_dir).read_bytes()
        training = self.dataset([1])
        self.assert_train_batch(training, 1)
        self.assertNotEqual(training.cache_dir, evaluation.cache_dir)
        self.assertEqual(pathlib.Path(evaluation.param_dir).read_bytes(), original)

    def test_changed_training_split_is_not_silently_overridden(self):
        first = self.dataset([1])
        second = self.dataset([2])
        self.assert_train_batch(second, 2)
        self.assertNotEqual(first.cache_dir, second.cache_dir)

    def test_compatible_training_cache_is_reused_without_raw_data(self):
        first = self.dataset([1])
        for path in (self.root / 'demo').glob('*.mat'):
            path.unlink()
        second = self.dataset([1])
        evaluation = self.dataset([])
        self.assertEqual(second.cache_dir, first.cache_dir)
        self.assertEqual(evaluation.cache_dir, first.cache_dir)
        self.assert_train_batch(second, 1)
        self.assertEqual(tuple(evaluation.load_test_input(1).shape), (1, 4, 2, 2))

    def test_split_specific_cache_is_reused_without_raw_data(self):
        self.dataset([])
        first = self.dataset([1])
        for path in (self.root / 'demo').glob('*.mat'):
            path.unlink()
        second = self.dataset([1])
        self.assertEqual(second.cache_dir, first.cache_dir)
        self.assert_train_batch(second, 1)
