"""Contract transcribed from the author's main_NS_multistage.py and trainer."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import main


class TestAuthorTrainingConfig(unittest.TestCase):
    def test_training_schedule_matches_author_script(self):
        expected = dict(in_length=1, learning_rate=0.001, weight_decay=1e-4,
                        batch_size=8, print_frequency=25, rounds=10, epochs=10,
                        epochs_overall=100, recurrent=10, steps_per_epoch=500,
                        gradient_clip=5.0, scheduler_step_size=1, scheduler_gamma=0.7)
        for name, value in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(getattr(main, name), value)

    def test_task_and_architecture_match_author_script(self):
        expected = dict(Re=100, Ma=2, t_interval=3, N=12, K=2, M=6, num_blocks=4)
        for name, value in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(getattr(main, name), value)
        self.assertEqual(main.Params['norm_factors'], [0.5, 0.5, 5, 10])
        self.assertTrue(main.Params['if_ln'])

    def test_training_and_testing_orders_match_author_script(self):
        training, testing = main.get_orders()
        self.assertEqual(training, list(range(41, 211)))
        self.assertEqual(testing, list(range(1, 41)))
        self.assertFalse(set(training) & set(testing))
