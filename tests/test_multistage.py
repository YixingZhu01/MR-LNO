import importlib
import inspect
import io
import copy
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from multistage import MultiStage  # noqa: E402
from Data.DatasetNS import ComNS_Dataset2D, NG  # noqa: E402
from lib.test import test_iterative_ComNS_InN as legacy_test  # noqa: E402
from lib.train import train_iterative_ComNS_InN as legacy_train  # noqa: E402
from lib.utils import spatial_gradient  # noqa: E402


class Scale(nn.Module):
    def __init__(self, scale=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.norm_factors = [1.0]
        self.if_ln = False

    def forward(self, value):
        return value * self.scale


class FourFieldScale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, 4, 1, 1), float(scale)))
        self.norm_factors = [1.0, 1.0, 1.0, 1.0]
        self.if_ln = False

    def forward(self, value):
        return value * self.scale


class LegacyFilterModel(Scale):
    def __init__(self, scale=1.0):
        super().__init__(scale)
        self.filter_d = torch.ones(1)
        self.filter_r = torch.ones(1)


class LegacyGeluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.activation = nn.GELU()
        self.norm_factors = [1.0]
        self.if_ln = False

    def forward(self, value):
        return self.activation(value) * self.scale


class TestDatasetBehavior(unittest.TestCase):
    def test_load_test_input_uses_four_channels_per_history_step(self):
        dataset = ComNS_Dataset2D.__new__(ComNS_Dataset2D)
        dataset.cache_names = {"test": ["sample"]}
        dataset.t_interval = 1
        frames = [
            {
                name: np.full((NG, NG), value, np.float32)
                for name, value in zip(("u", "v", "rho", "T"), values)
            }
            for values in ((1, 2, 3, 4), (5, 6, 7, 8))
        ]
        dataset.load_cache_file = lambda _: [frames]

        result = dataset.load_test_input(2)

        channels = [float(result[0, index, 0, 0]) for index in range(8)]
        self.assertEqual(channels, list(range(1, 9)))

    def test_dataset_module_main_has_no_stale_burgers_reference(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "Data.DatasetNS"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


class TestMetricBehavior(unittest.TestCase):
    def test_field_mse_uses_squared_errors(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "field_mse"))
        prediction = np.array([[[2.0]], [[4.0]], [[7.0]], [[10.0]]])
        truth = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])

        result = main.field_mse(prediction, truth)

        self.assertEqual(result, (5.0, 16.0, 36.0))

    def test_mean_square_error_aggregates_samples_and_writes_log(self):
        main = importlib.import_module("main")
        output = np.zeros((1, 2, 4, 1, 1), dtype=np.float64)
        output[0, 0, :, 0, 0] = (2.0, 4.0, np.log(7.0), np.log(10.0))
        output[0, 1, :, 0, 0] = (4.0, 6.0, np.log(5.0), np.log(8.0))
        truths = {
            1: (1.0, 2.0, 3.0, 4.0),
            2: (2.0, 3.0, 1.0, 2.0),
        }

        def loadmat(path):
            path_text = str(path)
            if "outputs" in path_text:
                return {"output": output.copy()}
            order = 1 if path_text.endswith("_1.mat") else 2
            values = truths[order]
            return {
                name: np.full((251, 1), value, dtype=np.float64)
                for name, value in zip(("u", "v", "rho", "T"), values)
            }

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                os.makedirs("MSE_t")
                stdout = io.StringIO()
                with (
                    mock.patch("main.Re", 20),
                    mock.patch("main.t_interval", 250),
                    mock.patch("main.scio.loadmat", side_effect=loadmat),
                    redirect_stdout(stdout),
                ):
                    main.mean_square_error("demo", "truth", [1, 2], ".")
                legacy_log_text = pathlib.Path("MSE_t/demo_MSE.log").read_text()
                true_mse_log_text = pathlib.Path(
                    "MSE_t/demo_true_MSE.log"
                ).read_text()
            finally:
                os.chdir(previous_directory)

        def parse_log(text):
            logged = {}
            for block in text.strip().split("\n\n"):
                title, value = block.splitlines()
                logged[title.removeprefix("Error of ").removesuffix(":")] = float(
                    value
                )
            return logged

        legacy = parse_log(legacy_log_text)
        true_mse = parse_log(true_mse_log_text)
        self.assertAlmostEqual(legacy["UV"], (np.sqrt(5.0) + np.sqrt(13.0)) / 2)
        self.assertAlmostEqual(legacy["rho"], 4.0)
        self.assertAlmostEqual(legacy["T"], 6.0)
        self.assertAlmostEqual(true_mse["UV"], 9.0)
        self.assertAlmostEqual(true_mse["rho"], 16.0)
        self.assertAlmostEqual(true_mse["T"], 36.0)
        self.assertIn("MSE(t=0.2, 0.5, 1, 2, 5)=", stdout.getvalue())
        self.assertIn("averaged MSE [uv,rho,T] =", stdout.getvalue())
        self.assertIn("True MSE(t=0.2, 0.5, 1, 2, 5)=", stdout.getvalue())


class TestReproducibility(unittest.TestCase):
    def test_reseeding_repeats_python_numpy_and_torch_sequences(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "configure_reproducibility"))

        main.configure_reproducibility(123, deterministic=False)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        main.configure_reproducibility(123, deterministic=False)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        self.assertEqual(first, second)

    def test_default_deterministic_mode_sets_backend_flags_and_model_seed(self):
        main = importlib.import_module("main")
        try:
            main.configure_reproducibility(321, deterministic=True)
            first = nn.Linear(3, 2).state_dict()
            main.configure_reproducibility(321, deterministic=True)
            second = nn.Linear(3, 2).state_dict()

            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            for name in first:
                self.assertTrue(torch.equal(first[name], second[name]))
        finally:
            main.configure_reproducibility(0, deterministic=False)

    def test_reproducibility_cli_defaults_are_deterministic(self):
        main = importlib.import_module("main")

        args = main.build_parser().parse_args(["-n", "demo"])

        self.assertTrue(hasattr(args, "seed"))
        self.assertEqual(args.seed, 0)
        self.assertTrue(args.deterministic)
        non_deterministic = main.build_parser().parse_args(
            ["-n", "demo", "--no-deterministic"]
        )
        self.assertFalse(non_deterministic.deterministic)

    def test_experiment_metadata_captures_training_and_data_configuration(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "experiment_metadata"))
        args = main.build_parser().parse_args(
            ["-n", "demo", "--stages", "2", "--seed", "9"]
        )

        metadata = main.experiment_metadata(args)

        self.assertEqual(metadata["experiment_name"], "demo")
        self.assertEqual(metadata["seed"], 9)
        self.assertEqual(metadata["training"]["recurrent"], main.recurrent)
        self.assertIn("optimizer", metadata["training"])
        self.assertEqual(metadata["training"]["optimizer"]["name"], "Adam")
        self.assertEqual(metadata["training"]["scheduler"]["name"], "StepLR")
        self.assertEqual(metadata["training"]["gradient_clip"], 5.0)
        self.assertEqual(metadata["network"]["num_blocks"], main.num_blocks)
        self.assertEqual(metadata["network"]["init_weight"], main.Params["init_weight"])
        self.assertEqual(metadata["task"]["orders_test"], main.get_orders()[1])
        self.assertEqual(len(metadata["source_sha256"]), 64)

    def test_version_two_experiment_contract_rejects_semantic_mismatches(self):
        main = importlib.import_module("main")
        args = main.build_parser().parse_args(
            ["-n", "demo", "--stages", "2", "--device", "cpu"]
        )
        base = main.experiment_metadata(args, resolved_device="cpu")
        mutations = (
            (("stages",), 3),
            (("task", "Re"), 500),
            (("task", "t_interval"), 50),
            (("training", "in_length"), 2),
            (("training", "recurrent"), 1),
        )

        for keys, value in mutations:
            with self.subTest(keys=keys):
                experiment = copy.deepcopy(base)
                target = experiment
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                checkpoint = {"format_version": 2, "experiment": experiment}

                with self.assertRaisesRegex(ValueError, "experiment contract"):
                    main.validate_experiment_contract(checkpoint, args)


class TestDeviceBehavior(unittest.TestCase):
    def test_auto_device_prefers_cuda(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "resolve_device"))
        with mock.patch("main.torch.cuda.is_available", return_value=True):
            self.assertEqual(main.resolve_device("auto"), torch.device("cuda"))

    def test_auto_device_falls_back_to_cpu(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "resolve_device"))
        with mock.patch("main.torch.cuda.is_available", return_value=False):
            self.assertEqual(main.resolve_device("auto"), torch.device("cpu"))

    def test_explicit_cuda_reports_when_unavailable(self):
        main = importlib.import_module("main")
        with mock.patch("main.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CUDA device"):
                main.resolve_device("cuda")

    def test_legendre_filters_are_registered_buffers(self):
        main = importlib.import_module("main")
        model = main.make_network().to("cpu").eval()

        buffers = dict(model.named_buffers())

        self.assertIn("filter_d", buffers)
        self.assertIn("filter_r", buffers)
        self.assertNotIn("filter_d", model.state_dict())
        self.assertNotIn("filter_r", model.state_dict())
        with torch.no_grad():
            output = model(torch.randn(1, 4, 128, 128))
        self.assertEqual(output.device.type, "cpu")

    def test_spatial_gradient_preserves_input_device(self):
        value = torch.zeros(1, 1, 4, 4)

        gradient = spatial_gradient(value)

        self.assertEqual(gradient.device, value.device)

    def test_legacy_training_uses_the_model_device(self):
        model = FourFieldScale().to("cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        value = torch.ones(1, 4, 1, 1)

        def generator():
            while True:
                yield value, [value * 2]

        with redirect_stdout(io.StringIO()):
            legacy_train(
                network=model,
                batch_size=1,
                epochs=1,
                max_ep=1,
                last_ep=0,
                optimizer=optimizer,
                dataset=None,
                train_gen=generator(),
                round=1,
                print_frequency=500,
                In_length=1,
            )

        self.assertEqual(model.scale.device.type, "cpu")

    def test_legacy_evaluation_uses_the_model_device(self):
        model = FourFieldScale().to("cpu")
        value = torch.ones(1, 4, 2, 2)

        class Dataset:
            @staticmethod
            def load_test_input(_):
                return value

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                os.makedirs("outputs")
                legacy_test(
                    model,
                    Dataset(),
                    iter([(value, [value])]),
                    round=1,
                    long_round=1,
                    out_name="cpu",
                    In_length=1,
                )
                self.assertTrue(pathlib.Path("outputs/cpu.mat").exists())
            finally:
                os.chdir(previous_directory)

    def test_evaluation_logs_density_and_temperature_for_every_history_step(self):
        class CaptureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.seen = None

            def forward(self, value):
                self.seen = value.detach().clone()
                return value[:, :4] * self.scale

        history = torch.tensor(
            [[[[0.0]], [[0.0]], [[np.e]], [[np.e**2]],
              [[0.0]], [[0.0]], [[np.e**3]], [[np.e**4]]]],
            dtype=torch.float32,
        )
        model = CaptureModel()

        class Dataset:
            @staticmethod
            def load_test_input(_):
                return history.clone()

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                os.makedirs("outputs")
                legacy_test(
                    model,
                    Dataset(),
                    iter([(history, [history[:, :4]])]),
                    round=1,
                    long_round=1,
                    out_name="history",
                    In_length=2,
                    if_ln=True,
                )
            finally:
                os.chdir(previous_directory)

        logged_fields = model.seen[0, [2, 3, 6, 7], 0, 0]
        self.assertTrue(
            torch.allclose(logged_fields, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )


class TestMultiStage(unittest.TestCase):
    def test_selected_stages_are_summed(self):
        model = MultiStage(lambda: Scale(1), stages=3)
        model.models[1].scale.data.fill_(2)
        model.models[2].scale.data.fill_(4)

        output = model(torch.ones(1, 1, 1, 1), stages=2)

        self.assertEqual(float(output.detach()), 3.0)

    def test_only_current_stage_is_trainable(self):
        model = MultiStage(lambda: Scale(1), stages=3)

        model.select_stage(2)

        flags = [next(stage.parameters()).requires_grad for stage in model.models]
        self.assertEqual(flags, [False, True, False])

    def test_stage_two_target_is_y_minus_stage_one(self):
        model = MultiStage(Scale, stages=2)
        model.models[0].scale.data.fill_(1)
        value = torch.ones(1, 1, 1, 1)

        current, residual, total = model.stage_values(value, value * 5, stage=2)

        self.assertEqual(float(current.detach()), 0.0)
        self.assertEqual(float(residual.detach()), 4.0)
        self.assertEqual(float(total.detach()), 1.0)

    def test_wrapper_preserves_lno_test_attributes(self):
        model = MultiStage(lambda: Scale(1), stages=2)

        self.assertEqual(model.norm_factors, [1.0])
        self.assertFalse(model.if_ln)

    def test_fit_trains_all_stages_sequentially(self):
        model = MultiStage(Scale, stages=2)
        value = torch.ones(1, 1, 1, 1)
        completed = []

        def generator():
            while True:
                yield value, [value * 2]

        losses = model.fit(
            generator_factory=generator,
            optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
            scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            ),
            rounds=1,
            epochs_per_round=1,
            steps_per_epoch=1,
            rollout_steps=1,
            channels_per_step=1,
            device="cpu",
            on_stage_end=lambda stage, model, history: completed.append(stage),
        )

        self.assertEqual(completed, [1, 2])
        self.assertEqual(sorted(losses), [1, 2])
        self.assertGreater(float(model.models[0].scale.detach()), 0.0)
        self.assertGreater(float(model.models[1].scale.detach()), 0.0)


class TestCheckpointCompatibility(unittest.TestCase):
    def test_single_stage_version_two_round_trip_is_weights_only(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "save_single_stage"))
        source = Scale(3)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "single.pt"
            main.save_single_stage(path, source, experiment={"seed": 7})
            loaded, checkpoint = main.load_single_stage(
                path,
                model_factory=Scale,
                map_location="cpu",
            )

        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["checkpoint_type"], "lno-single-stage")
        self.assertEqual(checkpoint["experiment"]["seed"], 7)
        self.assertIn("torch", checkpoint["runtime"])
        self.assertIn("resolved_device", checkpoint["runtime"])
        self.assertEqual(checkpoint["runtime"]["resolved_device"], "cpu")
        output = loaded(torch.ones(1, 1, 1, 1))
        self.assertEqual(float(output.detach()), 3.0)

    def test_legacy_single_stage_pickle_requires_explicit_opt_in(self):
        main = importlib.import_module("main")
        self.assertTrue(hasattr(main, "load_single_stage"))
        source = Scale(2)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy.pp"
            torch.save(source, path)
            with self.assertRaisesRegex(ValueError, "allow_legacy_pickle"):
                main.load_single_stage(path, Scale, map_location="cpu")
            with self.assertWarnsRegex(RuntimeWarning, "trusted legacy"):
                loaded, checkpoint = main.load_single_stage(
                    path,
                    Scale,
                    map_location="cpu",
                    allow_legacy_pickle=True,
                )

        self.assertEqual(checkpoint["format_version"], 0)
        output = loaded(torch.ones(1, 1, 1, 1))
        self.assertEqual(float(output.detach()), 2.0)

    def test_multistage_version_two_contains_experiment_metadata(self):
        main = importlib.import_module("main")
        self.assertIn("experiment", inspect.signature(main.save_multistage).parameters)
        source = MultiStage(Scale, stages=2)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "multi.pt"
            main.save_multistage(
                path,
                source,
                completed_stages=2,
                experiment={"seed": 11},
            )
            _, checkpoint = main.load_multistage(path, Scale, map_location="cpu")

        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["checkpoint_type"], "lno-multistage")
        self.assertEqual(checkpoint["experiment"]["seed"], 11)
        self.assertIn("numpy", checkpoint["runtime"])

    def test_version_one_multistage_checkpoint_remains_loadable(self):
        main = importlib.import_module("main")
        source = MultiStage(Scale, stages=2)
        source.models[0].scale.data.fill_(1)
        source.models[1].scale.data.fill_(2)
        checkpoint = {
            "format_version": 1,
            "stages": 2,
            "stage_state_dicts": [model.state_dict() for model in source.models],
            "network": {"N": 12, "K": 2, "M": 6, "num_blocks": 4},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "version1.pt"
            torch.save(checkpoint, path)
            loaded, loaded_checkpoint = main.load_multistage(
                path,
                Scale,
                map_location="cpu",
            )

        self.assertEqual(loaded_checkpoint["format_version"], 1)
        output = loaded(torch.ones(1, 1, 1, 1))
        self.assertEqual(float(output.detach()), 3.0)

    def test_checkpoint_rejects_mismatched_network_metadata(self):
        main = importlib.import_module("main")
        source = Scale(1)
        mismatches = {
            "M": 999,
            "if_ln": not main.Params["if_ln"],
            "norm_factors": [1.0, 1.0, 1.0, 1.0],
            "init_weight": [1.0] * len(main.Params["init_weight"]),
        }

        for key, value in mismatches.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "mismatch.pt"
                main.save_single_stage(path, source)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                checkpoint["network"][key] = value
                torch.save(checkpoint, path)

                with self.assertRaisesRegex(ValueError, "network configuration"):
                    main.load_single_stage(path, Scale, map_location="cpu")

    def test_version_two_rejects_missing_network_metadata_fields(self):
        main = importlib.import_module("main")
        source = Scale(1)

        for missing_key in main.network_metadata():
            with self.subTest(missing_key=missing_key), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "missing-network-field.pt"
                main.save_single_stage(path, source)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                del checkpoint["network"][missing_key]
                torch.save(checkpoint, path)

                with self.assertRaisesRegex(ValueError, "missing required fields"):
                    main.load_single_stage(path, Scale, map_location="cpu")

    def test_source_fingerprint_includes_active_legendre_filter(self):
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            tracked_paths = (
                "main.py",
                "multistage.py",
                "Data/DatasetNS.py",
                "lib/networkNS.py",
                "lib/train.py",
                "lib/test.py",
                "lib/utils.py",
                f"lib/legendres/LegendreConv{main.N}.mat",
            )
            for relative_path in tracked_paths:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative_path.encode("utf-8"))

            before = main.source_fingerprint(root)
            filter_path = root / f"lib/legendres/LegendreConv{main.N}.mat"
            filter_path.write_bytes(b"changed filter coefficients")
            after = main.source_fingerprint(root)

        self.assertNotEqual(before, after)

    def test_legacy_filter_tensors_are_upgraded_to_nonpersistent_buffers(self):
        main = importlib.import_module("main")
        source = LegacyFilterModel()

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy-filters.pp"
            torch.save(source, path)
            with self.assertWarns(RuntimeWarning):
                loaded, _ = main.load_single_stage(
                    path,
                    LegacyFilterModel,
                    map_location="cpu",
                    allow_legacy_pickle=True,
                )

        buffers = dict(loaded.named_buffers())
        self.assertIn("filter_d", buffers)
        self.assertIn("filter_r", buffers)
        self.assertNotIn("filter_d", loaded.state_dict())
        self.assertNotIn("filter_r", loaded.state_dict())

    def test_legacy_gelu_without_approximate_attribute_can_run_forward(self):
        main = importlib.import_module("main")
        source = LegacyGeluModel()
        del source.activation.approximate

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy-gelu.pp"
            torch.save(source, path)
            with self.assertWarns(RuntimeWarning):
                loaded, _ = main.load_single_stage(
                    path,
                    LegacyGeluModel,
                    map_location="cpu",
                    allow_legacy_pickle=True,
                )

        output = loaded(torch.ones(1, 1, 1, 1))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(loaded.activation.approximate, "none")

    def test_version_two_rejects_non_json_experiment_metadata(self):
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "unsafe.pt"
            with self.assertRaisesRegex(TypeError, "JSON-compatible"):
                main.save_single_stage(
                    path,
                    Scale(1),
                    experiment={"unsafe": object()},
                )


class TestLnoMainInterface(unittest.TestCase):
    def test_network_factory_resolves_filters_outside_project_working_directory(self):
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                with redirect_stdout(io.StringIO()):
                    model = main.make_network()
            finally:
                os.chdir(previous_directory)

        self.assertEqual(model.n, main.N)

    def test_model_is_wrapped_only_when_multistage_is_enabled(self):
        main = importlib.import_module("main")

        single = main.create_model(stages=1, model_factory=Scale)
        multiple = main.create_model(stages=3, model_factory=Scale)

        self.assertIsInstance(single, Scale)
        self.assertIsInstance(multiple, MultiStage)
        self.assertEqual(len(multiple.models), 3)

    def test_multistage_checkpoint_round_trip_uses_model_factory(self):
        main = importlib.import_module("main")
        source = MultiStage(Scale, stages=2)
        source.models[0].scale.data.fill_(1)
        source.models[1].scale.data.fill_(2)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.pt"
            main.save_multistage(path, source, completed_stages=2)
            loaded, checkpoint = main.load_multistage(
                path,
                model_factory=Scale,
                map_location="cpu",
            )

        self.assertEqual(checkpoint["stages"], 2)
        output = loaded(torch.ones(1, 1, 1, 1))
        self.assertEqual(float(output.detach()), 3.0)

    def test_multistage_load_rejects_requested_stage_count_mismatch(self):
        main = importlib.import_module("main")
        source = MultiStage(Scale, stages=2)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "partial.pt"
            main.save_multistage(path, source, completed_stages=1)
            with self.assertRaisesRegex(ValueError, "requested 2 stages"):
                main.load_multistage(
                    path,
                    model_factory=Scale,
                    map_location="cpu",
                    expected_stages=2,
                )

    def test_load_trained_model_validates_checkpoint_experiment_contract(self):
        main = importlib.import_module("main")
        args = main.build_parser().parse_args(
            ["-n", "demo", "--stages", "2", "--device", "cpu"]
        )
        experiment = main.experiment_metadata(args, resolved_device="cpu")
        experiment["task"]["Re"] = 500
        checkpoint = {"format_version": 2, "experiment": experiment}

        with mock.patch(
            "main.load_multistage",
            return_value=(MultiStage(Scale, stages=2), checkpoint),
        ):
            with self.assertRaisesRegex(ValueError, "experiment contract"):
                main.load_trained_model(args)

    def test_main_defaults_to_original_single_stage(self):
        main = importlib.import_module("main")

        args = main.build_parser().parse_args(["-n", "demo"])

        self.assertEqual(args.stages, 1)
        self.assertEqual(args.device, "auto")
        self.assertTrue(hasattr(args, "allow_legacy_pickle"))
        self.assertFalse(args.allow_legacy_pickle)

    def test_main_accepts_multi_stage_selection(self):
        main = importlib.import_module("main")

        args = main.build_parser().parse_args(["-n", "demo", "--stages", "3"])

        self.assertEqual(args.stages, 3)
        self.assertEqual(main.checkpoint_name("demo", 1), "demo_model.pp")
        self.assertEqual(main.checkpoint_name("demo", 3), "demo_multistage.pt")

        device_args = main.build_parser().parse_args(
            ["-n", "demo", "--device", "cpu"]
        )
        self.assertEqual(device_args.device, "cpu")

        legacy_args = main.build_parser().parse_args(
            ["-n", "demo", "--allow-legacy-pickle"]
        )
        self.assertTrue(legacy_args.allow_legacy_pickle)

    def test_eval_only_mode_skips_training_before_loading(self):
        main = importlib.import_module("main")
        defaults = main.build_parser().parse_args(["-n", "demo"])
        self.assertTrue(hasattr(defaults, "eval_only"))
        self.assertFalse(defaults.eval_only)

        network = object()
        with (
            mock.patch.object(sys, "argv", ["main.py", "-n", "demo", "--eval-only"]),
            mock.patch("main.configure_reproducibility"),
            mock.patch("main.os.makedirs"),
            mock.patch("main.train_and_save") as train_mock,
            mock.patch("main.load_trained_model", return_value=network) as load_mock,
            mock.patch("main.evaluate") as evaluate_mock,
        ):
            main.main()

        train_mock.assert_not_called()
        load_mock.assert_called_once()
        evaluate_mock.assert_called_once_with(mock.ANY, network)


if __name__ == "__main__":
    unittest.main()
