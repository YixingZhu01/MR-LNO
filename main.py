"""Original LNO training/testing flow with optional multi-stage residual training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import scipy
import scipy.io as scio
import torch
import torch.optim as optim

from Data.DatasetNS import ComNS_Dataset2D
from lib.networkNS import NetComNS_InN_legendre
from lib.test import test_iterative_ComNS_InN as test
from lib.train import train_iterative_ComNS_InN as train
from lib.utils import Initialization_factors
from multistage import MultiStage


CHECKPOINT_VERSION = 2
SINGLE_CHECKPOINT_TYPE = "lno-single-stage"
MULTISTAGE_CHECKPOINT_TYPE = "lno-multistage"


# Defaults aligned with the author's main_NS_multistage.py (supplied models).
in_length = 1
learning_rate = 0.001
weight_decay = 1e-4
batch_size = 8
print_frequency = 25
rounds = 10
epochs = 10
epochs_overall = rounds * epochs
recurrent = 10
steps_per_epoch = 500
gradient_clip = 5.0
scheduler_step_size = 1
scheduler_gamma = 0.7

# Original learning task.
Re = 100
Ma = 2
t_interval = 3

# Original LNO network configuration.
N = 12
K = 2
M = 6
num_blocks = 4
Params = {
    "n": N,
    "m": M,
    "k": K,
    "norm_factors": [0.5, 0.5, 5, 10],
    "init_weight": [
        math.sqrt(3),
        math.sqrt(Initialization_factors[f"({N},{K},{M})"]),
        math.sqrt(6),
        math.sqrt(3),
        math.sqrt(6),
        math.sqrt(3),
    ],
    "if_ln": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Original LNO with optional multi-stage residual training"
    )
    parser.add_argument("-n", "--out_name", required=True, help="experiment name")
    parser.add_argument(
        "--stages",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="1=original LNO; 2/3=enable multi-stage residual training",
    )
    parser.add_argument("--data-dir", default="D:/LNOdata/", help="LNO data root")
    parser.add_argument(
        "--device",
        default="auto",
        help="compute device: auto prefers CUDA and falls back to CPU",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use deterministic PyTorch algorithms (default: enabled)",
    )
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="allow loading a trusted legacy whole-model checkpoint",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training and evaluate an existing checkpoint",
    )
    return parser


def normalize_data_dir(data_dir: str) -> str:
    # DatasetNS.py concatenates data_dir and data_name directly.
    return data_dir.rstrip("/\\") + os.sep


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    requested = "auto" if device is None else str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return resolved


def configure_reproducibility(seed: int, deterministic: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def checkpoint_name(out_name: str, stages: int) -> str:
    return f"{out_name}_model.pp" if stages == 1 else f"{out_name}_multistage.pt"


def network_metadata() -> dict:
    return {
        "N": N,
        "K": K,
        "M": M,
        "num_blocks": num_blocks,
        "norm_factors": list(Params["norm_factors"]),
        "init_weight": list(Params["init_weight"]),
        "if_ln": bool(Params["if_ln"]),
    }


def source_fingerprint(project_root: str | Path | None = None) -> str:
    project_root = (
        Path(__file__).resolve().parent
        if project_root is None
        else Path(project_root)
    )
    relative_paths = (
        "main.py",
        "multistage.py",
        "Data/DatasetNS.py",
        "lib/networkNS.py",
        "lib/train.py",
        "lib/test.py",
        "lib/utils.py",
        f"lib/legendres/LegendreConv{N}.mat",
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        digest.update(relative_path.encode("utf-8"))
        digest.update((project_root / relative_path).read_bytes())
    return digest.hexdigest()


def runtime_metadata(device: str | torch.device | None = None) -> dict:
    resolved = torch.device("cpu" if device is None else device)
    metadata = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "resolved_device": str(resolved),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        metadata["gpu_name"] = torch.cuda.get_device_name(index)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(index))
    return metadata


def json_metadata(metadata: dict | None) -> dict:
    try:
        encoded = json.dumps(metadata or {}, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "checkpoint metadata must contain only JSON-compatible values"
        ) from error
    return json.loads(encoded)


def ensure_gelu_compatibility(model: torch.nn.Module) -> None:
    """Restore the exact-GELU default absent from older PyTorch pickles."""
    for module in model.modules():
        if isinstance(module, torch.nn.GELU) and not hasattr(module, "approximate"):
            module.approximate = "none"


def validate_network_metadata(checkpoint: dict) -> None:
    saved = checkpoint.get("network")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint is missing network configuration")
    expected = network_metadata()
    version = int(checkpoint.get("format_version", 1))
    required_keys = (
        set(expected)
        if version == CHECKPOINT_VERSION
        else {"N", "K", "M", "num_blocks"}
    )
    missing_keys = required_keys - saved.keys()
    if missing_keys:
        raise ValueError(
            "checkpoint network configuration is missing required fields: "
            f"{sorted(missing_keys)}"
        )
    mismatches = {
        key: (saved[key], expected[key])
        for key in required_keys
        if saved[key] != expected[key]
    }
    if mismatches:
        raise ValueError(
            f"checkpoint network configuration does not match current code: {mismatches}"
        )


def validate_experiment_contract(checkpoint: dict, args) -> None:
    """Reject version-2 weights trained for a different physical task."""
    if int(checkpoint.get("format_version", 1)) != CHECKPOINT_VERSION:
        return
    experiment = checkpoint.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("checkpoint is missing its experiment contract")

    orders_train, orders_test = get_orders()
    expected = {
        "stages": int(args.stages),
        "task": {
            "Re": Re,
            "Ma": Ma,
            "t_interval": t_interval,
            "orders_train": orders_train,
            "orders_test": orders_test,
        },
        "training": {
            "in_length": in_length,
            "recurrent": recurrent,
        },
    }
    mismatches = {}
    if experiment.get("stages") != expected["stages"]:
        mismatches["stages"] = (experiment.get("stages"), expected["stages"])
    saved_task = experiment.get("task")
    if saved_task != expected["task"]:
        mismatches["task"] = (saved_task, expected["task"])
    saved_training = experiment.get("training")
    for key, value in expected["training"].items():
        saved_value = (
            saved_training.get(key) if isinstance(saved_training, dict) else None
        )
        if saved_value != value:
            mismatches[f"training.{key}"] = (saved_value, value)
    if mismatches:
        raise ValueError(f"checkpoint experiment contract mismatch: {mismatches}")


def experiment_metadata(
    args,
    resolved_device: str | torch.device | None = None,
) -> dict:
    orders_train, orders_test = get_orders()
    resolved = resolve_device(args.device) if resolved_device is None else torch.device(
        resolved_device
    )
    return {
        "experiment_name": args.out_name,
        "stages": int(args.stages),
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "device_request": str(args.device),
        "resolved_device": str(resolved),
        "data_dir": normalize_data_dir(args.data_dir),
        "source_sha256": source_fingerprint(),
        "training": {
            "batch_size": batch_size,
            "rounds": rounds,
            "epochs_per_round": epochs,
            "steps_per_epoch": steps_per_epoch,
            "recurrent": recurrent,
            "in_length": in_length,
            "gradient_clip": gradient_clip,
            "optimizer": {
                "name": "Adam",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            },
            "scheduler": {
                "name": "StepLR",
                "step_size": scheduler_step_size,
                "gamma": scheduler_gamma,
            },
        },
        "task": {
            "Re": Re,
            "Ma": Ma,
            "t_interval": t_interval,
            "orders_train": orders_train,
            "orders_test": orders_test,
        },
        "network": network_metadata(),
    }


def make_network():
    return NetComNS_InN_legendre(num_blocks=num_blocks, Params=Params)


def create_model(stages: int, model_factory=make_network):
    if stages == 1:
        return model_factory()
    return MultiStage(model_factory, stages=stages)


def get_orders() -> tuple[list[int], list[int]]:
    if Re == 100 and Ma == 2:
        orders_all = list(range(1, 211))
        orders_test = list(range(1, 41))
    elif Re == 20:
        orders_all = list(range(1, 451))
        orders_test = list(range(1, 51))
    else:
        orders_all = list(range(1, 226))
        orders_test = list(range(1, 26))
    test_set = set(orders_test)
    return [order for order in orders_all if order not in test_set], orders_test


def make_dataset(data_dir: str, train_split: bool):
    orders_train, orders_test = get_orders()
    return ComNS_Dataset2D(
        data_dir=data_dir,
        data_name=f"ComNS128Re{Re}Ma{Ma}",
        orders_train=orders_train if train_split else [],
        orders_test=orders_test,
        t_interval=t_interval,
        chara_velocity="c",
        if_ln=Params["if_ln"],
    )


def train_original(network, dataset) -> None:
    """Unchanged single-stage path through the original lib/train.py."""
    optimizer = optim.Adam(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
        last_epoch=-1,
    )
    for round_index in range(rounds):
        generator = dataset.data_generator_series(
            out_length=recurrent,
            in_length=in_length,
            batch_size=batch_size,
        )
        train(
            network=network,
            batch_size=batch_size,
            epochs=epochs,
            max_ep=epochs_overall,
            last_ep=round_index * epochs,
            optimizer=optimizer,
            dataset=dataset,
            train_gen=generator,
            round=recurrent,
            print_frequency=print_frequency,
            In_length=in_length,
            gradient_clip=gradient_clip,
        )
        scheduler.step()


def save_single_stage(path: Path, model, experiment: dict | None = None) -> None:
    device = next(model.parameters()).device
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "checkpoint_type": SINGLE_CHECKPOINT_TYPE,
            "model_state_dict": model.state_dict(),
            "network": json_metadata(network_metadata()),
            "experiment": json_metadata(experiment),
            "runtime": json_metadata(runtime_metadata(device)),
        },
        path,
    )


def load_single_stage(
    path,
    model_factory,
    map_location=None,
    *,
    allow_legacy_pickle: bool = False,
):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as error:
        if not allow_legacy_pickle:
            raise ValueError(
                "legacy whole-model checkpoint requires allow_legacy_pickle=True"
            ) from error
        warnings.warn(
            "Loading a trusted legacy pickle with weights_only=False; "
            "never enable this for an untrusted file.",
            RuntimeWarning,
            stacklevel=2,
        )
        model = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("legacy checkpoint does not contain a PyTorch module")
        ensure_gelu_compatibility(model)
        for name in ("filter_d", "filter_r"):
            tensor = getattr(model, name, None)
            if torch.is_tensor(tensor) and name not in model._buffers:
                delattr(model, name)
                model.register_buffer(name, tensor, persistent=False)
        return model, {
            "format_version": 0,
            "checkpoint_type": "legacy-pickled-module",
        }

    if not isinstance(checkpoint, dict):
        raise ValueError("single-stage checkpoint must be a dictionary")
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported single-stage checkpoint version")
    if checkpoint.get("checkpoint_type") != SINGLE_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a single-stage LNO checkpoint")
    validate_network_metadata(checkpoint)
    model = model_factory()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def save_multistage(
    path: Path,
    model: MultiStage,
    completed_stages: int,
    experiment: dict | None = None,
) -> None:
    device = next(model.parameters()).device
    torch.save(
        {
            "format_version": CHECKPOINT_VERSION,
            "checkpoint_type": MULTISTAGE_CHECKPOINT_TYPE,
            "stages": completed_stages,
            "stage_state_dicts": [
                stage.state_dict() for stage in model.models[:completed_stages]
            ],
            "network": json_metadata(network_metadata()),
            "experiment": json_metadata(experiment),
            "runtime": json_metadata(runtime_metadata(device)),
        },
        path,
    )


def load_multistage(
    path,
    model_factory,
    map_location=None,
    *,
    expected_stages: int | None = None,
):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("multi-stage checkpoint must be a dictionary")
    version = int(checkpoint.get("format_version", 1))
    if version not in (1, CHECKPOINT_VERSION):
        raise ValueError("unsupported multi-stage checkpoint version")
    if version == CHECKPOINT_VERSION and checkpoint.get(
        "checkpoint_type"
    ) != MULTISTAGE_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is not a multi-stage LNO checkpoint")
    validate_network_metadata(checkpoint)
    stages = int(checkpoint["stages"])
    if expected_stages is not None and stages != expected_stages:
        raise ValueError(
            f"requested {expected_stages} stages, but checkpoint contains {stages}"
        )
    state_dicts = checkpoint["stage_state_dicts"]
    if stages != len(state_dicts):
        raise ValueError("checkpoint stage count does not match saved state dictionaries")
    network = MultiStage(model_factory, stages=stages)
    for stage, state_dict in zip(network.models, state_dicts):
        stage.load_state_dict(state_dict)
    return network, checkpoint


def train_and_save(args) -> Path:
    data_dir = normalize_data_dir(args.data_dir)
    device = resolve_device(args.device)
    experiment = experiment_metadata(args, resolved_device=device)
    dataset = make_dataset(data_dir, train_split=True)
    model_path = Path("models") / checkpoint_name(args.out_name, args.stages)

    print("orders_train =", get_orders()[0])
    print("orders_test =", get_orders()[1])
    print("multi-stage enabled =", args.stages > 1)
    print("stages =", args.stages)
    print("device =", device)

    network = create_model(args.stages).to(device)
    print(network)
    if args.stages == 1:
        train_original(network, dataset)
        save_single_stage(model_path, network, experiment=experiment)
        return model_path

    def on_stage_end(stage, trained_model, history):
        save_multistage(
            model_path,
            trained_model,
            stage,
            experiment=experiment,
        )
        print(f"Stage {stage}/{args.stages} saved to {model_path}")

    network.fit(
        generator_factory=lambda: dataset.data_generator_series(
            out_length=recurrent,
            in_length=in_length,
            batch_size=batch_size,
        ),
        optimizer_factory=lambda parameters: optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        ),
        scheduler_factory=lambda optimizer: optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma,
            last_epoch=-1,
        ),
        rounds=rounds,
        epochs_per_round=epochs,
        steps_per_epoch=steps_per_epoch,
        rollout_steps=recurrent,
        channels_per_step=4,
        device=device,
        gradient_clip=gradient_clip,
        on_stage_end=on_stage_end,
    )
    return model_path


def load_trained_model(args):
    model_path = Path("models") / checkpoint_name(args.out_name, args.stages)
    device = resolve_device(args.device)
    if args.stages == 1:
        network, checkpoint = load_single_stage(
            model_path,
            make_network,
            map_location=device,
            allow_legacy_pickle=args.allow_legacy_pickle,
        )
        validate_experiment_contract(checkpoint, args)
        return network.to(device)

    network, checkpoint = load_multistage(
        model_path,
        make_network,
        map_location=device,
        expected_stages=args.stages,
    )
    validate_experiment_contract(checkpoint, args)
    return network.to(device)


def evaluate(args, network) -> None:
    data_dir = normalize_data_dir(args.data_dir)
    _, orders_test = get_orders()
    ground_truth_name = f"ComNS128Re{Re}Ma{Ma}"
    dataset = make_dataset(data_dir, train_split=False)
    test_round = 10
    long_round = int(500 / t_interval)
    generator = dataset.data_generator_series(
        out_length=test_round,
        in_length=in_length,
        batch_size=batch_size,
        split="test",
    )
    network.eval()
    test(
        network,
        dataset,
        generator,
        test_round,
        long_round,
        args.out_name,
        In_length=in_length,
        if_ln=Params["if_ln"],
    )
    mean_square_error(
        args.out_name,
        ground_truth_name,
        orders_test,
        data_dir,
    )


def field_mse(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    if prediction.shape != truth.shape or prediction.shape[0] != 4:
        raise ValueError("prediction and truth must have matching four-field shapes")
    difference = prediction - truth
    return (
        float(np.mean(difference[0] ** 2 + difference[1] ** 2)),
        float(np.mean(difference[2] ** 2)),
        float(np.mean(difference[3] ** 2)),
    )


def legacy_field_error(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float, float]:
    """Return the historical original-LNO error metric for comparison."""
    if prediction.shape != truth.shape or prediction.shape[0] != 4:
        raise ValueError("prediction and truth must have matching four-field shapes")
    difference = prediction - truth
    return (
        float(np.mean(np.sqrt(difference[0] ** 2 + difference[1] ** 2))),
        float(np.mean(np.abs(difference[2]))),
        float(np.mean(np.abs(difference[3]))),
    )


def write_error_log(path: Path, errors: list[np.ndarray]) -> None:
    with open(path, "w") as log:
        for title, values in zip(("UV", "rho", "T"), errors):
            log.write(f"Error of {title}:\n")
            for value in values:
                log.write(f"{value}\n")
            log.write("\n")


def mean_square_error(out_name, ground_truth_name, orders_test, data_dir):
    output = scio.loadmat(Path("outputs") / f"{out_name}.mat")["output"]
    output[:, :, 2:4] = np.exp(output[:, :, 2:4])
    grid_size = output.shape[-1]

    if Re == 20:
        test_indices = [
            int(20 / t_interval) - 1,
            int(50 / t_interval) - 1,
            int(100 / t_interval) - 1,
            int(200 / t_interval) - 1,
        ]
        time_steps = int(250 / t_interval)
    else:
        test_indices = [
            int(20 / t_interval) - 1,
            int(50 / t_interval) - 1,
            int(100 / t_interval) - 1,
            int(200 / t_interval) - 1,
            int(500 / t_interval) - 1,
        ]
        time_steps = int(500 / t_interval)

    selected_legacy = np.zeros(len(test_indices))
    selected_true_mse = np.zeros(len(test_indices))
    legacy_errors = [np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)]
    true_mse_errors = [
        np.zeros(time_steps),
        np.zeros(time_steps),
        np.zeros(time_steps),
    ]
    for sample_index, order in enumerate(orders_test):
        ground_truth = scio.loadmat(
            Path(data_dir)
            / ground_truth_name
            / f"{ground_truth_name}_{order}.mat"
        )
        selected_index = 0
        for step in range(time_steps):
            truth_index = (step + 1) * t_interval
            truth_u = ground_truth["u"][truth_index].reshape(grid_size, grid_size)
            truth_v = ground_truth["v"][truth_index].reshape(grid_size, grid_size)
            truth_rho = ground_truth["rho"][truth_index].reshape(grid_size, grid_size)
            truth_t = ground_truth["T"][truth_index].reshape(grid_size, grid_size)
            prediction = output[step, sample_index]
            truth = np.stack((truth_u, truth_v, truth_rho, truth_t))
            legacy = legacy_field_error(prediction, truth)
            true_mse = field_mse(prediction, truth)
            for field_index in range(3):
                legacy_errors[field_index][step] += legacy[field_index]
                true_mse_errors[field_index][step] += true_mse[field_index]
            if step in test_indices:
                selected_legacy[selected_index] += legacy[0]
                selected_true_mse[selected_index] += true_mse[0]
                selected_index += 1

    selected_legacy /= len(orders_test)
    selected_true_mse /= len(orders_test)
    legacy_errors = [values / len(orders_test) for values in legacy_errors]
    true_mse_errors = [values / len(orders_test) for values in true_mse_errors]
    print("MSE(t=0.2, 0.5, 1, 2, 5)={}".format(selected_legacy))
    print(
        "averaged MSE [uv,rho,T] = {}".format(
            [np.mean(x) for x in legacy_errors]
        )
    )
    print("True MSE(t=0.2, 0.5, 1, 2, 5)={}".format(selected_true_mse))
    print(
        "averaged true MSE [uv,rho,T] = {}".format(
            [np.mean(x) for x in true_mse_errors]
        )
    )

    write_error_log(Path("MSE_t") / f"{out_name}_MSE.log", legacy_errors)
    write_error_log(
        Path("MSE_t") / f"{out_name}_true_MSE.log",
        true_mse_errors,
    )


def main() -> None:
    args = build_parser().parse_args()
    configure_reproducibility(args.seed, args.deterministic)
    for directory in ("models", "outputs", "MSE_t"):
        os.makedirs(directory, exist_ok=True)
    if not args.eval_only:
        train_and_save(args)
    evaluate(args, load_trained_model(args))


if __name__ == "__main__":
    main()
