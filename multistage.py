"""Optional multi-stage residual training for the original LNO model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from torch import nn


class MultiStage(nn.Module):
    """Add identical LNO stages and train one residual stage at a time."""

    def __init__(self, model_factory: Callable[[], nn.Module], stages: int):
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be positive")
        self.models = nn.ModuleList(model_factory() for _ in range(stages))
        self.norm_factors = self.models[0].norm_factors
        self.if_ln = self.models[0].if_ln

    def _check_stage(self, stage: int) -> int:
        if not 1 <= stage <= len(self.models):
            raise ValueError(f"stage must be between 1 and {len(self.models)}")
        return stage

    def forward(self, value: torch.Tensor, stages: int | None = None) -> torch.Tensor:
        count = len(self.models) if stages is None else self._check_stage(stages)
        output = self.models[0](value)
        for model in self.models[1:count]:
            output = output + model(value)
        return output

    def select_stage(self, stage: int) -> None:
        self._check_stage(stage)
        for index, model in enumerate(self.models, start=1):
            selected = index == stage
            model.train(selected)
            for parameter in model.parameters():
                parameter.requires_grad = selected

    def stage_values(self, value: torch.Tensor, target: torch.Tensor, stage: int):
        self._check_stage(stage)
        current = self.models[stage - 1](value)
        if stage == 1:
            return current, target, current
        with torch.no_grad():
            previous = self(value, stages=stage - 1)
        return current, target - previous, previous + current

    def fit(
        self,
        generator_factory: Callable[[], Iterable],
        optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer],
        *,
        rounds: int,
        epochs_per_round: int,
        steps_per_epoch: int,
        rollout_steps: int,
        channels_per_step: int,
        device: str | torch.device,
        scheduler_factory: Callable[[torch.optim.Optimizer], object] | None = None,
        loss_weights: Sequence[float] | None = None,
        gradient_clip: float = 5.0,
        on_stage_end: Callable | None = None,
    ) -> dict[int, list[float]]:
        """Train Stage 1 first, then each remaining residual stage."""
        counts = (rounds, epochs_per_round, steps_per_epoch, rollout_steps, channels_per_step)
        if min(counts) < 1:
            raise ValueError("training counts must be positive")

        histories: dict[int, list[float]] = {}
        for stage in range(1, len(self.models) + 1):
            self.select_stage(stage)
            optimizer = optimizer_factory(self.models[stage - 1].parameters())
            scheduler = scheduler_factory(optimizer) if scheduler_factory else None
            optimizer.zero_grad()
            stage_history: list[float] = []

            for _ in range(rounds):
                generator = iter(generator_factory())
                for _ in range(epochs_per_round):
                    running_loss = 0.0
                    for _ in range(steps_per_epoch):
                        inputs, targets = next(generator)
                        if len(targets) < rollout_steps:
                            raise ValueError("generator returned too few rollout targets")
                        inputs = inputs.to(device)
                        loss = inputs.new_zeros(())

                        for target in targets[:rollout_steps]:
                            current, residual, total = self.stage_values(
                                inputs,
                                target.to(device),
                                stage,
                            )
                            weights = (
                                loss_weights
                                if loss_weights is not None
                                else self.norm_factors
                            )
                            loss = loss + _weighted_field_l2(current, residual, weights)
                            inputs = _roll_history(inputs, total, channels_per_step)

                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.models[stage - 1].parameters(),
                            gradient_clip,
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        running_loss += float(loss.detach())

                    stage_history.append(running_loss / steps_per_epoch)
                if scheduler is not None:
                    scheduler.step()

            histories[stage] = stage_history
            if on_stage_end is not None:
                on_stage_end(stage, self, stage_history)

        return histories


def _weighted_field_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: Sequence[float],
) -> torch.Tensor:
    if prediction.shape != target.shape or len(weights) != prediction.shape[1]:
        raise ValueError("prediction, target, and loss weights are incompatible")
    field_norms = torch.linalg.vector_norm(
        (prediction - target).flatten(start_dim=2),
        dim=2,
    )
    weight_tensor = prediction.new_tensor(weights).view(1, -1)
    return (field_norms * weight_tensor).sum(dim=1).mean()


def _roll_history(
    history: torch.Tensor,
    prediction: torch.Tensor,
    channels_per_step: int,
) -> torch.Tensor:
    if prediction.shape[1] != channels_per_step:
        raise ValueError("prediction channels do not match channels_per_step")
    if history.shape[1] == channels_per_step:
        return prediction
    if history.shape[1] < channels_per_step or history.shape[1] % channels_per_step:
        raise ValueError("history channels must be a multiple of channels_per_step")
    return torch.cat((history[:, channels_per_step:], prediction), dim=1)
