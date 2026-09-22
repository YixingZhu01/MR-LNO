"""Safe inference for the author-provided Re100Ma2 three-stage model."""
import argparse
from pathlib import Path

import numpy as np
import torch

from lib.networkNS import NetComNS_InN_legendre
from main import resolve_device
from multistage import MultiStage

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / 'models' / 'Re100Ma2_t3_s3_inference.pt'


def load_pretrained(path=DEFAULT_CHECKPOINT, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'mrlno-inference-v1':
        raise ValueError('Expected an mrlno-inference-v1 checkpoint')
    configs = checkpoint['stages']
    if len(configs) != 3:
        raise ValueError('Expected three stages')
    iterator = iter(configs)

    def factory():
        config = next(iterator)
        if config['params']['if_ln'] is not True:
            raise ValueError('This inference interface requires log-density/temperature models')
        net = NetComNS_InN_legendre(config['num_blocks'], config['params'])
        net.load_state_dict(config['state_dict'], strict=True)
        for name in ('filter_d', 'filter_r'):
            saved = config[name]
            if saved.shape != getattr(net, name).shape:
                raise ValueError('Incompatible filter shape')
            getattr(net, name).copy_(saved)
        return net

    return MultiStage(factory, 3).to(resolve_device(device)).eval(), checkpoint


def encode_input(physical_input):
    value = np.array(physical_input, dtype=np.float32, copy=True)
    if value.ndim != 4 or value.shape[0] < 1 or value.shape[1:] != (4, 128, 128):
        raise ValueError('Input must have shape (batch, 4, 128, 128) in u,v,rho,T order')
    if not np.isfinite(value).all() or (value[:, 2:] <= 0).any():
        raise ValueError('Inputs must be finite; rho and T must be strictly positive')
    value[:, 2:] = np.log(value[:, 2:])
    return torch.from_numpy(value)


def rollout(model, physical_input, steps=1, stages=3):
    if steps < 1 or stages not in (1, 2, 3):
        raise ValueError('steps must be positive and stages must be 1, 2 or 3')
    device = next(model.parameters(), torch.empty(0)).device
    value = encode_input(physical_input).to(device)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            value = model(value, stages=stages)
            physical = value.clone()
            physical[:, 2:] = physical[:, 2:].exp()
            if not torch.isfinite(physical).all():
                raise ValueError('Non-finite prediction: rollout is unstable for this input')
            outputs.append(physical.cpu().numpy())
    return np.stack(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--smoke-test', action='store_true', help='synthetic input; not accuracy validation')
    source.add_argument('--input', type=Path, help='NPZ with physical input key, shape (B,4,128,128)')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stages', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--output', type=Path, help='new NPZ file; never overwrite an existing file')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists; choose a new path')
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.smoke_test:
        rng = np.random.default_rng(0)
        initial = rng.normal(0, 0.01, (1, 4, 128, 128)).astype(np.float32)
        initial[:, 2:] = np.exp(initial[:, 2:])
    else:
        with np.load(args.input, allow_pickle=False) as data:
            initial = data['input']
    model, checkpoint = load_pretrained(args.checkpoint, args.device)
    predictions = rollout(model, initial, args.steps, args.stages)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            np.savez_compressed(stream, prediction=predictions)
    print(f'PASS: stages=1..{args.stages}, device={next(model.parameters()).device}, '
          f'physical output shape={predictions.shape}, all values finite')
    print('Channels: u,v,rho,T. Input and output use the original nondimensionalization.')
    if args.smoke_test:
        print('Synthetic smoke test only; this does NOT validate physical accuracy.')


if __name__ == '__main__':
    main()
