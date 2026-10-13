"""Checkpoint formats used by inference, EMA exports, and training."""

from collections.abc import Mapping
from pathlib import Path

import torch


def model_state_dict(checkpoint: Mapping) -> dict:
    """Unwrap a full checkpoint or a single-profile EMA checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Expected a checkpoint mapping.')
    state = checkpoint.get('weights', checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError('Expected a state dict or a checkpoint containing "weights".')
    if any(key.startswith('ema_model.') for key in state):
        state = {key.removeprefix('ema_model.'): value
                 for key, value in state.items() if key.startswith('ema_model.')}
    output = {key.removeprefix('module.'): value for key, value in state.items()
              if isinstance(value, torch.Tensor) and not key.startswith('_extra_state')}
    if not output:
        raise ValueError('Checkpoint contains no model tensors.')
    return output


def load_model_state(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found: {path}')
    return model_state_dict(torch.load(path, map_location='cpu', weights_only=True))


def mmaudio_initialization(state: Mapping) -> dict:
    """Copy MMAudio's shared fusion blocks into both binaural branches.

    Keep dataset-specific normalization and empty-text features initialized by
    the training pipeline, rather than replacing them with mono statistics.
    """
    output = {}
    for key, value in model_state_dict(state).items():
        if key in {'latent_mean', 'latent_std', 'empty_string_feat',
                   't_embed.freqs', 'latent_rot', 'clip_rot'}:
            continue
        if key.startswith('fused_blocks.'):
            suffix = key.removeprefix('fused_blocks.')
            output[f'fused_blocks_left.{suffix}'] = value
            output[f'fused_blocks_right.{suffix}'] = value
        else:
            output[key] = value
    return output
