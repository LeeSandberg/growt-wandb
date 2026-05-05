"""Feature extraction from PyTorch models for Growt auditing.

IMPORTANT — IP protection:
This module extracts raw feature vectors only. ALL structural analysis
happens server-side via the Growt API.
No engine code is included in this package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    layer_name: Optional[str] = None,
    max_samples: int = 5000,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors from a model layer.

    Args:
        model: PyTorch model.
        dataloader: DataLoader yielding (inputs, labels) tuples.
        layer_name: Dot-separated layer name. If None, auto-detects
            the penultimate layer.
        max_samples: Maximum samples to extract.
        device: Device to run on. Defaults to model's device.

    Returns:
        (features, labels) as numpy arrays.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    target_layer = _resolve_layer(model, layer_name)

    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    collected = 0

    hook_output: list[torch.Tensor] = []

    def hook_fn(
        _module: torch.nn.Module, _input: tuple, output: torch.Tensor,
    ) -> None:
        hook_output.clear()
        if isinstance(output, torch.Tensor):
            hook_output.append(output.detach())
        elif isinstance(output, (tuple, list)):
            hook_output.append(output[0].detach())

    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for batch in dataloader:
                if collected >= max_samples:
                    break

                inputs, batch_labels = batch[0], batch[1]
                inputs = inputs.to(device)
                model(inputs)

                if hook_output:
                    feat = hook_output[0]
                    # CLS token for transformers [B, seq, D]
                    if feat.dim() == 3:
                        feat = feat[:, 0, :]
                    # Flatten spatial dims [B, C, H, W]
                    elif feat.dim() > 2:
                        feat = feat.mean(dim=list(range(2, feat.dim())))
                    features_list.append(feat.cpu())
                    labels_list.append(batch_labels)
                    collected += feat.shape[0]
    finally:
        handle.remove()

    all_features = torch.cat(features_list, dim=0)[:max_samples]
    all_labels = torch.cat(labels_list, dim=0)[:max_samples]

    return all_features.numpy(), all_labels.numpy()


def _resolve_layer(
    model: torch.nn.Module, layer_name: Optional[str],
) -> torch.nn.Module:
    """Find the target layer for feature extraction."""
    if layer_name:
        parts = layer_name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        return module

    # Penultimate child
    children = list(model.children())
    if len(children) >= 2:
        return children[-2]

    # Last non-classifier module with parameters
    for _name, module in reversed(list(model.named_modules())):
        if not isinstance(module, (torch.nn.Linear, torch.nn.Softmax, torch.nn.LogSoftmax)):
            if list(module.parameters()):
                return module

    raise ValueError(
        "Could not auto-detect penultimate layer. "
        "Please specify layer_name explicitly."
    )
