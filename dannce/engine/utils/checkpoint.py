from __future__ import annotations

from typing import Any

import torch


def torch_load_checkpoint(path: str, *args: Any, **kwargs: Any):
    """Load a full checkpoint dict across PyTorch versions.

    PyTorch 2.6 changed the default of ``weights_only`` to ``True``. DANNCE
    checkpoints store optimizer state and params, so they need full unpickling.
    Older torch versions do not accept the ``weights_only`` kwarg, so retry
    without it for backward compatibility.
    """

    kwargs.setdefault("map_location", "cpu")

    try:
        return torch.load(path, *args, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, *args, **kwargs)
