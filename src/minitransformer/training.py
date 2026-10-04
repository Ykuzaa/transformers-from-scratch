"""Small helpers shared by the training scripts (seeding, device, history plots)."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def plot_history(
    history: list[dict], path: str | Path, title: str, metric: str = "acc", metric_label: str = "Accuracy"
) -> None:
    """Train/validation loss and ``metric`` per epoch, side by side."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, label in [(axes[0], "loss", "Loss"), (axes[1], metric, metric_label)]:
        ax.plot(epochs, [h[f"train_{key}"] for h in history], "o-", label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], "o-", label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
