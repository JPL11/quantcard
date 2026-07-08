"""Example quantbench entrypoint: a tiny MLP on synthetic Gaussian blobs.

Run:
    quantbench report examples/tiny_mlp.py
"""

import torch

_NUM_CLASSES = 4
_DIM = 64


def _make_data(n=512, seed=0):
    # Class centers are fixed across splits; only samples vary with `seed`.
    centers = torch.randn(_NUM_CLASSES, _DIM, generator=torch.Generator().manual_seed(42)) * 3
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, _NUM_CLASSES, (n,), generator=g)
    inputs = centers[labels] + torch.randn(n, _DIM, generator=g)
    return inputs, labels


def get_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(_DIM, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, _NUM_CLASSES),
    )
    # Quick training so accuracy is meaningful (a few seconds on CPU).
    inputs, labels = _make_data(n=2048, seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), labels)
        loss.backward()
        opt.step()
    return model


def get_eval_batches():
    inputs, labels = _make_data(n=512, seed=2)
    for i in range(0, len(inputs), 128):
        yield inputs[i : i + 128], labels[i : i + 128]
