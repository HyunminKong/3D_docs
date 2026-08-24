"""What kind of motion the stream is currently in.

This is the retrieval key, and two constraints shaped it.

It must be computable *before* the chunk it describes is adapted to, so it reads
only the window that has already gone past. And it must not depend on absolute
pose, which drifts, so every feature is a per-frame difference: how far the
camera moved between frames, how much it turned, how variable both were, and how
much of the window was spent barely moving.

Vehicle telemetry was tested as an alternative and dropped. Across the 756
pairwise penalties, these pose statistics predicted compatibility at
rho = -0.336 against CAN bus's -0.316; controlling for pose left CAN bus with
nothing (-0.038, p = 0.29) while pose kept a contribution of its own
(-0.122, p = 0.0008). Position itself is deliberately absent -- how far apart two
segments were predicted compatibility at rho = +0.021, which is the negative
result the whole retrieval design rests on.
"""

import numpy as np
import torch

FEATURES = ("speed_mean", "speed_sd", "yaw_mean", "yaw_sd", "still_frac")


def _rotation_angles(rot):
    """Per-step geodesic angle between consecutive camera rotations, in radians."""
    rel = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])
    cos = (np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0
    return np.arccos(np.clip(cos, -1.0, 1.0))


def describe(c2w, fps=12.0, still_threshold=0.5):
    """Regime descriptor for one window of camera-to-world matrices.

    `c2w` is (n, 4, 4), ordered in time. Returns the raw feature vector; scaling
    is a separate step so that the same statistics can be pooled across scenes
    before being standardised.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    if len(c2w) < 3:
        return np.zeros(len(FEATURES))
    pos = c2w[:, :3, 3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps      # metres / second
    turn = _rotation_angles(c2w[:, :3, :3]) * fps                  # radians / second
    return np.array([
        step.mean(), step.std(),
        turn.mean(), turn.std(),
        float((step < still_threshold).mean()),
    ])


class RegimeEncoder(torch.nn.Module):
    """Descriptor -> key space.

    In the minimal configuration this is standardisation followed by identity, so
    the key space *is* the descriptor space and a retrieval can be read off by
    hand. The linear map is available for the trained configuration; keeping the
    identity default means a failure to retrieve cannot be blamed on an
    under-trained encoder before anything else has been ruled out.

    Running statistics are collected in the same pass that fills the bank, and
    frozen with it -- an encoder whose normalisation keeps shifting would make
    stored keys mean different things at different times.
    """

    def __init__(self, dim_key=None, momentum=0.02):
        super().__init__()
        n = len(FEATURES)
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self.momentum = momentum
        self.project = torch.nn.Linear(n, dim_key, bias=False) if dim_key else None

    @property
    def dim(self):
        return self.project.out_features if self.project is not None else len(FEATURES)

    def observe(self, x):
        """Update the running scale from a batch of raw descriptors."""
        if bool(self.frozen):
            return
        with torch.no_grad():
            self.mean.mul_(1 - self.momentum).add_(self.momentum * x.mean(0))
            self.var.mul_(1 - self.momentum).add_(self.momentum * x.var(0, unbiased=False))

    def freeze(self):
        self.frozen.fill_(True)

    def forward(self, x):
        x = (x - self.mean) / (self.var.sqrt() + 1e-6)
        return self.project(x) if self.project is not None else x
