"""Aggregation strategies.

We re-implement FedAvg and the three Reddi et al. (2021) **adaptive**
server-side optimisers — FedAdam, FedYogi, FedAdagrad — so they work
seamlessly with our PEFT (trainable-only) parameter contract.

A strategy is just a function ``(global, [client_deltas, n_examples]) → global``,
i.e. it consumes a *pseudo-gradient* ``Δ = global - mean(client_weights)`` and
applies its update rule. Flower's built-in strategies expect every Layer to
exist on every client, which is true for us because every client receives
exactly the same set of trainable params.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

ArrayList = List[np.ndarray]


# ---------------------------------------------------------------------------
def _weighted_mean(
    client_arrays: Sequence[Tuple[ArrayList, int]],
) -> ArrayList:
    """Sample-count weighted average of per-client arrays."""
    total = sum(n for _, n in client_arrays)
    if total == 0:
        # equal weights
        weights = [1 / len(client_arrays)] * len(client_arrays)
    else:
        weights = [n / total for _, n in client_arrays]

    summed: ArrayList = []
    for layer_idx in range(len(client_arrays[0][0])):
        agg = np.zeros_like(client_arrays[0][0][layer_idx], dtype=np.float64)
        for (arrays, _), w in zip(client_arrays, weights):
            agg += w * arrays[layer_idx].astype(np.float64)
        summed.append(agg.astype(client_arrays[0][0][layer_idx].dtype))
    return summed


# ---------------------------------------------------------------------------
@dataclass
class StrategyState:
    """Server-side state for adaptive optimisers."""
    m: Optional[ArrayList] = None  # 1st moment
    v: Optional[ArrayList] = None  # 2nd moment / accumulator
    step: int = 0


class Strategy:
    """Common interface for our aggregation strategies."""

    name: str = "base"

    def __init__(self, **kwargs):  # noqa: D401 - simple init
        self.state = StrategyState()

    def aggregate(
        self,
        global_arrays: ArrayList,
        client_arrays: Sequence[Tuple[ArrayList, int]],
    ) -> ArrayList:  # pragma: no cover - overridden
        raise NotImplementedError


class FedAvg(Strategy):
    name = "fedavg"

    def aggregate(self, global_arrays, client_arrays):
        return _weighted_mean(client_arrays)


class _AdaptiveBase(Strategy):
    """Reddi et al. adaptive-server-opt base class.

    Treats ``Δ = global - weighted_mean(client_weights)`` as a pseudo-gradient
    and runs Adam/Yogi/Adagrad on it.
    """

    name = "adaptive"

    def __init__(
        self,
        *,
        server_lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lr = server_lr
        self.b1 = beta1
        self.b2 = beta2
        self.tau = tau

    def _init_state(self, arrays: ArrayList) -> None:
        if self.state.m is None:
            self.state.m = [np.zeros_like(a, dtype=np.float64) for a in arrays]
            self.state.v = [np.zeros_like(a, dtype=np.float64) for a in arrays]
            self.state.step = 0

    def _update_moments(self, deltas: ArrayList) -> None:
        raise NotImplementedError

    def aggregate(self, global_arrays, client_arrays):
        self._init_state(global_arrays)
        avg = _weighted_mean(client_arrays)
        # Mean client delta — already in descent direction because each client
        # performed SGD/Adam on its local loss. Reddi et al.'s FedAdam
        # treats this as a pseudo-gradient (despite the descent sign):
        #   m_t = β1 m_{t-1} + (1-β1) Δ_t
        #   w_{t+1} = w_t + η · m_t / (√v_t + τ)
        deltas = [
            avg[i].astype(np.float64) - global_arrays[i].astype(np.float64)
            for i in range(len(global_arrays))
        ]
        self.state.step += 1
        self._update_moments(deltas)

        out: ArrayList = []
        for g_arr, m, v in zip(global_arrays, self.state.m, self.state.v):
            update = self.lr * m / (np.sqrt(v) + self.tau)
            new = g_arr.astype(np.float64) + update
            out.append(new.astype(g_arr.dtype))
        return out


class FedAdam(_AdaptiveBase):
    name = "fedadam"

    def _update_moments(self, grads):
        for i, g in enumerate(grads):
            self.state.m[i] = self.b1 * self.state.m[i] + (1 - self.b1) * g
            self.state.v[i] = self.b2 * self.state.v[i] + (1 - self.b2) * (g * g)


class FedYogi(_AdaptiveBase):
    name = "fedyogi"

    def _update_moments(self, grads):
        for i, g in enumerate(grads):
            self.state.m[i] = self.b1 * self.state.m[i] + (1 - self.b1) * g
            v = self.state.v[i]
            # Yogi: v_t = v_{t-1} - (1-β2) · g² · sign(v_{t-1} - g²)
            v_new = v - (1 - self.b2) * (g * g) * np.sign(v - (g * g))
            self.state.v[i] = v_new


class FedAdagrad(_AdaptiveBase):
    name = "fedadagrad"

    def _update_moments(self, grads):
        for i, g in enumerate(grads):
            self.state.m[i] = g
            self.state.v[i] = self.state.v[i] + g * g


# ---------------------------------------------------------------------------
def make_strategy(name: str, **kwargs) -> Strategy:
    name = name.lower()
    if name == "fedavg":
        return FedAvg()
    if name == "fedadam":
        return FedAdam(**kwargs)
    if name == "fedyogi":
        return FedYogi(**kwargs)
    if name == "fedadagrad":
        return FedAdagrad(**kwargs)
    raise ValueError(f"Unknown strategy: {name!r}")
