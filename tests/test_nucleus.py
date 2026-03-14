from __future__ import annotations

import torch

from verified_koopman.models.nucleus_bottleneck import ReLUNucleus, ThresholdNucleus


def _check_nucleus_axioms(R, x: torch.Tensor, y: torch.Tensor) -> None:
    Rx = R(x)

    # Extensive: x ≤ R(x)
    assert torch.all(x <= Rx)

    # Idempotent: R(R(x)) = R(x)
    assert torch.allclose(R(Rx), Rx)

    # Meet-preserving: R(x ⊓ y) = R(x) ⊓ R(y)
    lhs = R(torch.minimum(x, y))
    rhs = torch.minimum(R(x), R(y))
    assert torch.allclose(lhs, rhs)


def test_relu_nucleus_axioms() -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 8)
    y = torch.randn(1024, 8)
    _check_nucleus_axioms(ReLUNucleus(), x, y)


def test_threshold_nucleus_axioms() -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 8)
    y = torch.randn(1024, 8)
    _check_nucleus_axioms(ThresholdNucleus(0.2), x, y)

