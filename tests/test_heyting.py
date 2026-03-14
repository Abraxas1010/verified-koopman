from __future__ import annotations

import torch

from verified_koopman.analysis.heyting_analysis import check_himp_adjoint
from verified_koopman.models.learnable_heyting import LearnableBounds, ParametricHeytingOps


def test_himp_adjoint_hard_random() -> None:
    torch.manual_seed(0)
    bounds = LearnableBounds(8, init_lo=0.0, init_hi=2.0, learnable=False)
    hey = ParametricHeytingOps(bounds, temperature=0.05)

    a = bounds.clamp(torch.randn(512, 8))
    b = bounds.clamp(torch.randn(512, 8))
    c = bounds.clamp(torch.randn(512, 8))
    ok = check_himp_adjoint(hey, a, b, c)
    assert ok.all().item()

