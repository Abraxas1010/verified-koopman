from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class LyapunovCertificate:
    """
    Continuous-time Lyapunov certificate for x' = G x:

      V(x) = xᵀ P x, with P ≻ 0 and (Gᵀ P + P G) = -Q, Q ≻ 0.
    """

    G: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    stable: bool
    min_eig_P: float
    max_eig_sym_part_G: float

    def to_jsonable(self) -> Dict[str, float]:
        return {
            "stable": float(self.stable),
            "min_eig_P": float(self.min_eig_P),
            "max_eig_sym_part_G": float(self.max_eig_sym_part_G),
        }


def lyapunov_certificate_from_generator(
    G: np.ndarray,
    *,
    Q: Optional[np.ndarray] = None,
    eps_pd: float = 1e-9,
) -> LyapunovCertificate:
    """
    Compute a Lyapunov certificate for a *stable* generator matrix.

    Uses `scipy.linalg.solve_continuous_lyapunov` if SciPy is available.
    """

    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"expected square G, got shape {G.shape}")

    n = int(G.shape[0])
    if Q is None:
        Q = np.eye(n, dtype=np.float64)
    else:
        Q = np.asarray(Q, dtype=np.float64)
        if Q.shape != (n, n):
            raise ValueError(f"expected Q shape {(n, n)}, got {Q.shape}")

    sym = 0.5 * (G + G.T)
    max_eig_sym = float(np.linalg.eigvalsh(sym).max())

    try:
        import scipy.linalg  # type: ignore

        P = scipy.linalg.solve_continuous_lyapunov(G.T, -Q)
    except Exception as e:  # pragma: no cover
        raise RuntimeError("SciPy is required for Lyapunov certificate extraction") from e

    P = 0.5 * (P + P.T)
    eigP = np.linalg.eigvalsh(P)
    min_eig_P = float(eigP.min())
    stable = (max_eig_sym < 0.0) and (min_eig_P > float(eps_pd))

    return LyapunovCertificate(
        G=G,
        P=P,
        Q=Q,
        stable=bool(stable),
        min_eig_P=float(min_eig_P),
        max_eig_sym_part_G=float(max_eig_sym),
    )

