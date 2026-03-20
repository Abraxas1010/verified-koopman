from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from eigenlearner.data_io import load_trajectories
from eigenlearner.lattice_gff import (
    LatticeGFFConfig,
    flattened_mass_operator_spectrum,
    retained_mode_indices,
)


def _load_config_section(path: str | None, section: str) -> dict[str, Any]:
    if path is None:
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"expected top-level mapping in {path}")
    data = raw.get(section, {})
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at section {section!r} in {path}")
    return dict(data)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _to_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"expected array with ndim 2 or 3, got {arr.ndim}")


def _pair_indices(num_steps: int, lag: int, stride: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for t in range(0, max(0, num_steps - lag), stride):
        tp = t + lag
        if tp < num_steps:
            pairs.append((t, tp))
    return pairs


def _complex_fft_trajectories(arr: np.ndarray, field_shape: tuple[int, ...]) -> np.ndarray:
    n_traj, time_steps, _ = arr.shape
    fields = arr.reshape((n_traj, time_steps, *field_shape))
    coeffs = np.fft.fftn(fields, axes=tuple(range(2, 2 + len(field_shape))), norm="ortho")
    return coeffs.reshape(n_traj, time_steps, -1)


def _complex_confidence_radius(noise_var: float, effective_denom: float, delta_mode: float) -> float:
    if effective_denom <= 0.0:
        return float("inf")
    return float(math.sqrt(2.0 * noise_var * math.log(8.0 / delta_mode) / effective_denom))


def _rate_from_coeff(coeff_abs: float, lag_dt: float, min_coeff: float) -> float:
    coeff_clamped = min(1.0 - 1e-9, max(float(min_coeff), float(coeff_abs)))
    return float(-math.log(coeff_clamped) / lag_dt)


def _mode_tuple(mode: int, field_shape: tuple[int, ...]) -> list[int]:
    return [int(x) for x in np.unravel_index(int(mode), field_shape)]


def _parse_mode_list(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        return [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    raise ValueError(f"unsupported mode-list value: {raw!r}")


def compute_edmd_certificate(
    trajectories: np.ndarray,
    metadata: dict[str, Any],
    *,
    retained_modes: int | None,
    delta: float,
    lag: int,
    stride: int,
    min_coeff: float,
    max_report_modes: int,
    ess_model: str,
    target_rate: float | None = None,
    exclude_modes: list[int] | None = None,
    min_slack: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    arr = _to_3d(np.asarray(trajectories, dtype=np.float64))
    cfg_data = metadata.get("config", {})
    cfg = LatticeGFFConfig(**cfg_data)
    if int(lag) <= 0:
        raise ValueError("lag must be positive")
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must lie in (0, 1)")

    field_shape = tuple(int(x) for x in metadata["field_shape"])
    if int(np.prod(field_shape)) != int(arr.shape[-1]):
        raise ValueError("metadata field_shape is inconsistent with trajectory feature dimension")

    pairs = _pair_indices(int(arr.shape[1]), int(lag), int(stride))
    if not pairs:
        raise ValueError("no trajectory pairs available for requested lag/stride")

    coeffs = _complex_fft_trajectories(arr, field_shape)
    z_t = np.stack([coeffs[:, t, :] for (t, _) in pairs], axis=1).reshape(-1, coeffs.shape[-1])
    z_tp = np.stack([coeffs[:, tp, :] for (_, tp) in pairs], axis=1).reshape(-1, coeffs.shape[-1])

    exact_rates = flattened_mass_operator_spectrum(cfg.lattice_size, cfg.dimension, cfg.spacing, cfg.mass)
    retained_all = retained_mode_indices(
        cfg.lattice_size,
        cfg.dimension,
        cfg.spacing,
        cfg.mass,
        retained_modes=retained_modes,
    )
    exclude_set = {int(mode) for mode in (exclude_modes or [])}
    retained = np.array([int(mode) for mode in retained_all if int(mode) not in exclude_set], dtype=np.int64)
    if retained.size == 0:
        raise ValueError("no retained modes remain after applying exclusions")

    lag_dt = float(cfg.dt) * int(lag)
    exact_coeffs = np.exp(-lag_dt * exact_rates)
    noise_vars = (1.0 - exact_coeffs**2) / exact_rates

    # Split the failure budget per retained complex mode. Conjugate-paired modes
    # are still reported separately, so this allocation is conservative.
    delta_mode = float(delta) / max(1, int(retained.size))
    theoretical_gap_lower = float(metadata["theoretical_gap_lower"])
    target_rate_value = theoretical_gap_lower if target_rate is None else float(target_rate)
    min_exact_rate = float(np.min(exact_rates[retained]))
    structural_gap_slack = float(min_exact_rate - target_rate_value)
    excited_window = bool(exclude_set or target_rate is not None)
    if excited_window and structural_gap_slack <= float(min_slack):
        raise ValueError(
            "requested excited-mode target has insufficient structural slack: "
            f"{structural_gap_slack} <= {float(min_slack)}"
        )

    mode_rows: list[dict[str, Any]] = []
    lower_rate_bounds: list[float] = []
    for mode in retained:
        reg = z_t[:, int(mode)]
        tgt = z_tp[:, int(mode)]
        denom = float(np.sum(np.abs(reg) ** 2).real)
        numer = np.sum(tgt * np.conjugate(reg))
        a_hat = complex(numer / denom) if denom > 0.0 else complex(0.0, 0.0)

        if ess_model == "ar1":
            ess_factor = float((1.0 - exact_coeffs[int(mode)]) / (1.0 + exact_coeffs[int(mode)]))
        else:
            ess_factor = 1.0
        effective_denom = max(float(min_coeff), denom * max(ess_factor, 1.0e-6))
        coeff_radius = _complex_confidence_radius(float(noise_vars[int(mode)]), effective_denom, delta_mode)

        coeff_abs = abs(a_hat)
        coeff_upper = min(1.0 - 1.0e-9, coeff_abs + coeff_radius)
        coeff_lower = max(float(min_coeff), coeff_abs - coeff_radius)
        rate_hat = _rate_from_coeff(coeff_abs, lag_dt, float(min_coeff))
        rate_lower = _rate_from_coeff(coeff_upper, lag_dt, float(min_coeff))
        rate_upper = _rate_from_coeff(coeff_lower, lag_dt, float(min_coeff))

        lower_rate_bounds.append(rate_lower)
        mode_rows.append(
            {
                "mode": int(mode),
                "multi_index": _mode_tuple(int(mode), field_shape),
                "exact_rate": float(exact_rates[int(mode)]),
                "exact_multiplier": float(exact_coeffs[int(mode)]),
                "estimated_multiplier_real": float(a_hat.real),
                "estimated_multiplier_imag": float(a_hat.imag),
                "estimated_multiplier_abs": float(coeff_abs),
                "multiplier_error_bound": float(coeff_radius),
                "estimated_rate": float(rate_hat),
                "rate_lower_confidence": float(rate_lower),
                "rate_upper_confidence": float(rate_upper),
                "denominator_energy": float(denom),
                "effective_denom": float(effective_denom),
                "noise_variance_exact": float(noise_vars[int(mode)]),
            }
        )

    certified_gap_lower = float(min(lower_rate_bounds))
    estimated_gap = float(min(row["estimated_rate"] for row in mode_rows))
    edmd_gap_error_bound = float(max(0.0, estimated_gap - certified_gap_lower))
    certified = bool(certified_gap_lower >= target_rate_value)
    failure_reason = None
    if not certified:
        failure_reason = "insufficient_confidence_radius"
        if structural_gap_slack <= 1.0e-12:
            failure_reason = "formal_gap_has_zero_structural_slack"

    max_report_modes = max(1, min(int(max_report_modes), len(mode_rows)))
    mode_payloads = [
        {
            "mode": int(row["mode"]),
            "multiplierErrorBound": float(row["multiplier_error_bound"]),
            "noiseVarianceExact": float(row["noise_variance_exact"]),
            "effectiveDenom": float(row["effective_denom"]),
            "deltaMode": float(delta_mode),
        }
        for row in mode_rows
    ]

    if excited_window:
        lean_payload = {
            "retainedModes": int(retained.size),
            "samplePairs": int(z_t.shape[0]),
            "estimatedGap": float(estimated_gap),
            "certifiedGapLower": float(certified_gap_lower),
            "edmdGapErrorBound": float(edmd_gap_error_bound),
            "theoreticalGapLower": float(theoretical_gap_lower),
            "targetRate": float(target_rate_value),
            "minExcitedRate": float(min_exact_rate),
            "delta": float(delta),
            "structuralGapSlack": float(structural_gap_slack),
        }
        formal_anchor_theorems = [
            "latticeMassEigenvalue_lower_bound",
            "finiteLatticeMassGapCertificate",
            "latticeCovarianceTailBound_invMass",
            "edmdEstimator_sub_exact_eq_weightedInnovation",
            "edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_splitIndependentCenteredGaussian",
            "latticeOUInnovationCoordinate_map_eq_gaussianReal",
            "latticeOUInnovationCoordinate_iIndepFun",
            "latticeOUFourierModeEstimator_error_tail_le_half_delta",
            "latticeEDMDModeDiagnosticOfReport",
            "LatticeEDMDModeDiagnostic.error_tail_le_half_delta_of_latticeOUFourierModel",
            "latticeExcitedEDMDDiagnosticOfReport",
            "LatticeExcitedEDMDDiagnostic.nontrivial_of_target",
            "subgaussianAverageUpperTail_of_iIndep",
            "edmdRateLower_of_multiplierUpper",
        ]
    else:
        lean_payload = {
            "retainedModes": int(retained.size),
            "samplePairs": int(z_t.shape[0]),
            "estimatedGap": float(estimated_gap),
            "certifiedGapLower": float(certified_gap_lower),
            "edmdGapErrorBound": float(edmd_gap_error_bound),
            "theoreticalGapLower": float(theoretical_gap_lower),
            "delta": float(delta),
            "structuralGapSlack": float(structural_gap_slack),
        }
        formal_anchor_theorems = [
            "latticeMassEigenvalue_lower_bound",
            "finiteLatticeMassGapCertificate",
            "latticeCovarianceTailBound_invMass",
            "edmdEstimator_sub_exact_eq_weightedInnovation",
            "edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_splitIndependentCenteredGaussian",
            "latticeOUInnovationCoordinate_map_eq_gaussianReal",
            "latticeOUInnovationCoordinate_iIndepFun",
            "latticeOUFourierModeEstimator_error_tail_le_half_delta",
            "latticeEDMDModeDiagnosticOfReport",
            "LatticeEDMDModeDiagnostic.error_tail_le_half_delta_of_latticeOUFourierModel",
            "latticeEDMDDiagnosticOfReport",
            "subgaussianAverageUpperTail_of_iIndep",
            "edmdRateLower_of_multiplierUpper",
        ]
    report = {
        "status": "pass" if certified else "fail",
        "target_rate": float(target_rate_value),
        "min_excited_rate": float(min_exact_rate),
        "excluded_modes": [int(mode) for mode in sorted(exclude_set)],
        "certified_gap_lower": certified_gap_lower,
        "estimated_gap": estimated_gap,
        "edmd_gap_error_bound": edmd_gap_error_bound,
        "theoretical_gap_lower": theoretical_gap_lower,
        "certified_gap_minus_target": float(certified_gap_lower - target_rate_value),
        "certified_gap_minus_theory": float(certified_gap_lower - theoretical_gap_lower),
        "structural_gap_slack": structural_gap_slack,
        "strict_gap_certificate_possible": bool(structural_gap_slack > 1.0e-12),
        "failure_reason": failure_reason,
        "retained_modes": int(retained.size),
        "dictionary_error_zero_for_retained_modes": True,
        "dictionary_note": "The retained Fourier modes are exact Koopman eigenfunctions for the lattice OU process, so the certification lane has no dictionary truncation error on those modes.",
        "reported_modes": mode_rows[:max_report_modes],
        "sample_pairs": int(z_t.shape[0]),
        "lag": int(lag),
        "stride": int(stride),
        "delta": float(delta),
        "per_mode_delta": float(delta_mode),
        "ess_model": str(ess_model),
        "confidence_note": "Each retained complex Fourier mode receives a Bonferroni-split failure budget delta / K. The reported multiplier radius uses the formally justified conservative split re/im union bound sqrt(2 * noise_var * log(8 / delta_mode) / effective_denom). Conjugate-paired modes are still reported separately, so the multiple-testing allocation remains conservative.",
        "field_shape": list(field_shape),
        "formal_anchor": {
            "lean_modules": [
                "HeytingLean.Physics.KoopmanGFF.LatticeMassGap",
                "HeytingLean.Physics.KoopmanGFF.LatticeApprox",
                "HeytingLean.Physics.KoopmanGFF.LatticeCertificate",
                "HeytingLean.Physics.KoopmanGFF.EDMDConcentration",
                "HeytingLean.Physics.KoopmanGFF.EDMDRatioSpecialization",
                "HeytingLean.Physics.KoopmanGFF.LatticeOUModel",
            ],
            "theorems": formal_anchor_theorems,
            "note": "The lattice spectrum, the exact EDMD ratio decomposition, the conservative log(8 / delta_mode) radius inversion, and the exact finite-product Gaussian Fourier innovation law for the stationary lattice OU model are formalized in Lean. The per-mode certificate hypotheses are discharged from that model itself; the remaining non-Lean surface is only the numerical execution of those already-formal formulas in Python.",
        },
    }
    if excited_window:
        report["certificate_kind"] = "excited_mode_window"
    else:
        report["certificate_kind"] = "floor_diagnostic"
    return lean_payload, mode_payloads, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--retained-modes", type=int, default=None)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-coeff", type=float, default=1.0e-6)
    parser.add_argument("--max-report-modes", type=int, default=12)
    parser.add_argument("--ess-model", choices=["ar1", "none"], default="ar1")
    parser.add_argument("--target-rate", type=float, default=None)
    parser.add_argument("--exclude-modes", default=None)
    parser.add_argument("--min-slack", type=float, default=0.0)
    parser.add_argument("--payload-name", default="lean_edmd_payload.json")
    parser.add_argument("--mode-payload-name", default="lean_edmd_mode_payloads.json")
    parser.add_argument("--report-name", default="edmd_certificate_report.json")
    return parser


def main() -> None:
    parser = build_parser()
    preview, _ = parser.parse_known_args()
    parser.set_defaults(**_load_config_section(preview.config, "certify_edmd"))
    args = parser.parse_args()

    metadata = _load_json(args.metadata)
    loaded = load_trajectories(args.data_root, file=args.file, key=args.key, mmap=False)
    lean_payload, mode_payloads, report = compute_edmd_certificate(
        loaded.array,
        metadata,
        retained_modes=args.retained_modes,
        delta=float(args.delta),
        lag=int(args.lag),
        stride=int(args.stride),
        min_coeff=float(args.min_coeff),
        max_report_modes=int(args.max_report_modes),
        ess_model=str(args.ess_model),
        target_rate=None if args.target_rate is None else float(args.target_rate),
        exclude_modes=_parse_mode_list(args.exclude_modes),
        min_slack=float(args.min_slack),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / str(args.payload_name), lean_payload)
    _write_json(out_dir / str(args.mode_payload_name), mode_payloads)
    _write_json(out_dir / str(args.report_name), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
