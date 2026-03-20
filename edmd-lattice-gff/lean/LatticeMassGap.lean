import Lattice.Laplacian

/-!
# Lattice GFF Mass-Gap Surface

Stable Heyting-side re-exports of the finite-dimensional lattice Gaussian free
field mass-gap facts already proved in the vendored `GaussianField` package.

This module is intentionally packaging-oriented: it gives downstream Koopman/GFF
code a stable import surface for the finite-lattice covariance facts
(`λ_m ≥ m²`, `σ_m ≥ 0`, `σ_m ≤ 1 / m`) without claiming new mathematical
content beyond the upstream `GaussianField.Lattice` theorems.
-/

namespace HeytingLean.Physics.KoopmanGFF

open GaussianField

noncomputable section

/-- Stable alias for the lattice mass-operator eigenvalue `λ_m`. -/
abbrev latticeMassEigenvalue (d N : ℕ) [NeZero N] (a mass : ℝ) (m : ℕ) : ℝ :=
  GaussianField.latticeEigenvalue d N a mass m

/-- Stable alias for the lattice covariance singular value `σ_m = λ_m^{-1/2}`. -/
abbrev latticeCovarianceSingularValue (d N : ℕ) [NeZero N]
    (a mass : ℝ) (m : ℕ) : ℝ :=
  (GaussianField.latticeEigenvalue d N a mass m) ^ (-(1 : ℝ) / 2)

/-- The lattice mass-operator eigenvalue satisfies the explicit lower bound
`λ_m ≥ m²`. -/
theorem latticeMassEigenvalue_lower_bound (d N : ℕ) [NeZero N]
    (a mass : ℝ) (m : ℕ) :
    mass ^ 2 ≤ latticeMassEigenvalue d N a mass m := by
  unfold latticeMassEigenvalue
  rw [GaussianField.latticeEigenvalue_eq]
  linarith [GaussianField.latticeLaplacianEigenvalue_nonneg d N a m]

/-- Lattice mass-operator eigenvalues are strictly positive when `m > 0`. -/
theorem latticeMassEigenvalue_pos (d N : ℕ) [NeZero N]
    (a mass : ℝ) (ha : 0 < a) (hmass : 0 < mass) (m : ℕ) :
    0 < latticeMassEigenvalue d N a mass m := by
  simpa [latticeMassEigenvalue] using
    GaussianField.latticeEigenvalue_pos d N a mass ha hmass m

/-- Lattice covariance singular values are nonnegative. -/
theorem latticeCovarianceSingularValue_nonneg (d N : ℕ) [NeZero N]
    (a mass : ℝ) (ha : 0 < a) (hmass : 0 < mass) (m : ℕ) :
    0 ≤ latticeCovarianceSingularValue d N a mass m := by
  unfold latticeCovarianceSingularValue
  exact Real.rpow_nonneg (le_of_lt (GaussianField.latticeEigenvalue_pos d N a mass ha hmass m)) _

/-- Pointwise mass-gap bound on lattice covariance singular values:
`σ_m ≤ 1 / m`. -/
theorem latticeCovarianceSingularValue_le_invMass (d N : ℕ) [NeZero N]
    (a mass : ℝ) (hmass : 0 < mass) (m : ℕ) :
    latticeCovarianceSingularValue d N a mass m ≤ mass⁻¹ := by
  have hev_ge : mass ^ 2 ≤ GaussianField.latticeEigenvalue d N a mass m := by
    rw [GaussianField.latticeEigenvalue_eq]
    linarith [GaussianField.latticeLaplacianEigenvalue_nonneg d N a m]
  have h1 :
      (GaussianField.latticeEigenvalue d N a mass m) ^ (-(1 : ℝ) / 2) ≤
        (mass ^ 2) ^ (-(1 : ℝ) / 2) :=
    Real.rpow_le_rpow_of_nonpos (sq_pos_of_pos hmass) hev_ge (by norm_num)
  have h2 : (mass ^ 2) ^ (-(1 : ℝ) / 2) = mass⁻¹ := by
    rw [← Real.rpow_natCast mass 2, ← Real.rpow_mul (le_of_lt hmass)]
    norm_num
    exact Real.rpow_neg_one mass
  calc
    latticeCovarianceSingularValue d N a mass m
        = (GaussianField.latticeEigenvalue d N a mass m) ^ (-(1 : ℝ) / 2) := rfl
    _ ≤ (mass ^ 2) ^ (-(1 : ℝ) / 2) := h1
    _ = mass⁻¹ := h2

/-- The lattice covariance singular-value sequence is bounded by the explicit
mass witness `1 / m`. -/
theorem latticeCovarianceSingularValues_boundedByInvMass (d N : ℕ) [NeZero N]
    (a mass : ℝ) (ha : 0 < a) (hmass : 0 < mass) :
    ∃ C : ℝ, ∀ m : ℕ, |latticeCovarianceSingularValue d N a mass m| ≤ C := by
  refine ⟨mass⁻¹, ?_⟩
  intro m
  rw [abs_of_nonneg (latticeCovarianceSingularValue_nonneg d N a mass ha hmass m)]
  exact latticeCovarianceSingularValue_le_invMass d N a mass hmass m

/-- Finite-dimensional mass-gap certificate for the lattice GFF covariance
surface. -/
structure FiniteLatticeMassGapCertificate (d N : ℕ) [NeZero N]
    (a mass : ℝ) where
  eigenvalue_lower_bound : ∀ m : ℕ, mass ^ 2 ≤ latticeMassEigenvalue d N a mass m
  singular_value_nonneg : ∀ m : ℕ, 0 ≤ latticeCovarianceSingularValue d N a mass m
  singular_value_upper : ∀ m : ℕ, latticeCovarianceSingularValue d N a mass m ≤ mass⁻¹

/-- The vendored lattice spectral theorems assemble into an explicit
finite-dimensional mass-gap certificate. -/
theorem finiteLatticeMassGapCertificate (d N : ℕ) [NeZero N]
    (a mass : ℝ) (ha : 0 < a) (hmass : 0 < mass) :
    FiniteLatticeMassGapCertificate d N a mass where
  eigenvalue_lower_bound := latticeMassEigenvalue_lower_bound d N a mass
  singular_value_nonneg := latticeCovarianceSingularValue_nonneg d N a mass ha hmass
  singular_value_upper := latticeCovarianceSingularValue_le_invMass d N a mass hmass

end
