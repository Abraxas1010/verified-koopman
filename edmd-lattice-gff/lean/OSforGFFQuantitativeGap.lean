import HeytingLean.External.OSforGFFBridge
import OSforGFF.Covariance.Momentum

/-!
# OSforGFF Quantitative Mass-Gap Bridge

This module exposes a stable, importable quantitative continuum mass-gap theorem
surface for downstream HeytingLean modules.

The existing `OSforGFFBridge` already exports the qualitative OS axioms. The
missing downstream surface was an explicit exponential decay theorem with
mass-dependent rate in the `HeytingLean.External.*` namespace. That theorem
already exists in the vendored `OSforGFF` covariance development; this file
re-exports it in a predictable shape for Koopman/GFF work.
-/

namespace HeytingLean.External.OSforGFFQuantitativeGap

noncomputable section

/-- The explicit continuum mass-gap decay constant used in the covariance bound. -/
def massGapDecayConstant (m : ℝ) : ℝ :=
  m ^ 2 * (Real.sinh 1 + 2) / (4 * Real.pi ^ 2)

/-- Quantitative continuum mass-gap witness: separated two-point correlations
decay exponentially at rate `m`. -/
def QuantitativeContinuumMassGap (m : ℝ) : Prop :=
  ∀ u v : SpaceTime, 1 ≤ m * ‖u - v‖ →
    |freeCovariance m u v| ≤ massGapDecayConstant m * Real.exp (-m * ‖u - v‖)

/-- Stable export of the continuum two-point exponential decay theorem. -/
theorem freeCovariance_exponential_decay
    (m : ℝ) [Fact (0 < m)] (u v : SpaceTime)
    (h_sep : 1 ≤ m * ‖u - v‖) :
    |freeCovariance m u v| ≤ massGapDecayConstant m * Real.exp (-m * ‖u - v‖) := by
  simpa [massGapDecayConstant] using freeCovariance_exponential_bound' m u v h_sep

/-- Translation-invariant kernel form of the same mass-gap bound. -/
theorem freeCovarianceKernel_exponential_decay
    (m : ℝ) [Fact (0 < m)] (z : SpaceTime)
    (hz : 1 ≤ m * ‖z‖) :
    |freeCovarianceKernel m z| ≤ massGapDecayConstant m * Real.exp (-m * ‖z‖) := by
  simpa [freeCovarianceKernel, freeCovariance, freeCovarianceBessel, massGapDecayConstant]
    using freeCovariance_exponential_bound' m 0 z (by simpa using hz)

/-- The vendored continuum package now exposes an importable quantitative
mass-gap witness under the Heyting bridge namespace. -/
theorem gaussianFreeField_quantitativeContinuumMassGap
    (m : ℝ) [Fact (0 < m)] :
    QuantitativeContinuumMassGap m := by
  intro u v h_sep
  exact freeCovariance_exponential_decay m u v h_sep

/-- Combined export: qualitative OS4 plus quantitative continuum mass-gap
decay under one stable bridge theorem. -/
theorem gaussianFreeField_OS4_with_quantitativeContinuumMassGap
    (m : ℝ) [Fact (0 < m)] :
    HeytingLean.External.OSforGFFBridge.OS4_Clustering
        (HeytingLean.External.OSforGFFBridge.GaussianFreeFieldMeasure m) ∧
      QuantitativeContinuumMassGap m := by
  exact
    ⟨HeytingLean.External.OSforGFFBridge.gaussianFreeField_satisfies_OS4 m,
      gaussianFreeField_quantitativeContinuumMassGap m⟩

end

end HeytingLean.External.OSforGFFQuantitativeGap
