import HeytingLean.Physics.KoopmanGFF.LatticeMassGap

/-!
# Lattice Approximation Budget

Packaging layer for the explicit inverse-mass tail witness already available in
`LatticeMassGap`.

The point of this file is not to claim a full Koopman approximation theorem.
It gives Option D a stable Lean-side contract for the lattice covariance tail
budget that the numerical certification script can reference honestly.
-/

namespace HeytingLean.Physics.KoopmanGFF

noncomputable section

/-- A uniform bound on all lattice covariance singular values. -/
def LatticeCovarianceTailBound (d N : ℕ) [NeZero N] (a mass tail : ℝ) : Prop :=
  ∀ mode : ℕ, |latticeCovarianceSingularValue d N a mass mode| ≤ tail

/-- The inverse mass is a valid uniform covariance tail bound. -/
theorem latticeCovarianceTailBound_invMass (d N : ℕ) [NeZero N]
    (a mass : ℝ) (ha : 0 < a) (hmass : 0 < mass) :
    LatticeCovarianceTailBound d N a mass mass⁻¹ := by
  intro mode
  rw [abs_of_nonneg (latticeCovarianceSingularValue_nonneg d N a mass ha hmass mode)]
  exact latticeCovarianceSingularValue_le_invMass d N a mass hmass mode

/-- Packaged approximation budget for the lattice covariance surface. -/
structure LatticeApproximationBudget (d N : ℕ) [NeZero N] (a mass : ℝ) where
  tailBound : ℝ
  hTailBound : LatticeCovarianceTailBound d N a mass tailBound

/-- The explicit inverse-mass witness gives a canonical approximation budget. -/
def inverseMassApproximationBudget (d N : ℕ) [NeZero N]
    (a mass : ℝ) (ha : 0 < a) (hmass : 0 < mass) :
    LatticeApproximationBudget d N a mass := by
  refine ⟨mass⁻¹, ?_⟩
  exact latticeCovarianceTailBound_invMass d N a mass ha hmass

end

end HeytingLean.Physics.KoopmanGFF
