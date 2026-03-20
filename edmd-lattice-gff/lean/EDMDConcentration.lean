import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Moments.SubGaussian

/-!
# EDMD Concentration Wrappers

Lean-side wrappers around the local Mathlib sub-Gaussian / Hoeffding surface
used by the lattice GFF EDMD diagnostics.

The point of this file is not to formalize the full lattice EDMD ratio-estimator
analysis. It exposes the reusable scalar concentration and rate-inversion facts
that the runtime currently relies on.
-/

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace HeytingLean.Physics.KoopmanGFF

section SubGaussian

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {X : ℕ → Ω → ℝ} {c : NNReal} {n : ℕ}

/-- A random variable whose law is the centered real Gaussian with variance
parameter `v` is sub-Gaussian with the same parameter. -/
theorem centeredGaussian_hasSubgaussianMGF_of_map_eq_gaussianReal
    {X : Ω → ℝ} {v : NNReal}
    (hX : μ.map X = gaussianReal (0 : ℝ) v) :
    HasSubgaussianMGF X v μ := by
  have h0 : HasSubgaussianMGF id v (gaussianReal (0 : ℝ) v) := by
    refine ⟨?_, ?_⟩
    · intro t
      simpa [id, mul_comm] using
        (integrable_exp_mul_gaussianReal (μ := (0 : ℝ)) (v := v) t)
    · intro t
      rw [mgf_id_gaussianReal (μ := (0 : ℝ)) (v := v)]
      simp
  have hXm : AEMeasurable X μ := aemeasurable_of_map_neZero (hX ▸ inferInstance)
  have h1 : HasSubgaussianMGF id v (μ.map X) := by
    simpa [hX] using h0
  simpa [Function.comp, id] using (HasSubgaussianMGF.of_map hXm h1)

/-- Scaling a centered Gaussian random variable scales the sub-Gaussian
parameter by the square of the deterministic multiplier. -/
theorem const_mul_centeredGaussian_hasSubgaussianMGF_of_map_eq_gaussianReal
    {X : Ω → ℝ} {v : NNReal} (w : ℝ)
    (hX : μ.map X = gaussianReal (0 : ℝ) v) :
    HasSubgaussianMGF (fun ω ↦ w * X ω) (⟨w ^ 2, sq_nonneg w⟩ * v) μ := by
  have hXm : AEMeasurable X μ := aemeasurable_of_map_neZero (hX ▸ inferInstance)
  have hmap' :
      μ.map ((fun x ↦ w * x) ∘ X) =
        gaussianReal (0 : ℝ) (⟨w ^ 2, sq_nonneg w⟩ * v) := by
    rw [← AEMeasurable.map_map_of_aemeasurable (measurable_id'.const_mul w).aemeasurable hXm, hX]
    simpa using (gaussianReal_map_const_mul (μ := (0 : ℝ)) (v := v) w)
  have hmap :
      μ.map (fun ω ↦ w * X ω) =
        gaussianReal (0 : ℝ) (⟨w ^ 2, sq_nonneg w⟩ * v) := by
    simpa [Function.comp] using hmap'
  simpa using centeredGaussian_hasSubgaussianMGF_of_map_eq_gaussianReal
    (X := fun ω ↦ w * X ω) hmap

/-- A deterministic weighted sum of independent centered Gaussian variables is
sub-Gaussian with the expected quadratic-weight parameter. -/
theorem weightedIndependentCenteredGaussianSum_hasSubgaussianMGF
    {ι : Type*} {X : ι → Ω → ℝ} {v : ι → NNReal} {w : ι → ℝ} {s : Finset ι}
    (h_indep : iIndepFun X μ)
    (h_gauss : ∀ i ∈ s, μ.map (X i) = gaussianReal (0 : ℝ) (v i)) :
    HasSubgaussianMGF (fun ω ↦ ∑ i ∈ s, w i * X i ω)
      (∑ i ∈ s, (⟨w i ^ 2, sq_nonneg (w i)⟩ * v i)) μ := by
  let Y : ι → Ω → ℝ := fun i ω ↦ w i * X i ω
  have h_indepY : iIndepFun Y μ := by
    simpa [Y, Function.comp] using
      h_indep.comp (fun i x ↦ w i * x) (fun i ↦ measurable_const.mul measurable_id)
  have h_subG :
      ∀ i ∈ s, HasSubgaussianMGF (Y i) (⟨w i ^ 2, sq_nonneg (w i)⟩ * v i) μ := by
    intro i hi
    simpa [Y] using
      const_mul_centeredGaussian_hasSubgaussianMGF_of_map_eq_gaussianReal
        (X := X i) (v := v i) (w := w i) (h_gauss i hi)
  simpa [Y] using
    (HasSubgaussianMGF.sum_of_iIndepFun h_indepY
      (c := fun i ↦ (⟨w i ^ 2, sq_nonneg (w i)⟩ * v i)) (s := s) h_subG)

/-- Upper-tail bound for a deterministic weighted sum of independent centered
Gaussian variables. -/
theorem weightedIndependentCenteredGaussianSum_tail_le
    {ι : Type*} {X : ι → Ω → ℝ} {v : ι → NNReal} {w : ι → ℝ} {s : Finset ι}
    (h_indep : iIndepFun X μ)
    (h_gauss : ∀ i ∈ s, μ.map (X i) = gaussianReal (0 : ℝ) (v i))
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ ∑ i ∈ s, w i * X i ω} ≤
      Real.exp (- ε ^ 2 / (2 * ∑ i ∈ s, (⟨w i ^ 2, sq_nonneg (w i)⟩ * v i))) := by
  exact (weightedIndependentCenteredGaussianSum_hasSubgaussianMGF
    (h_indep := h_indep) (h_gauss := h_gauss)).measure_ge_le hε

/-- Two-sided tail bound for a sub-Gaussian random variable. -/
theorem subgaussianAbsTail_le
    [IsFiniteMeasure μ] {X : Ω → ℝ}
    (hX : HasSubgaussianMGF X c μ) {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ |X ω|} ≤ 2 * Real.exp (- ε ^ 2 / (2 * c)) := by
  have hSubset :
      {ω | ε ≤ |X ω|} ⊆ {ω | ε ≤ X ω} ∪ {ω | ε ≤ -X ω} := by
    intro ω hω
    by_cases hnonneg : 0 ≤ X ω
    · left
      simpa [abs_of_nonneg hnonneg] using hω
    · right
      have hneg : X ω < 0 := lt_of_not_ge hnonneg
      simpa [abs_of_neg hneg, neg_nonneg.mpr (le_of_lt hneg)] using hω
  calc
    μ.real {ω | ε ≤ |X ω|}
      ≤ μ.real ({ω | ε ≤ X ω} ∪ {ω | ε ≤ -X ω}) :=
        MeasureTheory.measureReal_mono hSubset
    _ ≤ μ.real {ω | ε ≤ X ω} + μ.real {ω | ε ≤ -X ω} :=
        MeasureTheory.measureReal_union_le _ _
    _ ≤ Real.exp (- ε ^ 2 / (2 * c)) + Real.exp (- ε ^ 2 / (2 * c)) := by
        gcongr
        · exact hX.measure_ge_le hε
        · exact hX.neg.measure_ge_le hε
    _ = 2 * Real.exp (- ε ^ 2 / (2 * c)) := by ring

/-- Scalar upper-tail bound for sums of independent sub-Gaussian variables with
a common MGF constant. This is a stable wrapper over Mathlib's range-sum
Hoeffding theorem. -/
theorem subgaussianSumUpperTail_of_iIndep
    (h_indep : iIndepFun X μ)
    (h_subG : ∀ i < n, HasSubgaussianMGF (X i) c μ)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ ∑ i ∈ Finset.range n, X i ω} ≤
      Real.exp (- ε ^ 2 / (2 * n * c)) := by
  simpa using
    (ProbabilityTheory.HasSubgaussianMGF.measure_sum_range_ge_le_of_iIndepFun
      (X := X) (μ := μ) (c := c) (n := n) h_indep h_subG hε)

/-- Normalized-average upper-tail bound deduced from the range-sum theorem. -/
theorem subgaussianAverageUpperTail_of_iIndep
    (h_indep : iIndepFun X μ)
    (h_subG : ∀ i < n, HasSubgaussianMGF (X i) c μ)
    (hn : 0 < n)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ (∑ i ∈ Finset.range n, X i ω) / (n : ℝ)} ≤
      Real.exp (- (((n : ℝ) * ε) ^ 2 / (2 * n * c))) := by
  have hnR : 0 < (n : ℝ) := by exact_mod_cast hn
  have hsum :=
    subgaussianSumUpperTail_of_iIndep (X := X) (μ := μ) (c := c) (n := n)
      h_indep h_subG (ε := (n : ℝ) * ε) (by positivity)
  have hset :
      {ω | ε ≤ (∑ i ∈ Finset.range n, X i ω) / (n : ℝ)} =
        {ω | (n : ℝ) * ε ≤ ∑ i ∈ Finset.range n, X i ω} := by
    ext ω
    simp [le_div_iff₀ hnR, mul_comm]
  rw [hset]
  convert hsum using 1
  ring_nf

end SubGaussian

section Hoeffding

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {X : ℕ → Ω → ℝ} {a b : ℝ} {n : ℕ}

/-- Centered bounded independent variables inherit the scalar sub-Gaussian sum
bound through Mathlib's Hoeffding lemma. -/
theorem hoeffdingCenteredSumUpperTail_of_iIndep
    (h_indep : iIndepFun X μ)
    (hm : ∀ i < n, AEMeasurable (X i) μ)
    (hb : ∀ i < n, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc a b)
    (hc : ∀ i < n, μ[X i] = 0)
    {ε : ℝ} (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ ∑ i ∈ Finset.range n, X i ω} ≤
      Real.exp (- ε ^ 2 / (2 * n * (((‖b - a‖₊ / 2) ^ 2 : NNReal)))) := by
  have h_subG :
      ∀ i < n, HasSubgaussianMGF (X i) ((‖b - a‖₊ / 2) ^ 2) μ := by
    intro i hi
    exact ProbabilityTheory.hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
      (hm i hi) (hb i hi) (hc i hi)
  simpa using
    subgaussianSumUpperTail_of_iIndep (X := X) (μ := μ)
      (c := ((‖b - a‖₊ / 2) ^ 2 : NNReal)) (n := n) h_indep h_subG hε

end Hoeffding

section RateInversion

/-- If a multiplier upper bound is valid and the time step is positive, the
corresponding `-log` rate lower bound is valid. -/
theorem edmdRateLower_of_multiplierUpper
    {a upper dt : ℝ}
    (hdt : 0 < dt)
    (ha : 0 < a)
    (h_upper : a ≤ upper) :
    -Real.log upper / dt ≤ -Real.log a / dt := by
  have hupper_pos : 0 < upper := lt_of_lt_of_le ha h_upper
  have hlog : Real.log a ≤ Real.log upper := Real.log_le_log ha h_upper
  have hneg : -Real.log upper ≤ -Real.log a := by linarith
  have hmul := mul_le_mul_of_nonneg_right hneg (le_of_lt (one_div_pos.mpr hdt))
  simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hmul

/-- If a multiplier lower bound is valid and positive, the corresponding
`-log` rate upper bound is valid. -/
theorem edmdRateUpper_of_multiplierLower
    {lower a dt : ℝ}
    (hdt : 0 < dt)
    (hlower : 0 < lower)
    (h_lower : lower ≤ a) :
    -Real.log a / dt ≤ -Real.log lower / dt := by
  have ha : 0 < a := lt_of_lt_of_le hlower h_lower
  have hlog : Real.log lower ≤ Real.log a := Real.log_le_log hlower h_lower
  have hneg : -Real.log a ≤ -Real.log lower := by linarith
  have hmul := mul_le_mul_of_nonneg_right hneg (le_of_lt (one_div_pos.mpr hdt))
  simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hmul

/-- Combined EDMD rate interval induced by a positive multiplier interval. -/
theorem edmdRateInterval_of_multiplierInterval
    {lower a upper dt : ℝ}
    (hdt : 0 < dt)
    (hlower : 0 < lower)
    (h_lower : lower ≤ a)
    (h_upper : a ≤ upper) :
    -Real.log upper / dt ≤ -Real.log a / dt ∧
      -Real.log a / dt ≤ -Real.log lower / dt := by
  constructor
  · exact edmdRateLower_of_multiplierUpper hdt (lt_of_lt_of_le hlower h_lower) h_upper
  · exact edmdRateUpper_of_multiplierLower hdt hlower h_lower

end RateInversion

end HeytingLean.Physics.KoopmanGFF
