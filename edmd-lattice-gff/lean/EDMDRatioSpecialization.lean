import HeytingLean.Physics.KoopmanGFF.EDMDConcentration

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace HeytingLean.Physics.KoopmanGFF

noncomputable section

/-- The exact finite-sample EDMD numerator for a single retained complex mode. -/
def edmdNumerator {n : ℕ} (z zNext : Fin n → ℂ) : ℂ :=
  ∑ i, zNext i * star (z i)

/-- The exact finite-sample EDMD energy denominator for a single retained complex mode. -/
def edmdEnergy {n : ℕ} (z : Fin n → ℂ) : ℂ :=
  ∑ i, z i * star (z i)

/-- The exact finite-sample EDMD multiplier estimator for a single retained complex mode. -/
def edmdEstimator {n : ℕ} (z zNext : Fin n → ℂ) : ℂ :=
  edmdNumerator z zNext / edmdEnergy z

/-- The weighted innovation contribution appearing in the exact EDMD quotient. -/
def weightedInnovation {n : ℕ} (innovation z : Fin n → ℂ) : ℂ :=
  (∑ i, innovation i * star (z i)) / edmdEnergy z

/-- The runtime square-root radius used by the exact-Fourier EDMD lane. -/
def edmdRuntimeRadius (noiseVar effectiveDenom delta : ℝ) : ℝ :=
  Real.sqrt (2 * noiseVar * Real.log (2 / delta) / effectiveDenom)

/-- Conservative complex-mode runtime radius obtained from a `re/im` union
bound. This is the fully formal fallback when only the split real Gaussian tail
is used. -/
def edmdConservativeRuntimeRadius (noiseVar effectiveDenom delta : ℝ) : ℝ :=
  Real.sqrt (2 * noiseVar * Real.log (8 / delta) / effectiveDenom)

/-- Split a real-valued innovation family indexed by `Fin n ⊕ Fin n` into a
complex innovation vector. The left copy supplies the real part, and the right
copy supplies the imaginary part. -/
def splitComplexInnovation {Ω : Type*} {n : ℕ}
    (ξ : Fin n ⊕ Fin n → Ω → ℝ) (ω : Ω) (i : Fin n) : ℂ :=
  (ξ (Sum.inl i) ω : ℂ) + (ξ (Sum.inr i) ω : ℂ) * Complex.I

/-- Deterministic weights for the real part of the weighted innovation. -/
def splitWeightsRe {n : ℕ} (z : Fin n → ℂ) (energy : ℝ) : Fin n ⊕ Fin n → ℝ
  | Sum.inl i => (z i).re / energy
  | Sum.inr i => (z i).im / energy

/-- Deterministic weights for the imaginary part of the weighted innovation. -/
def splitWeightsIm {n : ℕ} (z : Fin n → ℂ) (energy : ℝ) : Fin n ⊕ Fin n → ℝ
  | Sum.inl i => -(z i).im / energy
  | Sum.inr i => (z i).re / energy

/-- The exact EDMD energy denominator is the real sum of squared mode norms. -/
theorem edmdEnergy_eq_sum_normSq {n : ℕ} (z : Fin n → ℂ) :
    edmdEnergy z = ↑(∑ i, ‖z i‖ ^ 2) := by
  simp [edmdEnergy, Complex.mul_conj, Complex.sq_norm]

/-- The EDMD numerator splits into the exact drift contribution plus the weighted innovation term. -/
theorem edmdNumerator_eq_drift_plus_innovation
    {n : ℕ} {a : ℂ} {z zNext innovation : Fin n → ℂ}
    (hstep : ∀ i, zNext i = a * z i + innovation i) :
    edmdNumerator z zNext =
      a * edmdEnergy z + ∑ i, innovation i * star (z i) := by
  calc
    edmdNumerator z zNext
      = ∑ i, (a * z i + innovation i) * star (z i) := by
          simp [edmdNumerator, hstep]
    _ = ∑ i, (a * (z i * star (z i)) + innovation i * star (z i)) := by
          congr with i
          ring
    _ = ∑ i, a * (z i * star (z i)) + ∑ i, innovation i * star (z i) := by
          rw [Finset.sum_add_distrib]
    _ = a * edmdEnergy z + ∑ i, innovation i * star (z i) := by
          simp [edmdEnergy, Finset.mul_sum]

/-- The exact EDMD estimator error is exactly the weighted innovation quotient. -/
theorem edmdEstimator_sub_exact_eq_weightedInnovation
    {n : ℕ} {a : ℂ} {z zNext innovation : Fin n → ℂ}
    (hstep : ∀ i, zNext i = a * z i + innovation i)
    (henergy : edmdEnergy z ≠ 0) :
    edmdEstimator z zNext - a = weightedInnovation innovation z := by
  rw [edmdEstimator, weightedInnovation, edmdNumerator_eq_drift_plus_innovation hstep]
  field_simp [henergy]
  simp

/-- The estimator-error event is exactly the weighted-innovation event. -/
theorem edmdEstimator_error_event_eq
    {Ω : Type*} [MeasurableSpace Ω] {n : ℕ} {a : ℂ}
    {z zNext innovation : Ω → Fin n → ℂ}
    (hstep : ∀ ω i, zNext ω i = a * z ω i + innovation ω i)
    (henergy : ∀ ω, edmdEnergy (z ω) ≠ 0)
    {ε : ℝ} :
    {ω | ε ≤ ‖edmdEstimator (z ω) (zNext ω) - a‖} =
      {ω | ε ≤ ‖weightedInnovation (innovation ω) (z ω)‖} := by
  ext ω
  simp [edmdEstimator_sub_exact_eq_weightedInnovation (hstep ω) (henergy ω)]

/-- Any tail bound on the weighted innovation transports directly to the exact
EDMD estimator error. -/
theorem edmdEstimator_error_tail_le_of_weightedInnovationTail
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : ℕ} {a : ℂ}
    {z zNext innovation : Ω → Fin n → ℂ}
    (hstep : ∀ ω i, zNext ω i = a * z ω i + innovation ω i)
    (henergy : ∀ ω, edmdEnergy (z ω) ≠ 0)
    {ε B : ℝ}
    (hTail : μ.real {ω | ε ≤ ‖weightedInnovation (innovation ω) (z ω)‖} ≤ B) :
    μ.real {ω | ε ≤ ‖edmdEstimator (z ω) (zNext ω) - a‖} ≤ B := by
  rw [edmdEstimator_error_event_eq hstep henergy]
  exact hTail

/-- The real part of the split-complex weighted innovation is an explicit real
weighted sum over the split innovation coordinates. -/
theorem weightedInnovation_re_eq_splitSum
    {Ω : Type*} {n : ℕ} (ξ : Fin n ⊕ Fin n → Ω → ℝ) (ω : Ω) (z : Fin n → ℂ)
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2) :
    (weightedInnovation (splitComplexInnovation ξ ω) z).re =
      (∑ i, (ξ (Sum.inl i) ω * (z i).re + ξ (Sum.inr i) ω * (z i).im)) /
        (∑ i, ‖z i‖ ^ 2) := by
  rw [weightedInnovation, edmdEnergy_eq_sum_normSq, Complex.div_re]
  have hre : ((↑(∑ i, ‖z i‖ ^ 2) : ℂ)).re = ∑ i, ‖z i‖ ^ 2 := by
    change (∑ i, ‖z i‖ ^ 2 : ℝ) = ∑ i, ‖z i‖ ^ 2
    rfl
  have him : ((↑(∑ i, ‖z i‖ ^ 2) : ℂ)).im = 0 := by
    change (0 : ℝ) = 0
    rfl
  have hsq : Complex.normSq ((↑(∑ i, ‖z i‖ ^ 2) : ℂ)) = (∑ i, ‖z i‖ ^ 2) ^ 2 := by
    rw [Complex.normSq_apply, hre, him]
    ring
  rw [hre, him, hsq]
  simp [splitComplexInnovation]
  field_simp [henergy.ne']

/-- The imaginary part of the split-complex weighted innovation is an explicit
real weighted sum over the split innovation coordinates. -/
theorem weightedInnovation_im_eq_splitSum
    {Ω : Type*} {n : ℕ} (ξ : Fin n ⊕ Fin n → Ω → ℝ) (ω : Ω) (z : Fin n → ℂ)
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2) :
    (weightedInnovation (splitComplexInnovation ξ ω) z).im =
      (∑ i, (ξ (Sum.inr i) ω * (z i).re - ξ (Sum.inl i) ω * (z i).im)) /
        (∑ i, ‖z i‖ ^ 2) := by
  rw [weightedInnovation, edmdEnergy_eq_sum_normSq, Complex.div_im]
  have hre : ((↑(∑ i, ‖z i‖ ^ 2) : ℂ)).re = ∑ i, ‖z i‖ ^ 2 := by
    change (∑ i, ‖z i‖ ^ 2 : ℝ) = ∑ i, ‖z i‖ ^ 2
    rfl
  have him : ((↑(∑ i, ‖z i‖ ^ 2) : ℂ)).im = 0 := by
    change (0 : ℝ) = 0
    rfl
  have hsq : Complex.normSq ((↑(∑ i, ‖z i‖ ^ 2) : ℂ)) = (∑ i, ‖z i‖ ^ 2) ^ 2 := by
    rw [Complex.normSq_apply, hre, him]
    ring
  rw [hre, him, hsq]
  simp [splitComplexInnovation]
  field_simp [henergy.ne']
  calc
    ∑ x, (-(ξ (Sum.inl x) ω * (z x).im) + ξ (Sum.inr x) ω * (z x).re)
      = ∑ x, ξ (Sum.inr x) ω * (z x).re + ∑ x, -(ξ (Sum.inl x) ω * (z x).im) := by
          rw [Finset.sum_add_distrib, add_comm]
    _ = ∑ x, ξ (Sum.inr x) ω * (z x).re - ∑ x, ξ (Sum.inl x) ω * (z x).im := by
          simp [sub_eq_add_neg]

/-- Large complex norm forces one of the coordinate projections to be large at
scale `ε / √2`. -/
theorem complexNorm_event_subset_union
    {Ω : Type*} {W : Ω → ℂ} {ε : ℝ} (hε : 0 ≤ ε) :
    {ω | ε ≤ ‖W ω‖} ⊆
      {ω | ε / Real.sqrt 2 ≤ |(W ω).re|} ∪ {ω | ε / Real.sqrt 2 ≤ |(W ω).im|} := by
  intro ω hω
  have hω' : ε ≤ ‖W ω‖ := by
    simpa using hω
  by_contra hNot
  have hNotRe : ¬ ε / Real.sqrt 2 ≤ |(W ω).re| := by
    intro h
    exact hNot (Or.inl h)
  have hNotIm : ¬ ε / Real.sqrt 2 ≤ |(W ω).im| := by
    intro h
    exact hNot (Or.inr h)
  have hre : |(W ω).re| < ε / Real.sqrt 2 := lt_of_not_ge hNotRe
  have him : |(W ω).im| < ε / Real.sqrt 2 := lt_of_not_ge hNotIm
  have hthr_nonneg : 0 ≤ ε / Real.sqrt 2 := by
    positivity
  have hre_sq : (W ω).re ^ 2 < (ε / Real.sqrt 2) ^ 2 := by
    have hre' : |(W ω).re| < |ε / Real.sqrt 2| := by
      simpa [abs_of_nonneg hthr_nonneg] using hre
    exact (sq_lt_sq).2 hre'
  have him_sq : (W ω).im ^ 2 < (ε / Real.sqrt 2) ^ 2 := by
    have him' : |(W ω).im| < |ε / Real.sqrt 2| := by
      simpa [abs_of_nonneg hthr_nonneg] using him
    exact (sq_lt_sq).2 him'
  have hsqrt2 : (Real.sqrt 2) ^ 2 = 2 := by
    nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 2 by positivity)]
  have hnormsq : ‖W ω‖ ^ 2 = (W ω).re ^ 2 + (W ω).im ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  have hωsq : ε ^ 2 ≤ ‖W ω‖ ^ 2 := by
    nlinarith [hω', hε, norm_nonneg (W ω)]
  have hhalf : (ε / Real.sqrt 2) ^ 2 = ε ^ 2 / 2 := by
    have hsqrt2_ne : Real.sqrt 2 ≠ 0 := by positivity
    field_simp [hsqrt2_ne]
    nlinarith [hsqrt2]
  rw [hnormsq] at hωsq
  nlinarith [hre_sq, him_sq, hωsq, hhalf]

/-- The split real weights have total squared mass `energy⁻¹`. -/
theorem splitWeightsRe_sqSum {n : ℕ} (z : Fin n → ℂ) {energy : ℝ}
    (henergy : 0 < energy)
    (hsum : energy = ∑ x, ((z x).re ^ 2 + (z x).im ^ 2)) :
    ∑ j, (splitWeightsRe z energy j) ^ 2 = energy⁻¹ := by
  rw [Fintype.sum_sum_type]
  simp [splitWeightsRe, div_eq_mul_inv, pow_two]
  have hre :
      ∑ x, (z x).re * energy⁻¹ * ((z x).re * energy⁻¹) =
        (∑ x, (z x).re ^ 2) * (energy ^ 2)⁻¹ := by
    calc
      ∑ x, (z x).re * energy⁻¹ * ((z x).re * energy⁻¹)
          = ∑ x, ((z x).re ^ 2 * (energy ^ 2)⁻¹) := by
              congr with x
              field_simp [henergy.ne']
      _ = (∑ x, (z x).re ^ 2) * (energy ^ 2)⁻¹ := by
              simpa using
                (Finset.sum_mul (s := Finset.univ)
                  (f := fun x : Fin n => (z x).re ^ 2) (a := (energy ^ 2)⁻¹)).symm
  have him :
      ∑ x, (z x).im * energy⁻¹ * ((z x).im * energy⁻¹) =
        (∑ x, (z x).im ^ 2) * (energy ^ 2)⁻¹ := by
    calc
      ∑ x, (z x).im * energy⁻¹ * ((z x).im * energy⁻¹)
          = ∑ x, ((z x).im ^ 2 * (energy ^ 2)⁻¹) := by
              congr with x
              field_simp [henergy.ne']
      _ = (∑ x, (z x).im ^ 2) * (energy ^ 2)⁻¹ := by
              simpa using
                (Finset.sum_mul (s := Finset.univ)
                  (f := fun x : Fin n => (z x).im ^ 2) (a := (energy ^ 2)⁻¹)).symm
  rw [hre, him]
  have hcombine :
      (∑ x, (z x).re ^ 2) * (energy ^ 2)⁻¹ + (∑ x, (z x).im ^ 2) * (energy ^ 2)⁻¹ =
        ((∑ x, (z x).re ^ 2) + (∑ x, (z x).im ^ 2)) * (energy ^ 2)⁻¹ := by
    ring
  rw [hcombine]
  have hsum' : (∑ x, (z x).re ^ 2) + (∑ x, (z x).im ^ 2) = energy := by
    simpa [Finset.sum_add_distrib] using hsum.symm
  rw [hsum']
  field_simp [henergy.ne']

/-- The real part of the weighted innovation is a deterministic split weighted
sum over the real Gaussian coordinates. -/
theorem weightedInnovation_re_eq_splitWeightedSum
    {Ω : Type*} {n : ℕ} (ξ : Fin n ⊕ Fin n → Ω → ℝ) (ω : Ω) (z : Fin n → ℂ)
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2) :
    (weightedInnovation (splitComplexInnovation ξ ω) z).re =
      ∑ j, splitWeightsRe z (∑ i, ‖z i‖ ^ 2) j * ξ j ω := by
  rw [weightedInnovation_re_eq_splitSum ξ ω z henergy]
  simp [Fintype.sum_sum_type, Finset.sum_add_distrib, splitWeightsRe, div_eq_mul_inv,
    mul_comm, mul_left_comm]
  rw [left_distrib, Finset.mul_sum, Finset.mul_sum]
  congr 1
  · congr with x
    ring
  · congr with x
    ring

/-- The imaginary part of the weighted innovation is a deterministic split
weighted sum over the real Gaussian coordinates. -/
theorem weightedInnovation_im_eq_splitWeightedSum
    {Ω : Type*} {n : ℕ} (ξ : Fin n ⊕ Fin n → Ω → ℝ) (ω : Ω) (z : Fin n → ℂ)
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2) :
    (weightedInnovation (splitComplexInnovation ξ ω) z).im =
      ∑ j, splitWeightsIm z (∑ i, ‖z i‖ ^ 2) j * ξ j ω := by
  rw [weightedInnovation_im_eq_splitSum ξ ω z henergy]
  set A : ℝ := (∑ i, ‖z i‖ ^ 2)⁻¹
  simp [Fintype.sum_sum_type, splitWeightsIm, sub_eq_add_neg, div_eq_mul_inv,
    mul_comm, mul_left_comm]
  calc
    (∑ i, ‖z i‖ ^ 2)⁻¹ * ∑ x, ((z x).re * ξ (Sum.inr x) ω + -((z x).im * ξ (Sum.inl x) ω))
        = ∑ i, (∑ i, ‖z i‖ ^ 2)⁻¹ * ((z i).re * ξ (Sum.inr i) ω + -((z i).im * ξ (Sum.inl i) ω)) := by
            rw [Finset.mul_sum]
    ∑ i, (∑ i, ‖z i‖ ^ 2)⁻¹ * ((z i).re * ξ (Sum.inr i) ω + -((z i).im * ξ (Sum.inl i) ω))
        =
          ∑ i, ((∑ i, ‖z i‖ ^ 2)⁻¹ * ((z i).re * ξ (Sum.inr i) ω) +
            -((∑ i, ‖z i‖ ^ 2)⁻¹ * ((z i).im * ξ (Sum.inl i) ω))) := by
            apply Finset.sum_congr rfl
            intro i hi
            ring
    _ =
          ∑ i, (∑ i, ‖z i‖ ^ 2)⁻¹ * ((z i).re * ξ (Sum.inr i) ω) +
          ∑ i, -((∑ i, ‖z i‖ ^ 2)⁻¹ * ((z i).im * ξ (Sum.inl i) ω)) := by
            rw [Finset.sum_add_distrib]
    _ =
          -∑ x, (z x).im * (ξ (Sum.inl x) ω * (∑ i, ‖z i‖ ^ 2)⁻¹) +
          ∑ x, (z x).re * (ξ (Sum.inr x) ω * (∑ i, ‖z i‖ ^ 2)⁻¹) := by
            simp [Finset.sum_neg_distrib, mul_comm, mul_assoc, add_comm]

/-- The split imaginary weights have total squared mass `energy⁻¹`. -/
theorem splitWeightsIm_sqSum {n : ℕ} (z : Fin n → ℂ) {energy : ℝ}
    (henergy : 0 < energy)
    (hsum : energy = ∑ x, ((z x).re ^ 2 + (z x).im ^ 2)) :
    ∑ j, (splitWeightsIm z energy j) ^ 2 = energy⁻¹ := by
  rw [Fintype.sum_sum_type]
  simp [splitWeightsIm, div_eq_mul_inv, pow_two]
  have hre :
      ∑ x, (z x).im * energy⁻¹ * ((z x).im * energy⁻¹) =
        (∑ x, (z x).im ^ 2) * (energy ^ 2)⁻¹ := by
    calc
      ∑ x, (z x).im * energy⁻¹ * ((z x).im * energy⁻¹)
          = ∑ x, ((z x).im ^ 2 * (energy ^ 2)⁻¹) := by
              congr with x
              field_simp [henergy.ne']
      _ = (∑ x, (z x).im ^ 2) * (energy ^ 2)⁻¹ := by
              simpa using
                (Finset.sum_mul (s := Finset.univ)
                  (f := fun x : Fin n => (z x).im ^ 2) (a := (energy ^ 2)⁻¹)).symm
  have him :
      ∑ x, (z x).re * energy⁻¹ * ((z x).re * energy⁻¹) =
        (∑ x, (z x).re ^ 2) * (energy ^ 2)⁻¹ := by
    calc
      ∑ x, (z x).re * energy⁻¹ * ((z x).re * energy⁻¹)
          = ∑ x, ((z x).re ^ 2 * (energy ^ 2)⁻¹) := by
              congr with x
              field_simp [henergy.ne']
      _ = (∑ x, (z x).re ^ 2) * (energy ^ 2)⁻¹ := by
              simpa using
                (Finset.sum_mul (s := Finset.univ)
                  (f := fun x : Fin n => (z x).re ^ 2) (a := (energy ^ 2)⁻¹)).symm
  rw [hre, him]
  have hcombine :
      (∑ x, (z x).im ^ 2) * (energy ^ 2)⁻¹ + (∑ x, (z x).re ^ 2) * (energy ^ 2)⁻¹ =
        ((∑ x, (z x).im ^ 2) + (∑ x, (z x).re ^ 2)) * (energy ^ 2)⁻¹ := by
    ring
  rw [hcombine, add_comm]
  have hsum' : (∑ x, (z x).re ^ 2) + (∑ x, (z x).im ^ 2) = energy := by
    simpa [Finset.sum_add_distrib] using hsum.symm
  rw [hsum']
  field_simp [henergy.ne']

/-- The runtime square-root radius squares to the expected logarithmic expression. -/
theorem edmdRuntimeRadius_sq
    {noiseVar effectiveDenom delta : ℝ}
    (hNoise : 0 < noiseVar) (hEff : 0 < effectiveDenom) (hδ : 0 < delta ∧ delta < 2) :
    edmdRuntimeRadius noiseVar effectiveDenom delta ^ 2 =
      2 * noiseVar * Real.log (2 / delta) / effectiveDenom := by
  have hδ_le : delta ≤ 2 := le_of_lt hδ.2
  have htwo_div_ge_one : 1 ≤ 2 / delta := by
    rw [le_div_iff₀ hδ.1]
    linarith
  have hlog_nonneg : 0 ≤ Real.log (2 / delta) := Real.log_nonneg htwo_div_ge_one
  unfold edmdRuntimeRadius
  rw [Real.sq_sqrt]
  positivity

/-- The runtime radius turns the exponential tail expression into `δ / 2`. -/
theorem exp_neg_runtimeRadius_sq_mul_div
    {noiseVar effectiveDenom delta : ℝ}
    (hNoise : 0 < noiseVar) (hEff : 0 < effectiveDenom) (hδ : 0 < delta ∧ delta < 2) :
    Real.exp (-(edmdRuntimeRadius noiseVar effectiveDenom delta ^ 2 * effectiveDenom) /
        (2 * noiseVar)) = delta / 2 := by
  rw [edmdRuntimeRadius_sq hNoise hEff hδ]
  have htwo_div_pos : 0 < 2 / delta := by
    exact div_pos (by positivity) hδ.1
  have hNoise0 : noiseVar ≠ 0 := hNoise.ne'
  have hEff0 : effectiveDenom ≠ 0 := hEff.ne'
  have hRewrite :
      -((2 * noiseVar * Real.log (2 / delta) / effectiveDenom) * effectiveDenom) / (2 * noiseVar) =
        -Real.log (2 / delta) := by
    field_simp [hNoise0, hEff0]
  rw [hRewrite]
  rw [Real.exp_neg, Real.exp_log htwo_div_pos]
  field_simp [hδ.1.ne']

/-- The conservative runtime radius squares to the `log (8 / δ)` expression
coming from the `re/im` union bound. -/
theorem edmdConservativeRuntimeRadius_sq
    {noiseVar effectiveDenom delta : ℝ}
    (hNoise : 0 < noiseVar) (hEff : 0 < effectiveDenom) (hδ : 0 < delta ∧ delta < 8) :
    edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ^ 2 =
      2 * noiseVar * Real.log (8 / delta) / effectiveDenom := by
  have hδ_le : delta ≤ 8 := le_of_lt hδ.2
  have height_div_ge_one : 1 ≤ 8 / delta := by
    rw [le_div_iff₀ hδ.1]
    linarith
  have hlog_nonneg : 0 ≤ Real.log (8 / delta) := Real.log_nonneg height_div_ge_one
  unfold edmdConservativeRuntimeRadius
  rw [Real.sq_sqrt]
  positivity

/-- The conservative runtime radius turns the conservative complex tail
expression into `δ / 2`. -/
theorem four_mul_exp_neg_conservativeRuntimeRadius_sq_mul_div
    {noiseVar effectiveDenom delta : ℝ}
    (hNoise : 0 < noiseVar) (hEff : 0 < effectiveDenom) (hδ : 0 < delta ∧ delta < 8) :
    4 * Real.exp
        (-(edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ^ 2 * effectiveDenom) /
          (2 * noiseVar)) = delta / 2 := by
  rw [edmdConservativeRuntimeRadius_sq hNoise hEff hδ]
  have height_div_pos : 0 < 8 / delta := by
    exact div_pos (by positivity) hδ.1
  have hNoise0 : noiseVar ≠ 0 := hNoise.ne'
  have hEff0 : effectiveDenom ≠ 0 := hEff.ne'
  have hRewrite :
      -((2 * noiseVar * Real.log (8 / delta) / effectiveDenom) * effectiveDenom) / (2 * noiseVar) =
        -Real.log (8 / delta) := by
    field_simp [hNoise0, hEff0]
  rw [hRewrite]
  rw [Real.exp_neg, Real.exp_log height_div_pos]
  field_simp [hδ.1.ne']
  ring

/-- Fixed-regressor Gaussian closure: if the split real and imaginary
innovation coordinates are independent centered Gaussians with variance
`noiseVar / 2`, then the complex weighted-innovation term obeys the conservative
`re/im` union-bound tail with the exact denominator energy. -/
theorem weightedInnovation_tail_le_of_splitIndependentCenteredGaussian
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    (ξ : Fin n ⊕ Fin n → Ω → ℝ) (z : Fin n → ℂ)
    {noiseVar ε : ℝ}
    (h_indep : iIndepFun ξ μ)
    (hNoise : 0 < noiseVar)
    (h_gauss :
      ∀ j : Fin n ⊕ Fin n,
        μ.map (ξ j) = gaussianReal (0 : ℝ) ((⟨noiseVar / 2, by positivity⟩ : NNReal)))
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2)
    (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ ‖weightedInnovation (splitComplexInnovation ξ ω) z‖} ≤
      4 * Real.exp (-(ε ^ 2 * (∑ i, ‖z i‖ ^ 2)) / (2 * noiseVar)) := by
  let energy : ℝ := ∑ i, ‖z i‖ ^ 2
  let vHalf : NNReal := ⟨noiseVar / 2, by positivity⟩
  let rePart : Ω → ℝ := fun ω ↦ ∑ j, splitWeightsRe z energy j * ξ j ω
  let imPart : Ω → ℝ := fun ω ↦ ∑ j, splitWeightsIm z energy j * ξ j ω
  have hnorm :
      ∀ x : Fin n, ‖z x‖ ^ 2 = (z x).re ^ 2 + (z x).im ^ 2 := by
    intro x
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  have hsum :
      energy = ∑ x, ((z x).re ^ 2 + (z x).im ^ 2) := by
    unfold energy
    simp [hnorm, Finset.sum_add_distrib]
  let cRe : Fin n ⊕ Fin n → NNReal :=
    fun j =>
      let q : NNReal := ⟨(splitWeightsRe z energy j) ^ 2, sq_nonneg _⟩
      q * vHalf
  let cIm : Fin n ⊕ Fin n → NNReal :=
    fun j =>
      let q : NNReal := ⟨(splitWeightsIm z energy j) ^ 2, sq_nonneg _⟩
      q * vHalf
  have hReSummand :
      ∀ j : Fin n ⊕ Fin n,
        HasSubgaussianMGF (fun ω ↦ splitWeightsRe z energy j * ξ j ω) (cRe j) μ := by
    intro j
    unfold cRe
    simpa [vHalf] using
      const_mul_centeredGaussian_hasSubgaussianMGF_of_map_eq_gaussianReal
        (μ := μ) (X := ξ j) (v := vHalf) (w := splitWeightsRe z energy j) (h_gauss j)
  have hImSummand :
      ∀ j : Fin n ⊕ Fin n,
        HasSubgaussianMGF (fun ω ↦ splitWeightsIm z energy j * ξ j ω) (cIm j) μ := by
    intro j
    unfold cIm
    simpa [vHalf] using
      const_mul_centeredGaussian_hasSubgaussianMGF_of_map_eq_gaussianReal
        (μ := μ) (X := ξ j) (v := vHalf) (w := splitWeightsIm z energy j) (h_gauss j)
  have hReRaw :
      HasSubgaussianMGF rePart
        (∑ j, cRe j) μ := by
    unfold rePart
    simpa [cRe] using
      (HasSubgaussianMGF.sum_of_iIndepFun
        (μ := μ)
        (X := fun j ω ↦ splitWeightsRe z energy j * ξ j ω)
        (c := cRe)
        (s := Finset.univ)
        (h_indep.comp (fun j x ↦ splitWeightsRe z energy j * x)
          (fun j ↦ measurable_const.mul measurable_id))
        (by
          intro j _
          exact hReSummand j))
  have hImRaw :
      HasSubgaussianMGF imPart
        (∑ j, cIm j) μ := by
    unfold imPart
    simpa [cIm] using
      (HasSubgaussianMGF.sum_of_iIndepFun
        (μ := μ)
        (X := fun j ω ↦ splitWeightsIm z energy j * ξ j ω)
        (c := cIm)
        (s := Finset.univ)
        (h_indep.comp (fun j x ↦ splitWeightsIm z energy j * x)
          (fun j ↦ measurable_const.mul measurable_id))
        (by
          intro j _
          exact hImSummand j))
  have hReParamReal :
      ∑ j, (splitWeightsRe z energy j) ^ 2 * (noiseVar / 2) = noiseVar / (2 * energy) := by
    rw [← Finset.sum_mul]
    rw [splitWeightsRe_sqSum z henergy hsum]
    field_simp [henergy.ne', hNoise.ne']
    ring_nf
    have hone : (∑ x, ‖z x‖ ^ 2) * (∑ x, ‖z x‖ ^ 2)⁻¹ = 1 := by
      exact mul_inv_cancel₀ henergy.ne'
    nlinarith [hone]
  have hReParam :
      ∑ j, cRe j =
        ((⟨noiseVar / (2 * energy), by positivity⟩ : NNReal)) := by
    ext
    simpa [cRe, vHalf] using hReParamReal
  have hImParamReal :
      ∑ j, (splitWeightsIm z energy j) ^ 2 * (noiseVar / 2) = noiseVar / (2 * energy) := by
    rw [← Finset.sum_mul]
    rw [splitWeightsIm_sqSum z henergy hsum]
    field_simp [henergy.ne', hNoise.ne']
    ring_nf
    have hone : (∑ x, ‖z x‖ ^ 2) * (∑ x, ‖z x‖ ^ 2)⁻¹ = 1 := by
      exact mul_inv_cancel₀ henergy.ne'
    nlinarith [hone]
  have hImParam :
      ∑ j, cIm j =
        ((⟨noiseVar / (2 * energy), by positivity⟩ : NNReal)) := by
    ext
    simpa [cIm, vHalf] using hImParamReal
  have hRe :
      HasSubgaussianMGF rePart ((⟨noiseVar / (2 * energy), by positivity⟩ : NNReal)) μ := by
    simpa [hReParam] using hReRaw
  have hIm :
      HasSubgaussianMGF imPart ((⟨noiseVar / (2 * energy), by positivity⟩ : NNReal)) μ := by
    simpa [hImParam] using hImRaw
  have hAbsRe :
      μ.real {ω | ε / Real.sqrt 2 ≤ |rePart ω|} ≤
        2 * Real.exp (-((ε / Real.sqrt 2) ^ 2) / (2 * (noiseVar / (2 * energy)))) := by
    simpa [rePart] using
      (subgaussianAbsTail_le (μ := μ) (X := rePart)
        (c := ((⟨noiseVar / (2 * energy), by positivity⟩ : NNReal))) hRe
        (ε := ε / Real.sqrt 2) (by positivity))
  have hAbsIm :
      μ.real {ω | ε / Real.sqrt 2 ≤ |imPart ω|} ≤
        2 * Real.exp (-((ε / Real.sqrt 2) ^ 2) / (2 * (noiseVar / (2 * energy)))) := by
    simpa [imPart] using
      (subgaussianAbsTail_le (μ := μ) (X := imPart)
        (c := ((⟨noiseVar / (2 * energy), by positivity⟩ : NNReal))) hIm
        (ε := ε / Real.sqrt 2) (by positivity))
  have hExpRewrite :
      2 * Real.exp (-((ε / Real.sqrt 2) ^ 2) / (2 * (noiseVar / (2 * energy)))) =
        2 * Real.exp (-(ε ^ 2 * energy) / (2 * noiseVar)) := by
    have hsqrt2 : (Real.sqrt 2) ^ 2 = 2 := by
      nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 2 by positivity)]
    have hRewrite :
        -((ε / Real.sqrt 2) ^ 2) / (2 * (noiseVar / (2 * energy))) =
          -(ε ^ 2 * energy) / (2 * noiseVar) := by
      have hsqrt2_ne : Real.sqrt 2 ≠ 0 := by positivity
      have hhalf : (ε / Real.sqrt 2) ^ 2 = ε ^ 2 / 2 := by
        field_simp [hsqrt2_ne]
        nlinarith [hsqrt2]
      rw [hhalf]
      field_simp [hNoise.ne', henergy.ne']
    exact congrArg (fun t : ℝ => 2 * Real.exp t) hRewrite
  have hSubset :
      {ω | ε ≤ ‖weightedInnovation (splitComplexInnovation ξ ω) z‖} ⊆
        {ω | ε / Real.sqrt 2 ≤ |rePart ω|} ∪ {ω | ε / Real.sqrt 2 ≤ |imPart ω|} := by
    intro ω hω
    have hbase := complexNorm_event_subset_union
      (W := fun ω ↦ weightedInnovation (splitComplexInnovation ξ ω) z) (ε := ε) hε hω
    rcases hbase with hReEvent | hImEvent
    · left
      simpa [rePart, weightedInnovation_re_eq_splitWeightedSum ξ ω z henergy] using hReEvent
    · right
      simpa [imPart, weightedInnovation_im_eq_splitWeightedSum ξ ω z henergy] using hImEvent
  calc
    μ.real {ω | ε ≤ ‖weightedInnovation (splitComplexInnovation ξ ω) z‖}
      ≤ μ.real ({ω | ε / Real.sqrt 2 ≤ |rePart ω|} ∪
          {ω | ε / Real.sqrt 2 ≤ |imPart ω|}) :=
        MeasureTheory.measureReal_mono hSubset
    _ ≤ μ.real {ω | ε / Real.sqrt 2 ≤ |rePart ω|} +
          μ.real {ω | ε / Real.sqrt 2 ≤ |imPart ω|} :=
        MeasureTheory.measureReal_union_le _ _
    _ ≤ (2 * Real.exp (-(ε ^ 2 * energy) / (2 * noiseVar))) +
          (2 * Real.exp (-(ε ^ 2 * energy) / (2 * noiseVar))) := by
        have hAbsRe' := hAbsRe
        have hAbsIm' := hAbsIm
        rw [hExpRewrite] at hAbsRe' hAbsIm'
        linarith
    _ = 4 * Real.exp (-(ε ^ 2 * energy) / (2 * noiseVar)) := by ring

/-- The exact denominator-energy Gaussian tail can be weakened to any smaller
effective denominator, matching the runtime's ESS-corrected denominator. -/
theorem weightedInnovation_tail_le_of_splitIndependentCenteredGaussian_of_le_effectiveDenom
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {n : ℕ}
    (ξ : Fin n ⊕ Fin n → Ω → ℝ) (z : Fin n → ℂ)
    {noiseVar effectiveDenom ε : ℝ}
    (h_indep : iIndepFun ξ μ)
    (hNoise : 0 < noiseVar)
    (h_gauss :
      ∀ j : Fin n ⊕ Fin n,
        μ.map (ξ j) = gaussianReal (0 : ℝ) ((⟨noiseVar / 2, by positivity⟩ : NNReal)))
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2)
    (hEffLe : effectiveDenom ≤ ∑ i, ‖z i‖ ^ 2)
    (hε : 0 ≤ ε) :
    μ.real {ω | ε ≤ ‖weightedInnovation (splitComplexInnovation ξ ω) z‖} ≤
      4 * Real.exp (-(ε ^ 2 * effectiveDenom) / (2 * noiseVar)) := by
  have hbase :=
    weightedInnovation_tail_le_of_splitIndependentCenteredGaussian
      (μ := μ) (ξ := ξ) (z := z) (noiseVar := noiseVar) (ε := ε)
      h_indep hNoise h_gauss henergy hε
  have hmono :
      4 * Real.exp (-(ε ^ 2 * (∑ i, ‖z i‖ ^ 2)) / (2 * noiseVar)) ≤
        4 * Real.exp (-(ε ^ 2 * effectiveDenom) / (2 * noiseVar)) := by
    have hmul : ε ^ 2 * effectiveDenom ≤ ε ^ 2 * (∑ i, ‖z i‖ ^ 2) := by
      nlinarith [hEffLe]
    have hdiv :
        (ε ^ 2 * effectiveDenom) / (2 * noiseVar) ≤
          (ε ^ 2 * (∑ i, ‖z i‖ ^ 2)) / (2 * noiseVar) := by
      exact div_le_div_of_nonneg_right hmul (by positivity)
    have hexp :
        -(ε ^ 2 * (∑ i, ‖z i‖ ^ 2)) / (2 * noiseVar) ≤
          -(ε ^ 2 * effectiveDenom) / (2 * noiseVar) := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using (neg_le_neg hdiv)
    have hexp' := Real.exp_le_exp.mpr hexp
    exact mul_le_mul_of_nonneg_left hexp' (by positivity)
  exact hbase.trans hmono

/-- Fixed-regressor conservative transport theorem: once the weighted-innovation
tail is available for a deterministic retained regressor vector, the
`log (8 / δ)` radius gives an estimator error bound at level `δ / 2`. This is
the strongest honest theorem available before formalizing the missing
fixed-regressor lattice-OU tail step. -/
theorem edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_weightedInnovationTail
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : ℕ} {a : ℂ}
    {z : Fin n → ℂ} {zNext innovation : Ω → Fin n → ℂ}
    {noiseVar effectiveDenom delta : ℝ}
    (hstep : ∀ ω i, zNext ω i = a * z i + innovation ω i)
    (henergy : edmdEnergy z ≠ 0)
    (hNoise : 0 < noiseVar) (hEff : 0 < effectiveDenom) (hδ : 0 < delta ∧ delta < 8)
    (hTail :
      ∀ {ε : ℝ}, 0 ≤ ε →
        μ.real {ω | ε ≤ ‖weightedInnovation (innovation ω) z‖} ≤
          4 * Real.exp (-(ε ^ 2 * effectiveDenom) / (2 * noiseVar))) :
    μ.real {ω |
      edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ≤
        ‖edmdEstimator z (zNext ω) - a‖} ≤ delta / 2 := by
  have hstep' :
      ∀ ω i, zNext ω i = a * (Function.const Ω z ω i) + innovation ω i := by
    simpa using hstep
  have henergy' : ∀ ω, edmdEnergy (Function.const Ω z ω) ≠ 0 := by
    intro ω
    simpa using henergy
  have hRadius_nonneg : 0 ≤ edmdConservativeRuntimeRadius noiseVar effectiveDenom delta := by
    unfold edmdConservativeRuntimeRadius
    exact Real.sqrt_nonneg _
  calc
    μ.real {ω |
      edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ≤
        ‖edmdEstimator z (zNext ω) - a‖}
      ≤ 4 * Real.exp
          (-(edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ^ 2 * effectiveDenom) /
            (2 * noiseVar)) := by
        simpa using
          (edmdEstimator_error_tail_le_of_weightedInnovationTail
            (μ := μ)
            (z := Function.const Ω z)
            (zNext := zNext)
            (innovation := innovation)
            hstep' henergy' (hTail hRadius_nonneg))
    _ = delta / 2 := four_mul_exp_neg_conservativeRuntimeRadius_sq_mul_div hNoise hEff hδ

/-- Full fixed-regressor closure: a deterministic retained regressor vector and
independent centered Gaussian split innovations justify the conservative
runtime radius used by the formally safe EDMD lane. -/
theorem edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_splitIndependentCenteredGaussian
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {n : ℕ} {a : ℂ}
    (ξ : Fin n ⊕ Fin n → Ω → ℝ) (z : Fin n → ℂ) {zNext : Ω → Fin n → ℂ}
    {noiseVar effectiveDenom delta : ℝ}
    (hstep : ∀ ω i, zNext ω i = a * z i + splitComplexInnovation ξ ω i)
    (h_indep : iIndepFun ξ μ)
    (hNoise : 0 < noiseVar)
    (h_gauss :
      ∀ j : Fin n ⊕ Fin n,
        μ.map (ξ j) = gaussianReal (0 : ℝ) ((⟨noiseVar / 2, by positivity⟩ : NNReal)))
    (henergy : 0 < ∑ i, ‖z i‖ ^ 2)
    (hEff : 0 < effectiveDenom)
    (hEffLe : effectiveDenom ≤ ∑ i, ‖z i‖ ^ 2)
    (hδ : 0 < delta ∧ delta < 8) :
    μ.real {ω |
      edmdConservativeRuntimeRadius noiseVar effectiveDenom delta ≤
        ‖edmdEstimator z (zNext ω) - a‖} ≤ delta / 2 := by
  have henergy' : edmdEnergy z ≠ 0 := by
    rw [edmdEnergy_eq_sum_normSq]
    exact_mod_cast henergy.ne'
  exact
    edmdEstimator_error_tail_at_conservativeRuntimeRadius_of_weightedInnovationTail
      (μ := μ)
      (z := z)
      (zNext := zNext)
      (innovation := fun ω ↦ splitComplexInnovation ξ ω)
      hstep
      henergy'
      hNoise
      hEff
      hδ
      (by
        intro ε hε
        exact weightedInnovation_tail_le_of_splitIndependentCenteredGaussian_of_le_effectiveDenom
          (μ := μ) (ξ := ξ) (z := z) (noiseVar := noiseVar)
          (effectiveDenom := effectiveDenom) (ε := ε)
          h_indep hNoise h_gauss henergy hEffLe hε)

/-- If the weighted-innovation term satisfies the expected exponential tail
bound, the runtime radius gives the advertised EDMD estimator error bound. -/
theorem edmdEstimator_error_tail_at_runtimeRadius_of_weightedInnovationTail
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {n : ℕ} {a : ℂ}
    {z zNext innovation : Ω → Fin n → ℂ}
    {noiseVar effectiveDenom delta : ℝ}
    (hstep : ∀ ω i, zNext ω i = a * z ω i + innovation ω i)
    (henergy : ∀ ω, edmdEnergy (z ω) ≠ 0)
    (hNoise : 0 < noiseVar) (hEff : 0 < effectiveDenom) (hδ : 0 < delta ∧ delta < 2)
    (hTail :
      ∀ {ε : ℝ}, 0 ≤ ε →
        μ.real {ω | ε ≤ ‖weightedInnovation (innovation ω) (z ω)‖} ≤
          Real.exp (-(ε ^ 2 * effectiveDenom) / (2 * noiseVar))) :
    μ.real {ω |
      edmdRuntimeRadius noiseVar effectiveDenom delta ≤
        ‖edmdEstimator (z ω) (zNext ω) - a‖} ≤ delta / 2 := by
  have hRadius_nonneg : 0 ≤ edmdRuntimeRadius noiseVar effectiveDenom delta := by
    unfold edmdRuntimeRadius
    exact Real.sqrt_nonneg _
  calc
    μ.real {ω |
      edmdRuntimeRadius noiseVar effectiveDenom delta ≤
        ‖edmdEstimator (z ω) (zNext ω) - a‖}
      ≤ Real.exp (-(edmdRuntimeRadius noiseVar effectiveDenom delta ^ 2 * effectiveDenom) /
          (2 * noiseVar)) := by
        exact edmdEstimator_error_tail_le_of_weightedInnovationTail hstep henergy (hTail hRadius_nonneg)
    _ = delta / 2 := exp_neg_runtimeRadius_sq_mul_div hNoise hEff hδ

end

end HeytingLean.Physics.KoopmanGFF
