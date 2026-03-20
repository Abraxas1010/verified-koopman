import HeytingLean.External.OSforGFFKoopmanBridge

/-!
# Koopman/GFF Generating Observables

This file exposes the first honest Koopman-facing theorem surface for the
Gaussian free field.

We do **not** claim a full `L²(μ_GFF)` semigroup or spectral-gap theorem here.
Instead we work with the exponential cylinder observables already controlled by
the vendored `OSforGFF` development:

`A_f(ω) = exp(⟨ω, f⟩)`.

For this observable class, OSforGFF already proves time-translation duality,
stationarity, Gaussian factorization, and covariance decay. This file packages
those results under the `HeytingLean.Physics.KoopmanGFF` namespace as:

- an exact restricted correlation factorization theorem, which is the honest
  Gaussian chokepoint for any future exponential upgrade, and
- a polynomial witness theorem coming from the already-formalized OS4
  clustering/ergodicity lane.
-/

namespace HeytingLean.Physics.KoopmanGFF

open HeytingLean.External.OSforGFFKoopmanBridge
open MeasureTheory
open scoped BigOperators

noncomputable section

private abbrev bridgeConjSchwartz : TestFunctionℂ → TestFunctionℂ :=
  HeytingLean.External.OSforGFFKoopmanBridge.conjSchwartz

/-- The exponential cylinder observable attached to a complex test function. -/
def generatingObservable (f : TestFunctionℂ) (ω : FieldConfiguration) : ℂ :=
  Complex.exp (distributionPairingℂ_real ω f)

/-- The same observable after Euclidean time translation of the configuration. -/
def timeTranslatedGeneratingObservable (t : ℝ) (f : TestFunctionℂ)
    (ω : FieldConfiguration) : ℂ :=
  Complex.exp (distributionPairingℂ_real (TimeTranslation.timeTranslationDistribution t ω) f)

/-- The stationary GFF mean of the exponential cylinder observable. -/
def generatingObservableMean (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) : ℂ :=
  ∫ ω, generatingObservable f ω ∂(gaussianFreeField_free m).toMeasure

/-- The centered restricted Koopman-style correlation for a single exponential
cylinder observable. -/
def restrictedKoopmanCorrelation (m : ℝ) [Fact (0 < m)]
    (f : TestFunctionℂ) (s u : ℝ) : ℂ :=
  ∫ ω,
      timeTranslatedGeneratingObservable s f ω *
        starRingEnd ℂ (timeTranslatedGeneratingObservable u f ω)
      ∂(gaussianFreeField_free m).toMeasure -
    generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f)

/-- The Schwinger two-point function controlling the restricted correlation
through the exact Gaussian MGF factorization. -/
def translatedGeneratingSchwinger (m : ℝ) [Fact (0 < m)]
    (f : TestFunctionℂ) (t : ℝ) : ℂ :=
  SchwingerFunctionℂ₂ (gaussianFreeField_free m)
    (TimeTranslation.timeTranslationSchwartzℂ (-t) f) (bridgeConjSchwartz f)

/-- An explicit polynomial restricted-correlation witness with extractable
constant. -/
structure RestrictedKoopmanPolynomialCorrelationWitness
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) where
  constant : ℝ
  constant_nonneg : 0 ≤ constant
  bound : ∀ s u : ℝ, s ≥ 0 → u ≥ 0 →
    ‖restrictedKoopmanCorrelation m f s u‖ ≤
      constant * (1 + |s - u|)^ (-(3 : ℝ))

/-- Restricted Koopman-style covariance decay for a single generating
observable under time translation. -/
def RestrictedKoopmanCorrelationDecay (m : ℝ) [Fact (0 < m)]
    (f : TestFunctionℂ) : Prop :=
  ∃ c : ℝ, 0 ≤ c ∧
    ∀ s u : ℝ, s ≥ 0 → u ≥ 0 →
      ‖restrictedKoopmanCorrelation m f s u‖
        ≤ c * (1 + |s - u|)^ (-(3 : ℝ))

/-- Time translation of the configuration is dual to translating the test
function in the opposite direction. -/
theorem timeTranslatedGeneratingObservable_duality
    (t : ℝ) (f : TestFunctionℂ) (ω : FieldConfiguration) :
    timeTranslatedGeneratingObservable t f ω =
      generatingObservable (TimeTranslation.timeTranslationSchwartzℂ (-t) f) ω := by
  simp [timeTranslatedGeneratingObservable, generatingObservable,
    HeytingLean.External.OSforGFFKoopmanBridge.timeTranslationDistribution_pairingℂ]

/-- The mean of the exponential cylinder observable is stationary under
Euclidean time translation. -/
theorem generatingObservableMean_timeTranslation_invariant
    (m : ℝ) [Fact (0 < m)] (t : ℝ) (f : TestFunctionℂ) :
    ∫ ω, timeTranslatedGeneratingObservable t f ω ∂(gaussianFreeField_free m).toMeasure =
      generatingObservableMean m f := by
  simp only [generatingObservableMean, timeTranslatedGeneratingObservable_duality,
    generatingObservable]
  simpa using gaussianFreeField_generating_time_invariant m (-t) f

/-- The mean of the conjugated exponential cylinder observable is the complex
conjugate of the original mean. -/
theorem generatingObservableMean_conj
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) :
    ∫ ω, Complex.exp (distributionPairingℂ_real ω (bridgeConjSchwartz f))
      ∂(gaussianFreeField_free m).toMeasure =
      starRingEnd ℂ (generatingObservableMean m f) := by
  unfold generatingObservableMean generatingObservable
  rw [← integral_conj]
  congr 1
  ext ω
  rw [(Complex.exp_conj _).symm,
    HeytingLean.External.OSforGFFKoopmanBridge.distributionPairingℂ_real_conj]

/-- The restricted correlation is stationary and factors through the time
difference `s - u`. -/
theorem restrictedKoopmanCorrelation_stationary
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) (s u : ℝ) :
    restrictedKoopmanCorrelation m f s u =
      restrictedKoopmanCorrelation m f (s - u) 0 := by
  have h_stat := gaussianFreeField_product_expectation_stationarity m f s u
  have h_sub := congrArg
    (fun z =>
      z - (generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f))) h_stat
  simpa [restrictedKoopmanCorrelation, timeTranslatedGeneratingObservable, generatingObservableMean]
    using h_sub

/-- Exact Gaussian factorization of the restricted correlation at time
difference `t`. This is the honest chokepoint for any later exponential
upgrade: only the translated Schwinger two-point function remains to bound. -/
theorem restrictedKoopmanCorrelation_factorization_at_zero
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) (t : ℝ) :
    restrictedKoopmanCorrelation m f t 0 =
      generatingObservableMean m f *
        starRingEnd ℂ (generatingObservableMean m f) *
        (Complex.exp (translatedGeneratingSchwinger m f t) - 1) := by
  unfold restrictedKoopmanCorrelation translatedGeneratingSchwinger
  have h_int_rw :
      ∫ ω,
          timeTranslatedGeneratingObservable t f ω *
            starRingEnd ℂ (timeTranslatedGeneratingObservable 0 f ω)
          ∂(gaussianFreeField_free m).toMeasure =
        ∫ ω,
          Complex.exp (distributionPairingℂ_real ω (TimeTranslation.timeTranslationSchwartzℂ (-t) f) +
            distributionPairingℂ_real ω (bridgeConjSchwartz f))
          ∂(gaussianFreeField_free m).toMeasure := by
    congr 1 with ω
    rw [timeTranslatedGeneratingObservable_duality]
    rw [timeTranslatedGeneratingObservable_duality]
    simp only [generatingObservable]
    have h_conj :
        starRingEnd ℂ
            (Complex.exp
              (distributionPairingℂ_real ω (TimeTranslation.timeTranslationSchwartzℂ 0 f))) =
          Complex.exp (distributionPairingℂ_real ω (bridgeConjSchwartz f)) := by
      simp
      rw [(Complex.exp_conj _).symm,
        HeytingLean.External.OSforGFFKoopmanBridge.distributionPairingℂ_real_conj]
    have h_conj' :
        starRingEnd ℂ
            (Complex.exp
              (distributionPairingℂ_real ω (TimeTranslation.timeTranslationSchwartzℂ (-0) f))) =
          Complex.exp (distributionPairingℂ_real ω (bridgeConjSchwartz f)) := by
      simpa using h_conj
    rw [h_conj', ← Complex.exp_add]
  have h_gauss :=
    gaussianFreeField_joint_mgf_factorization m
      (TimeTranslation.timeTranslationSchwartzℂ (-t) f) (bridgeConjSchwartz f)
  have h_shifted_mean :
      ∫ ω, Complex.exp (distributionPairingℂ_real ω (TimeTranslation.timeTranslationSchwartzℂ (-t) f))
        ∂(gaussianFreeField_free m).toMeasure =
        generatingObservableMean m f := by
    simpa [generatingObservableMean, generatingObservable] using
      gaussianFreeField_generating_time_invariant m (-t) f
  have h_conj_mean := generatingObservableMean_conj m f
  have h_main :
      ∫ ω,
          timeTranslatedGeneratingObservable t f ω *
            starRingEnd ℂ (timeTranslatedGeneratingObservable 0 f ω)
          ∂(gaussianFreeField_free m).toMeasure -
        generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f) =
        generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f) *
          (Complex.exp
            (SchwingerFunctionℂ₂ (gaussianFreeField_free m)
              (TimeTranslation.timeTranslationSchwartzℂ (-t) f) (bridgeConjSchwartz f)) - 1) := by
    calc
      ∫ ω,
          timeTranslatedGeneratingObservable t f ω *
            starRingEnd ℂ (timeTranslatedGeneratingObservable 0 f ω)
          ∂(gaussianFreeField_free m).toMeasure -
        generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f)
          = generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f) *
              Complex.exp
                (SchwingerFunctionℂ₂ (gaussianFreeField_free m)
                  (TimeTranslation.timeTranslationSchwartzℂ (-t) f) (bridgeConjSchwartz f)) -
            generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f) := by
              rw [h_int_rw, h_gauss, h_shifted_mean, h_conj_mean]
      _ = generatingObservableMean m f *
            starRingEnd ℂ (generatingObservableMean m f) *
            (Complex.exp
              (SchwingerFunctionℂ₂ (gaussianFreeField_free m)
                (TimeTranslation.timeTranslationSchwartzℂ (-t) f) (bridgeConjSchwartz f)) - 1) := by
              ring
  exact h_main

/-- Exact Gaussian factorization of the restricted correlation at arbitrary
times. -/
theorem restrictedKoopmanCorrelation_factorization
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) (s u : ℝ) :
    restrictedKoopmanCorrelation m f s u =
      generatingObservableMean m f *
        starRingEnd ℂ (generatingObservableMean m f) *
        (Complex.exp (translatedGeneratingSchwinger m f (s - u)) - 1) := by
  rw [restrictedKoopmanCorrelation_stationary]
  simpa [restrictedKoopmanCorrelation, translatedGeneratingSchwinger] using
    restrictedKoopmanCorrelation_factorization_at_zero m f (s - u)

/-- General bound on the restricted correlation coming from the exact Gaussian
factorization and the universal estimate `‖e^z - 1‖ ≤ ‖z‖ e^{‖z‖}`. Any future
exponential theorem will be obtained by inserting an exponential bound on the
translated Schwinger function into this estimate. -/
theorem restrictedKoopmanCorrelation_bound_from_schwinger
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) (s u : ℝ) :
    ‖restrictedKoopmanCorrelation m f s u‖ ≤
      ‖generatingObservableMean m f‖ * ‖generatingObservableMean m f‖ *
        (‖translatedGeneratingSchwinger m f (s - u)‖ *
          Real.exp ‖translatedGeneratingSchwinger m f (s - u)‖) := by
  rw [restrictedKoopmanCorrelation_factorization]
  calc
    ‖generatingObservableMean m f * starRingEnd ℂ (generatingObservableMean m f) *
        (Complex.exp (translatedGeneratingSchwinger m f (s - u)) - 1)‖
        = ‖generatingObservableMean m f‖ * ‖starRingEnd ℂ (generatingObservableMean m f)‖ *
            ‖Complex.exp (translatedGeneratingSchwinger m f (s - u)) - 1‖ := by
            rw [norm_mul, norm_mul]
    _ = ‖generatingObservableMean m f‖ * ‖generatingObservableMean m f‖ *
          ‖Complex.exp (translatedGeneratingSchwinger m f (s - u)) - 1‖ := by
            rw [RCLike.norm_conj]
    _ ≤ ‖generatingObservableMean m f‖ * ‖generatingObservableMean m f‖ *
          (‖translatedGeneratingSchwinger m f (s - u)‖ *
            Real.exp ‖translatedGeneratingSchwinger m f (s - u)‖) := by
            gcongr
            exact exp_sub_one_bound_general _

/-- Noncomputable polynomial witness coming from the existing OS4
clustering/ergodicity infrastructure, with explicit structure fields for the
constant, its nonnegativity proof, and the bound. -/
noncomputable def gaussianFreeField_restrictedKoopmanPolynomialWitness
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) :
    RestrictedKoopmanPolynomialCorrelationWitness m f := by
  have h_cluster : OS4PolynomialClustering m :=
    gaussianFreeField_satisfies_OS4_polynomialClustering m
  let hcov := clustering_implies_covariance_decay m f h_cluster
  refine
    { constant := Classical.choose hcov
      constant_nonneg := (Classical.choose_spec hcov).1
      bound := ?_ }
  intro s u hs hu
  exact ((Classical.choose_spec hcov).2 s u hs hu)

/-- First honest Koopman-facing decay theorem for the GFF:
the centered covariance of a time-translated exponential cylinder observable
decays polynomially in the time separation. -/
theorem gaussianFreeField_restrictedKoopmanCorrelationDecay
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) :
    RestrictedKoopmanCorrelationDecay m f := by
  let h := gaussianFreeField_restrictedKoopmanPolynomialWitness m f
  exact ⟨h.constant, h.constant_nonneg, h.bound⟩

end

end HeytingLean.Physics.KoopmanGFF
