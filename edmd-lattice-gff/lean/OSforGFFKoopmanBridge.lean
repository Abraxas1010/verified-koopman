import HeytingLean.External.OSforGFFBridge
import OSforGFF.OS.OS4_Clustering
import OSforGFF.OS.OS4_Ergodicity

/-!
# OSforGFF Koopman Bridge

Stable HeytingLean aliases for the specific `OSforGFF` surfaces used by the
Koopman/GFF development.

This widens the external bridge just enough to avoid direct vendored imports in
`Physics/KoopmanGFF/*` while keeping the real mathematical content in the
upstream package.
-/

namespace HeytingLean.External.OSforGFFKoopmanBridge

open MeasureTheory
open scoped BigOperators

noncomputable section

abbrev GaussianFreeFieldMeasure (m : ℝ) [Fact (0 < m)] :
    ProbabilityMeasure FieldConfiguration :=
  HeytingLean.External.OSforGFFBridge.GaussianFreeFieldMeasure m

abbrev OS4PolynomialClustering (m : ℝ) [Fact (0 < m)] : Prop :=
  OS4_Ergodicity.OS4''_Clustering m

/-- Stable alias for complex-conjugating a Schwartz test function. -/
abbrev conjSchwartz : TestFunctionℂ → TestFunctionℂ :=
  _root_.conjSchwartz

/-- Stable alias for conjugating the distribution pairing on complex test
functions. -/
theorem distributionPairingℂ_real_conj
    (ω : FieldConfiguration) (f : TestFunctionℂ) :
    starRingEnd ℂ (distributionPairingℂ_real ω f) =
      distributionPairingℂ_real ω (conjSchwartz f) := by
  simpa [conjSchwartz] using _root_.distributionPairingℂ_real_conj ω f

theorem timeTranslationDistribution_pairingℂ
    (s : ℝ) (ω : FieldConfiguration) (g : TestFunctionℂ) :
    distributionPairingℂ_real (TimeTranslation.timeTranslationDistribution s ω) g =
      distributionPairingℂ_real ω (TimeTranslation.timeTranslationSchwartzℂ (-s) g) :=
  OS4infra.timeTranslationDistribution_pairingℂ s ω g

theorem gaussianFreeField_generating_time_invariant
    (m : ℝ) [Fact (0 < m)] (s : ℝ) (f : TestFunctionℂ) :
    ∫ ω, Complex.exp (distributionPairingℂ_real ω (TimeTranslation.timeTranslationSchwartzℂ s f))
      ∂(gaussianFreeField_free m).toMeasure =
    ∫ ω, Complex.exp (distributionPairingℂ_real ω f)
      ∂(gaussianFreeField_free m).toMeasure :=
  OS4infra.gff_generating_time_invariant m s f

theorem gaussianFreeField_joint_mgf_factorization
    (m : ℝ) [Fact (0 < m)] (f g : TestFunctionℂ) :
    (∫ ω, Complex.exp (distributionPairingℂ_real ω f + distributionPairingℂ_real ω g)
      ∂(gaussianFreeField_free m).toMeasure) =
    (∫ ω, Complex.exp (distributionPairingℂ_real ω f) ∂(gaussianFreeField_free m).toMeasure) *
    (∫ ω, Complex.exp (distributionPairingℂ_real ω g) ∂(gaussianFreeField_free m).toMeasure) *
    Complex.exp (SchwingerFunctionℂ₂ (gaussianFreeField_free m) f g) :=
  OS4infra.gff_joint_mgf_factorization m f g

theorem gaussianFreeField_product_expectation_stationarity
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) (s u : ℝ) :
    let μ := (gaussianFreeField_free m).toMeasure
    let A := fun t ω => Complex.exp (distributionPairingℂ_real (TimeTranslation.timeTranslationDistribution t ω) f)
    ∫ ω, A s ω * starRingEnd ℂ (A u ω) ∂μ =
      ∫ ω, A (s - u) ω * starRingEnd ℂ (A 0 ω) ∂μ :=
  OS4_Ergodicity.gff_product_expectation_stationarity m f s u

theorem gaussianFreeField_covariance_timeTranslation_continuous
    (m : ℝ) [Fact (0 < m)] (f g : TestFunctionℂ) :
    Continuous (fun s => SchwingerFunctionℂ₂ (gaussianFreeField_free m)
      (TimeTranslation.timeTranslationSchwartzℂ s f) g) :=
  OS4_Ergodicity.gff_covariance_timeTranslation_continuous m f g

theorem translatedSchwinger_twoPoint_eq_bilinear
    (m : ℝ) [Fact (0 < m)] (f g : TestFunctionℂ) (s : ℝ) :
    SchwingerFunctionℂ₂ (gaussianFreeField_free m) f
        (TimeTranslation.timeTranslationSchwartzℂ (-s) g) =
      ∫ x : SpaceTime, ∫ y : SpaceTime,
        f x * (freeCovarianceKernel m (x - y) : ℂ) * g (y - TimeTranslation.timeShiftConst s) :=
  QFT.schwinger2_time_translated_eq_bilinear m f g s

theorem gaussianFreeField_satisfies_OS4_polynomialClustering
    (m : ℝ) [Fact (0 < m)] :
    OS4PolynomialClustering m := by
  simpa [OS4_Ergodicity.OS4''_Clustering] using
    QFT.gaussianFreeField_satisfies_OS4_PolynomialClustering m 6 (by norm_num)

theorem clustering_implies_covariance_decay
    (m : ℝ) [Fact (0 < m)] (f : TestFunctionℂ) (h_cluster : OS4PolynomialClustering m) :
    ∃ (c : ℝ), c ≥ 0 ∧ ∀ s u : ℝ, s ≥ 0 → u ≥ 0 →
      let μ := (gaussianFreeField_free m).toMeasure
      let A := fun t ω => Complex.exp (distributionPairingℂ_real (TimeTranslation.timeTranslationDistribution t ω) f)
      let EA := ∫ ω, Complex.exp (distributionPairingℂ_real ω f) ∂μ
      ‖∫ ω, A s ω * starRingEnd ℂ (A u ω) ∂μ - EA * starRingEnd ℂ EA‖ ≤
        c * (1 + |s - u|)^ (-(3 : ℝ)) :=
  OS4_Ergodicity.clustering_implies_covariance_decay m f h_cluster

theorem exp_sub_one_bound_general (x : ℂ) :
    ‖Complex.exp x - 1‖ ≤ ‖x‖ * Real.exp ‖x‖ :=
  OS4infra.exp_sub_one_bound_general x

end

end HeytingLean.External.OSforGFFKoopmanBridge
