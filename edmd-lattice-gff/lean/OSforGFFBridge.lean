import HeytingLean.External.OSforGFFStable

/-!
# OSforGFF Bridge

Stable HeytingLean aliases for the main OSforGFF surfaces:
- the bundled Osterwalder-Schrader predicate,
- the Gaussian free field measure,
- the individual OS axioms,
- the master theorem assembling OS0-OS4.

The goal is not to rename the mathematics. The goal is to provide a predictable
`HeytingLean.External.*` namespace for downstream reuse and search.
-/

namespace HeytingLean.External.OSforGFFBridge

open MeasureTheory
open scoped BigOperators

noncomputable section

abbrev GaussianFreeFieldMeasure (m : ℝ) [Fact (0 < m)] : ProbabilityMeasure FieldConfiguration :=
  μ_GFF m

abbrev SatisfiesAllOS (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.SatisfiesAllOS μ

abbrev OS0_Analyticity (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.OS0_Analyticity μ

abbrev OS1_Regularity (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.OS1_Regularity μ

abbrev OS2_EuclideanInvariance (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.OS2_EuclideanInvariance μ

abbrev OS3_ReflectionPositivity (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.OS3_ReflectionPositivity μ

abbrev OS4_Clustering (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.OS4_Clustering μ

abbrev OS4_Ergodicity (μ : ProbabilityMeasure FieldConfiguration) : Prop :=
  _root_.OS4_Ergodicity μ

theorem gaussianFreeField_satisfies_OS0 (m : ℝ) [Fact (0 < m)] :
    OS0_Analyticity (GaussianFreeFieldMeasure m) :=
  QFT.gaussianFreeField_satisfies_OS0 m

theorem gaussianFreeField_satisfies_OS1 (m : ℝ) [Fact (0 < m)] :
    OS1_Regularity (GaussianFreeFieldMeasure m) :=
  gaussianFreeField_satisfies_OS1_revised m

theorem gaussianFreeField_satisfies_OS2 (m : ℝ) [Fact (0 < m)] :
    OS2_EuclideanInvariance (GaussianFreeFieldMeasure m) :=
  gaussian_satisfies_OS2 (μ_GFF m)
    (by exact isGaussianGJ_gaussianFreeField_free m)
    (QFT.CovarianceEuclideanInvariantℂ_μ_GFF m)

theorem gaussianFreeField_satisfies_OS3 (m : ℝ) [Fact (0 < m)] :
    OS3_ReflectionPositivity (GaussianFreeFieldMeasure m) :=
  QFT.gaussianFreeField_OS3 m

theorem gaussianFreeField_satisfies_OS4 (m : ℝ) [Fact (0 < m)] :
    OS4_Clustering (GaussianFreeFieldMeasure m) :=
  QFT.gaussianFreeField_satisfies_OS4 m

theorem gaussianFreeField_satisfies_OS4_ergodicity (m : ℝ) [Fact (0 < m)] :
    OS4_Ergodicity (GaussianFreeFieldMeasure m) :=
  OS4_Ergodicity.OS4_PolynomialClustering_implies_OS4_Ergodicity m
    (QFT.gaussianFreeField_satisfies_OS4_PolynomialClustering m 6 (by norm_num))

theorem gaussianFreeField_satisfies_all_OS_axioms (m : ℝ) [Fact (0 < m)] :
    SatisfiesAllOS (GaussianFreeFieldMeasure m) :=
  OSforGFF.gaussianFreeField_satisfies_all_OS_axioms m

end

end HeytingLean.External.OSforGFFBridge
