import Mathlib.Data.Real.Basic
import Mathlib.Order.MinMax

namespace VerifiedKoopman

open scoped Classical

noncomputable section

/-!
Bounded Heyting-style operations on a bounded nonnegative orthant.

For the ReLU nucleus on `Fin n → ℝ`, the fixed points are `∀ i, 0 ≤ v i`,
which is not a bounded lattice (no global ⊤). For “logic-like” operations we
therefore work in a bounded orthant `[0, top]^n`.

We define:
- meet: componentwise `min`
- join: componentwise `max`
- implication (bounded): `(a ↣ b)_i := if a_i ≤ b_i then top else b_i`
- negation: `¬a := a ↣ ⊥`

and prove the defining adjunction law:

  (a ⊓ c ≤ b) ↔ (c ≤ a ↣ b).
-/

variable {n : Nat} {top : ℝ}

/-- Bounded nonnegative vectors `[0, top]^n` as a subtype of `Fin n → ℝ`. -/
abbrev BVec (n : Nat) (top : ℝ) := { v : Fin n → ℝ // (∀ i, 0 ≤ v i) ∧ (∀ i, v i ≤ top) }

namespace BVec

instance : CoeFun (BVec n top) (fun _ => Fin n → ℝ) where
  coe v := v.val

theorem nonneg (v : BVec n top) : ∀ i, 0 ≤ (v : Fin n → ℝ) i := v.property.1
theorem le_top (v : BVec n top) : ∀ i, (v : Fin n → ℝ) i ≤ top := v.property.2

@[ext] theorem ext {a b : BVec n top} (h : ∀ i, a i = b i) : a = b := by
  apply Subtype.ext
  funext i
  exact h i

/-- Componentwise meet (inf): `min`. -/
def meet (a b : BVec n top) : BVec n top :=
  ⟨fun i => min (a i) (b i),
    ⟨fun i => le_min (nonneg a i) (nonneg b i),
     fun i => le_trans (min_le_left _ _) (le_top a i)⟩⟩

/-- Componentwise join (sup): `max`. -/
def join (a b : BVec n top) : BVec n top :=
  ⟨fun i => max (a i) (b i),
    ⟨fun i => le_trans (nonneg a i) (le_max_left _ _),
     fun i => max_le (le_top a i) (le_top b i)⟩⟩

/-- Bottom element: the zero vector. -/
def bot (n : Nat) (top : ℝ) (htop : 0 ≤ top) : BVec n top :=
  ⟨fun _ => 0, ⟨fun _ => le_rfl, fun _ => htop⟩⟩

/-- Bounded Heyting-style implication (requires `0 ≤ top`). -/
def himp (a b : BVec n top) (htop : 0 ≤ top) : BVec n top :=
  ⟨fun i => if a i ≤ b i then top else b i,
    ⟨fun i => by
        by_cases h : a i ≤ b i
        · simp [h, htop]
        · simp [h, nonneg b i],
     fun i => by
        by_cases h : a i ≤ b i
        · simp [h]
        · simp [h, le_top b i]⟩⟩

/-- Negation / pseudocomplement: `¬a := a ↣ ⊥`. -/
def hnot (a : BVec n top) (htop : 0 ≤ top) : BVec n top :=
  himp a (bot n top htop) htop

/--
Adjunction law (defining Heyting implication):

`a ⊓ c ≤ b` iff `c ≤ a ↣ b`.
-/
theorem himp_adjoint (a b c : BVec n top) (htop : 0 ≤ top) :
    meet a c ≤ b ↔ c ≤ himp a b htop := by
  constructor
  · intro h i
    have hmin : min (a i) (c i) ≤ b i := h i
    by_cases hab : a i ≤ b i
    · simpa [himp, hab] using (le_top c i)
    · have hab' : b i < a i := lt_of_not_ge hab
      by_cases hac : a i ≤ c i
      · have : a i ≤ b i := by simpa [min_eq_left hac] using hmin
        exact False.elim (hab this)
      · have hac' : c i < a i := lt_of_not_ge hac
        have : c i ≤ b i := by
          simpa [min_eq_right (le_of_lt hac')] using hmin
        simpa [himp, hab] using this
  · intro h i
    by_cases hab : a i ≤ b i
    · exact le_trans (min_le_left _ _) hab
    · have hc : c i ≤ b i := by
        have := h i
        simpa [himp, hab] using this
      exact le_trans (min_le_right _ _) hc

end BVec

end
end VerifiedKoopman
