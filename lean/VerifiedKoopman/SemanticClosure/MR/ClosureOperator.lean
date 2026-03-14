import VerifiedKoopman.SemanticClosure.MR.InverseEvaluation

namespace VerifiedKoopman
namespace SemanticClosure
namespace MR

universe u v

variable {S : MRSystem.{u, v}} {b : S.B}

def closeSelector (S : MRSystem.{u, v}) (b : S.B) (RI : RightInverseAt S b) :
    Selector S → Selector S :=
  fun Φ => RI.β (Φ b)

namespace closeSelector

variable (RI : RightInverseAt S b)

@[simp] lemma evalAt_close (Φ : Selector S) :
    evalAt S b (closeSelector S b RI Φ) = Φ b := by
  simpa [closeSelector] using RI.right_inv (Φ b)

@[simp] lemma close_apply_at (Φ : Selector S) :
    closeSelector S b RI Φ b = Φ b := by
  simpa [evalAt] using (evalAt_close (S := S) (b := b) (RI := RI) Φ)

theorem idem (Φ : Selector S) :
    closeSelector S b RI (closeSelector S b RI Φ) = closeSelector S b RI Φ := by
  dsimp [closeSelector]
  congr
  simpa [evalAt] using (RI.right_inv (Φ b))

end closeSelector

namespace closeSelector

open Classical

variable {S : MRSystem.{u, v}} {b : S.B}

noncomputable def of_evalAt_surjective (S : MRSystem.{u, v}) (b : S.B)
    (hsurj : Function.Surjective (evalAt S b)) : Selector S → Selector S :=
  closeSelector S b (InverseEvaluation.of_evalAt_surjective (S := S) (b := b) hsurj)

theorem of_evalAt_surjective_idem (hsurj : Function.Surjective (evalAt S b)) (Φ : Selector S) :
    of_evalAt_surjective S b hsurj (of_evalAt_surjective S b hsurj Φ) =
      of_evalAt_surjective S b hsurj Φ := by
  let RI : RightInverseAt S b :=
    InverseEvaluation.of_evalAt_surjective (S := S) (b := b) hsurj
  simpa [of_evalAt_surjective, RI] using
    (closeSelector.idem (S := S) (b := b) (RI := RI) Φ)

theorem of_evalAt_surjective_eq_id (Hinj : InjectiveEvalAt S b)
    (hsurj : Function.Surjective (evalAt S b)) (Φ : Selector S) :
    of_evalAt_surjective S b hsurj Φ = Φ := by
  let RI : RightInverseAt S b :=
    InverseEvaluation.of_evalAt_surjective (S := S) (b := b) hsurj
  have hLeft : Function.LeftInverse RI.β (evalAt S b) :=
    InverseEvaluation.beta_leftInverse_of_injective (S := S) (b := b) Hinj RI
  simpa [of_evalAt_surjective, closeSelector, evalAt] using (hLeft Φ)

end closeSelector

def IsClosed (S : MRSystem.{u, v}) (b : S.B) (RI : RightInverseAt S b) (Φ : Selector S) : Prop :=
  closeSelector S b RI Φ = Φ

namespace IsClosed

variable (RI : RightInverseAt S b)

lemma iff_eq (Φ : Selector S) :
    IsClosed S b RI Φ ↔ Φ = RI.β (Φ b) := by
  simp [IsClosed, closeSelector, eq_comm]

lemma exists_eq_beta_iff (Φ : Selector S) :
    IsClosed S b RI Φ ↔ ∃ g : AdmissibleMap S, RI.β g = Φ := by
  constructor
  · intro h
    refine ⟨Φ b, ?_⟩
    simpa [IsClosed, closeSelector] using h
  · rintro ⟨g, rfl⟩
    have hb : RI.β g b = g := by
      simpa [evalAt] using (RightInverseAt.evalAt_beta (S := S) (b := b) RI g)
    simp [IsClosed, closeSelector, hb]

lemma close_isClosed (Φ : Selector S) :
    IsClosed S b RI (closeSelector S b RI Φ) := by
  simpa [IsClosed] using (closeSelector.idem (S := S) (b := b) (RI := RI) Φ)

end IsClosed

end MR
end SemanticClosure
end VerifiedKoopman
