import VerifiedKoopman.SemanticClosure.MR.Basic

namespace VerifiedKoopman
namespace SemanticClosure
namespace MR

universe u v

variable {S : MRSystem.{u, v}} (b : S.B)

structure InjectiveEvalAt (S : MRSystem.{u, v}) (b : S.B) : Prop where
  inj : Function.Injective (evalAt S b)

namespace InjectiveEvalAt

variable {b}

lemma eq_of_eval_eq (Hinj : InjectiveEvalAt S b) {Φ Ψ : Selector S} :
    Φ b = Ψ b → Φ = Ψ := by
  intro h
  rcases Hinj with ⟨hInj⟩
  apply hInj
  simpa using h

end InjectiveEvalAt

structure InverseEvaluation (S : MRSystem.{u, v}) (b : S.B) where
  β : AdmissibleMap S → Selector S
  right_inv : Function.RightInverse β (evalAt S b)

abbrev RightInverseAt (S : MRSystem.{u, v}) (b : S.B) : Type (max u v) :=
  InverseEvaluation S b

namespace InverseEvaluation

variable {b}

@[simp] lemma evalAt_beta (RI : InverseEvaluation S b) (g : AdmissibleMap S) :
    evalAt S b (RI.β g) = g :=
  RI.right_inv g

theorem beta_injective (RI : InverseEvaluation S b) : Function.Injective RI.β := by
  intro x y hxy
  have hEval : evalAt S b (RI.β x) = evalAt S b (RI.β y) :=
    congrArg (evalAt S b) hxy
  calc
    x = evalAt S b (RI.β x) := (evalAt_beta (S := S) (b := b) RI x).symm
    _ = evalAt S b (RI.β y) := hEval
    _ = y := evalAt_beta (S := S) (b := b) RI y

theorem evalAt_surjective (RI : InverseEvaluation S b) : Function.Surjective (evalAt S b) := by
  intro g
  exact ⟨RI.β g, evalAt_beta (S := S) (b := b) RI g⟩

end InverseEvaluation

namespace InverseEvaluation

variable {b}

noncomputable def of_evalAt_surjective (hsurj : Function.Surjective (evalAt S b)) :
    InverseEvaluation S b := by
  classical
  refine ⟨fun g => Classical.choose (hsurj g), ?_⟩
  intro g
  exact (Classical.choose_spec (hsurj g))

theorem beta_leftInverse_of_injective (Hinj : InjectiveEvalAt S b) (RI : InverseEvaluation S b) :
    Function.LeftInverse RI.β (evalAt S b) := by
  intro Φ
  apply Hinj.inj
  simpa using (RI.right_inv (evalAt S b Φ))

end InverseEvaluation

namespace RightInverseAt

variable {b}

@[simp] lemma evalAt_beta (RI : RightInverseAt S b) (g : AdmissibleMap S) :
    evalAt S b (RI.β g) = g :=
  InverseEvaluation.evalAt_beta (S := S) (b := b) RI g

theorem beta_injective (RI : RightInverseAt S b) : Function.Injective RI.β :=
  InverseEvaluation.beta_injective (S := S) (b := b) RI

theorem evalAt_surjective (RI : RightInverseAt S b) : Function.Surjective (evalAt S b) :=
  InverseEvaluation.evalAt_surjective (S := S) (b := b) RI

end RightInverseAt

structure EvalImage (S : MRSystem.{u, v}) (b : S.B) : Type (max u v) where
  g : AdmissibleMap S
  Φ : Selector S
  eval_eq : evalAt S b Φ = g

namespace EvalImage

variable {b}

def betaOnImage (x : EvalImage S b) : Selector S :=
  x.Φ

@[simp] lemma evalAt_betaOnImage (x : EvalImage S b) :
    evalAt S b (betaOnImage (S := S) (b := b) x) = x.g :=
  x.eval_eq

def ofSelector (Φ : Selector S) : EvalImage S b :=
  ⟨Φ b, Φ, rfl⟩

@[simp] lemma betaOnImage_ofSelector (Φ : Selector S) :
    betaOnImage (S := S) (b := b) (ofSelector (S := S) (b := b) Φ) = Φ :=
  rfl

def toRange (x : EvalImage S b) : Set.range (evalAt S b) :=
  ⟨x.g, ⟨x.Φ, x.eval_eq⟩⟩

end EvalImage

end MR
end SemanticClosure
end VerifiedKoopman
