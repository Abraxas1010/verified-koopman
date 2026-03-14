import Mathlib.Data.Set.Basic

namespace VerifiedKoopman
namespace SemanticClosure
namespace MR

universe u v

structure MRSystem where
  A : Type u
  B : Type v
  H : Set (A → B)
  f : A → B
  hf : f ∈ H
  Sel : Set (B → {g : A → B // g ∈ H})
  Φf : B → {g : A → B // g ∈ H}
  hΦf : Φf ∈ Sel

abbrev AdmissibleMap (S : MRSystem.{u, v}) : Type (max u v) :=
  {g : S.A → S.B // g ∈ S.H}

abbrev Selector (S : MRSystem.{u, v}) : Type (max u v) :=
  {Φ : S.B → AdmissibleMap S // Φ ∈ S.Sel}

instance (S : MRSystem.{u, v}) : CoeFun (Selector S) (fun _ => S.B → AdmissibleMap S) where
  coe Φ := Φ.1

def evalAt (S : MRSystem.{u, v}) (b : S.B) : Selector S → AdmissibleMap S :=
  fun Φ => Φ b

@[simp] lemma evalAt_apply (S : MRSystem.{u, v}) (b : S.B) (Φ : Selector S) :
    evalAt S b Φ = Φ b := rfl

def EvalAtInjective (S : MRSystem.{u, v}) (b : S.B) : Prop :=
  Function.Injective (evalAt S b)

end MR
end SemanticClosure
end VerifiedKoopman
