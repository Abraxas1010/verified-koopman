import Lean
import Std.Data.HashSet
import Lean.Data.Json
import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import VerifiedKoopman.NucleusReLU
import VerifiedKoopman.NucleusThreshold
import VerifiedKoopman.SemanticClosure.MR.ClosureOperator

open Lean Json Std
open VerifiedKoopman.SemanticClosure
open VerifiedKoopman.SemanticClosure.MR

namespace VerifiedKoopman
namespace Scripts

/-- Semantic stages used in the visualization. -/
inductive Stage
  | metabolism
  | repair
  | environment
  deriving DecidableEq, Inhabited, Repr

namespace Stage

def all : List Stage := [.metabolism, .repair, .environment]

def label : Stage → String
  | .metabolism => "metabolism"
  | .repair => "repair"
  | .environment => "environment"

def index : Stage → Nat
  | .metabolism => 0
  | .repair => 1
  | .environment => 2

def position : Stage → (Float × Float × Float)
  | .metabolism => (-2.3, -0.5, 0.0)
  | .repair => (2.3, -0.5, 0.0)
  | .environment => (0.0, 2.2, 0.0)

end Stage

/-- Minimal MR system mirroring the ClosingTheLoop toy example. -/
def stageSystem : MRSystem where
  A := Unit
  B := Stage
  H := Set.univ
  f := fun _ => Stage.metabolism
  hf := by simp
  Sel := Set.univ
  Φf := fun _ => ⟨fun _ => Stage.metabolism, by simp⟩
  hΦf := by simp

def constMap (s : Stage) : AdmissibleMap stageSystem :=
  ⟨fun _ => s, by simp [stageSystem]⟩

def baseSelector : Selector stageSystem :=
  ⟨fun
      | Stage.metabolism => constMap Stage.repair
      | Stage.repair => constMap Stage.environment
      | Stage.environment => constMap Stage.metabolism,
    by simp [stageSystem]⟩

def rightInverse (b : Stage) : RightInverseAt stageSystem b where
  β g :=
    ⟨fun _ => g, by simp [stageSystem]⟩
  right_inv := by intro g; rfl

def closureOnce : Selector stageSystem :=
  closeSelector stageSystem Stage.metabolism (rightInverse Stage.metabolism) baseSelector

def closureTwice : Selector stageSystem :=
  closeSelector stageSystem Stage.metabolism (rightInverse Stage.metabolism) closureOnce

/-- Extract the image of a selector as stage indices in the canonical order. -/
def selectorImage (Φ : Selector stageSystem) : List Nat :=
  Stage.all.map fun node =>
    Stage.index ((Φ node).1 ())

def selectorVariance (Φ : Selector stageSystem) : Nat :=
  let ids := selectorImage Φ
  (ids.foldl (fun (acc : HashSet Nat) i => acc.insert i) ({})).size

def selectorFrameData :
    List (String × Selector stageSystem) :=
  [("preimage", baseSelector), ("closure", closureOnce), ("fixed", closureTwice)]

def varianceDuration (var : Nat) : Float :=
  4.5 + Float.ofNat var * 1.2

/-- Helper to build `Json.num` from floats (panics on NaN). -/
def floatJson (f : Float) : Json :=
  match JsonNumber.fromFloat? f with
  | Sum.inr numVal => Json.num numVal
  | Sum.inl msg => panic! s!"non-finite float in export: {msg}"

def natJson (n : Nat) : Json :=
  Json.num (n : JsonNumber)

def ratToFloat (q : ℚ) : Float :=
  let num := Float.ofInt q.num
  let den := Float.ofNat q.den
  num / den

def ratJson (q : ℚ) : Json :=
  floatJson (ratToFloat q)

def arrayJson {n : Nat} (f : Fin n → ℚ) : Json :=
  Json.arr (Array.ofFn fun i : Fin n => ratJson (f i))

def stageNodesJson : Json :=
  Json.arr <|
    (Stage.all.map fun s =>
      let (x, y, z) := Stage.position s
      let coords :=
        Json.arr #[floatJson x, floatJson y, floatJson z]
      Json.mkObj
        [ ("id", Json.str (Stage.label s))
        , ("index", natJson (Stage.index s))
        , ("position", coords) ]).toArray

def betaDomainJson : Json :=
  let admissible :=
    Stage.all.map fun s =>
      Json.mkObj
        [ ("label", Json.str (Stage.label s))
        , ("value", natJson (Stage.index s)) ]
  let lo := Stage.index Stage.metabolism
  let hi := Stage.index Stage.environment
  Json.mkObj
    [ ("basePoint", Json.str (Stage.label Stage.metabolism))
    , ("bounds", Json.mkObj [("lo", natJson lo), ("hi", natJson hi)])
    , ("targets", Json.arr admissible.toArray) ]

def closureTimelineJson : Json :=
  Json.arr <|
    (selectorFrameData.map fun (label, Φ) =>
      let variance := selectorVariance Φ
      let duration := varianceDuration variance
      let image := selectorImage Φ
      Json.mkObj
        [ ("phase", Json.str label)
        , ("selectorImage", Json.arr (image.map natJson).toArray)
        , ("variance", natJson variance)
        , ("durationSeconds", floatJson duration) ]).toArray

def sampleVector : Fin 3 → ℚ
  | ⟨0, _⟩ => (-4) / 5
  | ⟨1, _⟩ => 7 / 20
  | ⟨2, _⟩ => 9 / 10

def thresholdVector : Fin 3 → ℚ
  | ⟨0, _⟩ => (-1) / 10
  | ⟨1, _⟩ => 1 / 2
  | ⟨2, _⟩ => 3 / 5

def reluOutput (i : Fin 3) : ℚ :=
  max (sampleVector i) 0

def thresholdOutput (i : Fin 3) : ℚ :=
  max (sampleVector i) (thresholdVector i)

def nucleiJson : Json :=
  Json.mkObj
    [ ("dimension", natJson 3)
    , ("relu",
        Json.mkObj
          [ ("input", arrayJson sampleVector)
          , ("output", arrayJson reluOutput)
          , ("theorem", Json.str "VerifiedKoopman.reluNucleus_idempotent") ])
    , ("threshold",
        Json.mkObj
          [ ("threshold", arrayJson thresholdVector)
          , ("input", arrayJson sampleVector)
          , ("output", arrayJson thresholdOutput)
          , ("theorem", Json.str "VerifiedKoopman.thresholdNucleus_fixed_iff") ]) ]

def semanticClosureJson : Json :=
  Json.mkObj
    [ ("nodes", stageNodesJson)
    , ("betaDomain", betaDomainJson)
    , ("closureIterations", closureTimelineJson) ]

def repoGitSha : IO (Option String) := do
  let out ← IO.Process.output {cmd := "git", args := #["rev-parse", "HEAD"]}
  if out.exitCode = 0 then
    pure <| some out.stdout.trim
  else
    pure none

def isoTimestamp : IO String := do
  let out ← IO.Process.output {cmd := "date", args := #["-u", "+%Y-%m-%dT%H:%M:%SZ"]}
  if out.exitCode = 0 then
    pure out.stdout.trim
  else
    pure ""

partial def findFlag? : List String → String → Option String
  | [], _ => none
  | [x], flag => if x = flag then some "" else none
  | x :: y :: rest, flag =>
      if x = flag then some y else findFlag? (y :: rest) flag

def defaultOutput : System.FilePath :=
  "build/proof_exports/semantic_closure_proof_data.json"

def writeJson (path : System.FilePath) (payload : Json) : IO Unit := do
  match path.parent with
  | some dir => IO.FS.createDirAll dir
  | none => pure ()
  IO.FS.writeFile path (payload.pretty ++ "\n")

def main (argv : List String) : IO UInt32 := do
  try
    let sha? ← repoGitSha
    let timestamp ← isoTimestamp
    let metadata :=
      Json.mkObj
        [ ("generatedAt", Json.str timestamp)
        , ("gitSha", Json.str (sha?.getD ""))
        , ("source", Json.str "verified_koopman_release/lean")
        , ("dimension", natJson 3) ]
    let payload :=
      Json.mkObj
        [ ("metadata", metadata)
        , ("semanticClosure", semanticClosureJson)
        , ("nuclei", nucleiJson) ]
    let outPath :=
      match findFlag? argv "--out" with
      | some "" => defaultOutput
      | some custom => System.FilePath.mk custom
      | none => defaultOutput
    writeJson outPath payload
    IO.println s!"[export_semantic_closure] wrote {outPath}"
    return 0
  catch e =>
    IO.eprintln s!"[export_semantic_closure] error: {e}"
    return 1

end Scripts
end VerifiedKoopman

def main (argv : List String) : IO UInt32 :=
  VerifiedKoopman.Scripts.main argv
