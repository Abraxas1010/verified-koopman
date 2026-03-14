import Lake
open Lake DSL

package verifiedKoopman where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.24.0"

@[default_target]
lean_lib VerifiedKoopman where
  globs := #[.submodules `VerifiedKoopman]

lean_exe export_semantic_closure where
  root := `Scripts.ExportSemanticClosure
