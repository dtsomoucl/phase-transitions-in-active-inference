import Lake
open Lake DSL

package «ReducedBifurcationDemo» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.22.0"

@[default_target]
lean_lib ReducedBifurcationDemo

lean_exe reducedBifurcationMain where
  root := `Main
