#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m verified_koopman.verification.lean_gate --lean-dir "$ROOT_DIR/lean"
