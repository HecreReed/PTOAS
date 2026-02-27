#!/usr/bin/env bash
set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi

TESTDATA_DIR=${TESTDATA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../testdata" && pwd)"}
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_stage9_out"}
mkdir -p "${OUT_DIR}"

failed=0
for f in "${TESTDATA_DIR}"/*.pto; do
  [[ -e "$f" ]] || continue
  base=$(basename "$f" .pto)

  bc1="${OUT_DIR}/${base}.ptobc"
  pto2="${OUT_DIR}/${base}.roundtrip.pto"
  bc2="${OUT_DIR}/${base}.roundtrip.ptobc"

  "${PTOBC_BIN}" encode "$f" -o "$bc1"
  "${PTOBC_BIN}" decode "$bc1" -o "$pto2"
  "${PTOBC_BIN}" encode "$pto2" -o "$bc2"

  # If xxd exists, compare bytes; otherwise just ensure we produced outputs.
  if command -v cmp >/dev/null 2>&1; then
    cmp "$bc1" "$bc2" || { echo "mismatch: $base"; failed=1; }
  fi

done

exit $failed
