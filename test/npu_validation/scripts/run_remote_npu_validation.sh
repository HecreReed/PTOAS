#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

STAGE="${STAGE:-run}"         # build|run
RUN_MODE="${RUN_MODE:-npu}"   # npu|sim
SOC_VERSION="${SOC_VERSION:-Ascend910}"
GOLDEN_MODE="${GOLDEN_MODE:-npu}"  # sim|npu|skip
PTO_ISA_COMMIT="${PTO_ISA_COMMIT:-}"
DEVICE_ID="${DEVICE_ID:-}"
SKIP_CASES="${SKIP_CASES:-}"          # comma/space separated testcase names
RUN_ONLY_CASES="${RUN_ONLY_CASES:-}"  # comma/space separated testcase names

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/test/npu_validation/scripts/generate_testcase.py" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../../test/npu_validation/scripts/generate_testcase.py" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  echo "ERROR: cannot locate repo root from SCRIPT_DIR=${SCRIPT_DIR}" >&2
  exit 1
fi

log() { echo "[$(date +'%F %T')] $*"; }

log "=== Remote NPU Validation ==="
log "STAGE=${STAGE} RUN_MODE=${RUN_MODE} SOC_VERSION=${SOC_VERSION}"
log "GOLDEN_MODE=${GOLDEN_MODE}"
log "DEVICE_ID=${DEVICE_ID:-auto}"
log "PTO_ISA_COMMIT=${PTO_ISA_COMMIT}"
log "ROOT_DIR=${ROOT_DIR}"

RESULTS_TSV="${RESULTS_TSV:-${ROOT_DIR}/remote_npu_validation_results.tsv}"
# Put all generated validation projects under a single root to avoid sprinkling
# `npu_validation/` folders under every sample directory.
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/npu_validation}"

normalize_list() {
  local s="$1"
  s="${s//$'\n'/,}"
  s="${s//$'\t'/,}"
  s="${s// /,}"
  while [[ "$s" == *",,"* ]]; do
    s="${s//,,/,}"
  done
  s="${s#,}"
  s="${s%,}"
  echo "$s"
}

list_contains() {
  local list="$1"
  local item="$2"
  [[ -n "${item}" ]] || return 1
  [[ ",${list}," == *",${item},"* ]]
}

SKIP_CASES_NORM="$(normalize_list "${SKIP_CASES}")"
RUN_ONLY_CASES_NORM="$(normalize_list "${RUN_ONLY_CASES}")"

source_rc() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  log "Sourcing ${f}"
  set +e +u +o pipefail
  # shellcheck disable=SC1090
  source "$f" || true
  set -euo pipefail
  set -o pipefail
}

for f in "$HOME/.bash_profile" "$HOME/.bashrc"; do
  source_rc "$f"
done

if [[ -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
  log "Sourcing /usr/local/Ascend/cann/set_env.sh"
  set +e +u +o pipefail
  # shellcheck disable=SC1091
  source "/usr/local/Ascend/cann/set_env.sh" || true
  set -euo pipefail
  set -o pipefail
elif [[ -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  log "Sourcing /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  set +e +u +o pipefail
  # shellcheck disable=SC1091
  source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" || true
  set -euo pipefail
  set -o pipefail
fi

log "=== Tool Versions ==="
whoami || true
hostname || true
uname -a || true
python3 --version || true
cmake --version || true
make --version || true
command -v bisheng || true
bisheng --version || true

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for d in /usr/local/Ascend/cann /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest; do
    [[ -d "$d" ]] || continue
    export ASCEND_HOME_PATH="$d"
    break
  done
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  log "ERROR: ASCEND_HOME_PATH is not set and cannot be auto-detected."
  exit 1
fi
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

# Detect the real board chip name for validation-only decisions.
# Keep SOC_VERSION unchanged so generate_testcase.py continues to choose the
# established compiler arch for Ascend910 board runs.
_board_chip=""
if command -v npu-smi &>/dev/null; then
  _board_chip="$(timeout 5 npu-smi info -l 2>/dev/null | grep -i 'Chip Name' | head -1 | sed 's/.*: *//' | tr -d ' ' || true)"
  if [[ -n "${_board_chip}" ]]; then
    log "Detected board chip from npu-smi: ${_board_chip} (compile SOC_VERSION stays ${SOC_VERSION})"
  fi
fi

if ! command -v bisheng >/dev/null 2>&1; then
  if [[ -x "${ASCEND_HOME_PATH}/bin/bisheng" ]]; then
    export PATH="${ASCEND_HOME_PATH}/bin:${PATH}"
  fi
fi

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# Some CANN installs do not provide a simulator directory named exactly
# "Ascend910". Map it to a real directory so we can link/run camodel.
SIM_SOC_VERSION="${SOC_VERSION}"
if [[ "${SOC_VERSION}" == "Ascend910" ]]; then
  if [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910A/lib" ]]; then
    SIM_SOC_VERSION="Ascend910A"
  elif [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910ProA/lib" ]]; then
    SIM_SOC_VERSION="Ascend910ProA"
  fi
fi

# Detect A3 (Ascend910B) board for golden-script gating.
# This is separate from SOC_VERSION/SIM_SOC_VERSION used for compilation
# to avoid changing the compiler arch (dav-c220 vs dav-c310).
export PTOAS_BOARD_IS_A3=0
if [[ "$(printf '%s' "${_board_chip}" | tr '[:upper:]' '[:lower:]')" == *910b* ]]; then
  export PTOAS_BOARD_IS_A3=1
  log "Detected A3 board from npu-smi chip name: ${_board_chip}"
fi
for _sim_dir in "${ASCEND_HOME_PATH}/aarch64-linux/simulator" \
                "${ASCEND_HOME_PATH}/simulator" \
                "${ASCEND_HOME_PATH}/tools/simulator"; do
  for _d in "${_sim_dir}"/Ascend910B*/lib; do
    if [[ -d "$_d" ]]; then
      export PTOAS_BOARD_IS_A3=1
      log "Detected A3 board (Ascend910B) from simulator dir: $_d"
      break 2
    fi
  done
done
log "SIM_SOC_VERSION=${SIM_SOC_VERSION}"

LD_LIBRARY_PATH_NPU="${LD_LIBRARY_PATH}"
LD_LIBRARY_PATH_SIM="${LD_LIBRARY_PATH}"
for d in \
  "${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SIM_SOC_VERSION}/lib" \
  "${ASCEND_HOME_PATH}/simulator/${SIM_SOC_VERSION}/lib" \
  "${ASCEND_HOME_PATH}/tools/simulator/${SIM_SOC_VERSION}/lib"; do
  [[ -d "$d" ]] && LD_LIBRARY_PATH_SIM="$d:${LD_LIBRARY_PATH_SIM}"
done

if [[ "${STAGE}" == "run" ]]; then
  log "=== NPU Device Check ==="
  id || true
  ls -l /dev/davinci* 2>/dev/null || true
  available_devnodes=()
  auto_device_ids=()
  visible_phys_ids=()
  shopt -s nullglob
  for node in /dev/davinci[0-9]*; do
    [[ -e "${node}" ]] || continue
    available_devnodes+=("${node}")
  done
  shopt -u nullglob

  # In containers, ACL expects logical device ids [0..N-1], while /dev/davinciX
  # often exposes physical ids. Prefer visible-device mapping for runtime ids.
  if [[ -n "${ASCEND_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _vis_raw <<< "${ASCEND_VISIBLE_DEVICES}"
    unset IFS
    for v in "${_vis_raw[@]}"; do
      v="${v//[[:space:]]/}"
      [[ -n "${v}" ]] || continue
      [[ "${v}" =~ ^[0-9]+$ ]] || continue
      visible_phys_ids+=("${v}")
    done
  fi

  if [[ -z "${DEVICE_ID}" ]]; then
    [[ ${#available_devnodes[@]} -gt 0 ]] || {
      log "ERROR: no /dev/davinciN device found"
      exit 1
    }
    IFS=$'\n' available_devnodes=($(printf '%s\n' "${available_devnodes[@]}" | sort -V))
    unset IFS
    for devnode in "${available_devnodes[@]}"; do
      [[ -r "${devnode}" && -w "${devnode}" ]] || {
        log "WARN: skip ${devnode} (need read/write access)"
        continue
      }
      # Keep this for diagnostics only; runtime ids are built below.
      :
    done
    [[ ${#available_devnodes[@]} -gt 0 ]] || {
      log "ERROR: no accessible /dev/davinciN device found"
      exit 1
    }

    logical_count=0
    if [[ ${#visible_phys_ids[@]} -gt 0 ]]; then
      logical_count=${#visible_phys_ids[@]}
      log "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES} (logical count=${logical_count})"
    else
      logical_count=${#available_devnodes[@]}
      log "ASCEND_VISIBLE_DEVICES not set; fallback logical count from /dev nodes=${logical_count}"
    fi
    for ((i=0; i<logical_count; ++i)); do
      auto_device_ids+=("${i}")
    done
    [[ ${#auto_device_ids[@]} -gt 0 ]] || {
      log "ERROR: failed to construct logical DEVICE_ID candidates"
      exit 1
    }
    log "Auto-select logical DEVICE_ID candidates: ${auto_device_ids[*]}"
  else
    [[ "${DEVICE_ID}" =~ ^[0-9]+$ ]] || {
      log "ERROR: DEVICE_ID must be a non-negative integer, got: ${DEVICE_ID}";
      exit 1;
    }
    # With container remapping, logical DEVICE_ID may not equal /dev/davinciX suffix.
    # Validate by visible range first when available.
    if [[ ${#visible_phys_ids[@]} -gt 0 ]]; then
      if (( DEVICE_ID < 0 || DEVICE_ID >= ${#visible_phys_ids[@]} )); then
        log "ERROR: DEVICE_ID=${DEVICE_ID} out of logical range [0, ${#visible_phys_ids[@]}) under ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES}"
        exit 1
      fi
    fi
  fi
  python3 -c "import numpy as np; print('numpy', np.__version__)" >/dev/null
fi

PTO_ISA_ROOT="${ASCEND_HOME_PATH}/include/pto"
# Allow CI to vendor a pto-isa working tree into the payload (no `.git`).
# This avoids requiring outbound GitHub connectivity on the remote NPU host.

status=0
ok_count=0
fail_count=0
skip_count=0
printf "testcase\tstatus\tstage\tinfo\n" > "${RESULTS_TSV}"
while IFS= read -r -d '' cpp; do
  # macOS tarballs may contain AppleDouble metadata files like `._foo-pto.cpp`.
  # They are not valid C++ sources; skip them.
  if [[ "$(basename "${cpp}")" == ._* ]]; then
    continue
  fi

  base="$(basename "${cpp}" .cpp)"
  testcase="${base}"
  testcase="${testcase%-pto}"
  testcase="${testcase%_pto}"

  # AsyncComm smoke sample issues async remote DMA against plain local buffers.
  # In board-runtime STAGE=run this can trigger invalid MPU access on single-rank
  # execution, so skip it in runtime stage.
  if [[ "${STAGE}" == "run" && "${testcase}" == "async_comm" ]]; then
    skip_count=$((skip_count + 1))
    printf "%s\tSKIP\t%s\truntime skip: async_comm\n" "${testcase}" "${STAGE}" >> "${RESULTS_TSV}"
    log "SKIP: ${testcase} (runtime skip)"
    continue
  fi

  if [[ -n "${RUN_ONLY_CASES_NORM}" ]] && ! list_contains "${RUN_ONLY_CASES_NORM}" "${testcase}"; then
    continue
  fi
  if [[ -n "${SKIP_CASES_NORM}" ]] && list_contains "${SKIP_CASES_NORM}" "${testcase}"; then
    skip_count=$((skip_count + 1))
    printf "%s\tSKIP\t%s\tlisted in SKIP_CASES\n" "${testcase}" "${STAGE}" >> "${RESULTS_TSV}"
    log "SKIP: ${testcase} (SKIP_CASES)"
    continue
  fi
  if [[ "${testcase}" == "gemvmx" ]]; then
    soc_lc="$(printf '%s' "${SOC_VERSION:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ "$soc_lc" != *"a5"* && "$soc_lc" != *"950"* ]]; then
      skip_count=$((skip_count + 1))
      printf "%s\tSKIP\t%s\trequires A5 (set SOC_VERSION to A5/950)\n" "${testcase}" "${STAGE}" >> "${RESULTS_TSV}"
      log "SKIP: ${testcase} (requires A5 SOC_VERSION)"
      continue
    fi
    if [[ "${PTOAS_BOARD_IS_A3:-0}" == "1" ]]; then
      skip_count=$((skip_count + 1))
      printf "%s\tSKIP\t%s\trequires A5 board\n" "${testcase}" "${STAGE}" >> "${RESULTS_TSV}"
      log "SKIP: ${testcase} (requires A5 board)"
      continue
    fi
  fi

  echo
  log "=== CASE: ${cpp} ==="

  case_dir="$(cd "$(dirname "${cpp}")" && pwd)"
  sample_name="$(basename "${case_dir}")"
  nv_dir="${OUTPUT_ROOT}/${sample_name}/${testcase}"

  set +e
  python3 "${ROOT_DIR}/test/npu_validation/scripts/generate_testcase.py" \
    --input "${cpp}" \
    --testcase "${testcase}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-mode "${RUN_MODE}" \
    --soc-version "${SIM_SOC_VERSION}"
  gen_rc=$?
  set -euo pipefail
  if [[ $gen_rc -ne 0 ]]; then
    status=1
    fail_count=$((fail_count + 1))
    printf "%s\tFAIL\tgen\texit=%s\n" "${testcase}" "${gen_rc}" >> "${RESULTS_TSV}"
    log "ERROR: generate_testcase failed (exit ${gen_rc}): ${testcase}"
    continue
  fi

  set +e
  (
    set -euo pipefail
    cd "${nv_dir}"

    CUSTOM_GOLDEN=0
    CUSTOM_COMPARE=0
    if [[ -f "./validation_meta.env" ]]; then
      # shellcheck disable=SC1091
      source "./validation_meta.env"
    fi

    enable_sim_golden="OFF"
    [[ "${GOLDEN_MODE}" == "sim" ]] && enable_sim_golden="ON"
    cmake -S . -B ./build \
      -DSOC_VERSION="${SIM_SOC_VERSION}" \
      -DENABLE_SIM_GOLDEN="${enable_sim_golden}" \
      -DPTO_ISA_ROOT="${PTO_ISA_ROOT}"
    cmake --build ./build --parallel

    if [[ "${STAGE}" != "run" ]]; then
      log "BUILD OK: ${testcase}"
      exit 0
    fi

    run_case_once() {
      case "${GOLDEN_MODE}" in
        sim)
          python3 ./golden.py || return $?
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH_SIM}" ./build/${testcase}_sim || return $?
          copy_outputs_as_golden || return $?
          if [[ "${RUN_MODE}" == "npu" ]]; then
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase} || return $?
          fi
          COMPARE_STRICT=1 python3 ./compare.py || return $?
          ;;
        npu)
          if [[ "${RUN_MODE}" != "npu" ]]; then
            log "ERROR: GOLDEN_MODE=npu requires RUN_MODE=npu"
            return 2
          fi
          python3 ./golden.py || return $?
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase} || return $?
          copy_outputs_as_golden || return $?
          python3 ./golden.py || return $?
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase} || return $?
          COMPARE_STRICT=1 python3 ./compare.py || return $?
          ;;
        skip)
          python3 ./golden.py || return $?
          if [[ "${RUN_MODE}" == "npu" ]]; then
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase} || return $?
          fi
          log "WARN: compare skipped (GOLDEN_MODE=skip)"
          ;;
        *)
          log "ERROR: unknown GOLDEN_MODE=${GOLDEN_MODE} (expected: sim|npu|skip)"
          return 2
          ;;
      esac
    }

    copy_outputs_as_golden() {
      if [[ -f "./outputs.txt" ]]; then
        while IFS= read -r name; do
          [[ -n "${name}" ]] || continue
          cp -f "./${name}.bin" "./golden_${name}.bin"
      done < "./outputs.txt"
        return 0
      fi
      for f in ./*.bin; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"
        cp -f "$f" "./golden_${base}"
      done
    }

    candidate_device_ids=("${DEVICE_ID}")
    if [[ -z "${DEVICE_ID}" ]]; then
      candidate_device_ids=("${auto_device_ids[@]}")
    fi

    last_run_rc=1
    for run_device_id in "${candidate_device_ids[@]}"; do
      if [[ -z "${DEVICE_ID}" ]]; then
        log "Trying DEVICE_ID=${run_device_id} for ${testcase}"
      fi
      if (
        set -euo pipefail
        export ACL_DEVICE_ID="${run_device_id}"
        run_case_once
      ); then
        if [[ -z "${DEVICE_ID}" ]]; then
          log "Selected DEVICE_ID=${run_device_id} for ${testcase}"
        fi
        last_run_rc=0
        break
      else
        last_run_rc=$?
      fi
    done
    [[ ${last_run_rc} -eq 0 ]] || exit "${last_run_rc}"
    log "OK: ${testcase}"
  )
  case_rc=$?
  set -euo pipefail
  if [[ $case_rc -ne 0 ]]; then
    status=1
    fail_count=$((fail_count + 1))
    printf "%s\tFAIL\t%s\texit=%s\n" "${testcase}" "${STAGE}" "${case_rc}" >> "${RESULTS_TSV}"
    log "ERROR: testcase failed (exit ${case_rc}): ${testcase}"
  else
    ok_count=$((ok_count + 1))
    printf "%s\tOK\t%s\t-\n" "${testcase}" "${STAGE}" >> "${RESULTS_TSV}"
  fi
done < <(find "${ROOT_DIR}/test/samples" -type f -name '*-pto.cpp' -print0)

log "=== SUMMARY ==="
log "OK=${ok_count} FAIL=${fail_count} SKIP=${skip_count}"
log "RESULTS_TSV=${RESULTS_TSV}"

exit "${status}"
