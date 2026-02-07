#!/usr/bin/env bash
set -euo pipefail

RUN_MODE="@RUN_MODE@"
SOC_VERSION="@SOC_VERSION@"
BUILD_DIR="${BUILD_DIR:-build}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/golden.py"

# Best-effort resolve PTO_ISA_ROOT for generated CMakeLists.txt.
if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
  search_dir="${ROOT_DIR}"
  for _ in {1..8}; do
    if [[ -d "${search_dir}/pto-isa/include" && -d "${search_dir}/pto-isa/tests/common" ]]; then
      PTO_ISA_ROOT="${search_dir}/pto-isa"
      break
    fi
    if [[ "${search_dir}" == "/" ]]; then
      break
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  export PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
fi

# Best-effort load Ascend/CANN environment (toolchains + runtime). Be careful with set -euo pipefail.
if [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  echo "[INFO] Sourcing /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  set +e
  set +u
  set +o pipefail
  source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" || true
  set -o pipefail
  set -u
  set -e
fi

# Improve runtime linking robustness.
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
  if [[ "${RUN_MODE}" == "sim" ]]; then
    for d in \
      "${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SOC_VERSION}/lib" \
      "${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib" \
      "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"; do
      [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
    done
  fi
fi

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cd "${ROOT_DIR}/${BUILD_DIR}"
if [[ -n "${PTO_ISA_ROOT:-}" ]]; then
  cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" -DPTO_ISA_ROOT="${PTO_ISA_ROOT}" ..
else
  cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" ..
fi
make -j

cd "${ROOT_DIR}"
"${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"

python3 "${ROOT_DIR}/compare.py"
