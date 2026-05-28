#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ASCEND_ENV_PATH="${ASCEND_HOME_PATH}/bin"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"
CMAKE_ARGS=""
HARDENING_CACHE_FILE="${BASE_PATH}/cmake/LinuxHardeningCache.cmake"
LLVM_GIT_URL="https://gitcode.com/GitHub_Trending/ll/llvm-project.git"
LLVM_GIT_REF="llvmorg-19.1.7"
LLVM_CLONE_RETRY_COUNT=3
LLVM_CLONE_RETRY_INTERVAL=5

#print usage message
usage() {
  echo "Usage:"
  echo ""
  echo "    -h, --help  Print usage"
  echo "    --pkg Build run package"
  echo ""
}

print_success() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_GREEN}[SUCCESS] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

ensure_hardening_cache() {
  if [ ! -f "${HARDENING_CACHE_FILE}" ]; then
    print_error "missing hardening cache: ${HARDENING_CACHE_FILE}"
    exit 1
  fi
}

has_rpath() {
  local path="$1"
  if command -v patchelf >/dev/null 2>&1; then
    local rpath_value
    rpath_value="$(patchelf --print-rpath "$path" 2>/dev/null || true)"
    [[ -n "$rpath_value" ]]
    return
  fi
  readelf -d "$path" 2>/dev/null | grep -Eq '(RPATH|RUNPATH)'
}

remove_rpath() {
  local path="$1"
  if ! has_rpath "$path"; then
    return
  fi
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --remove-rpath "$path" || true
  fi
  if has_rpath "$path" && command -v chrpath >/dev/null 2>&1; then
    chrpath -d "$path" || true
  fi
}

strip_binary() {
  local path="$1"
  if ! command -v strip >/dev/null 2>&1; then
    return
  fi
  strip --strip-unneeded "$path" 2>/dev/null || strip "$path" 2>/dev/null || true
}

harden_package_artifacts() {
  local ptoas_bin="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"
  local llvm_lib_dir="${LLVM_BUILD_DIR}/lib"

  if [ -f "${ptoas_bin}" ]; then
    remove_rpath "${ptoas_bin}"
    strip_binary "${ptoas_bin}"
  fi

  if [ -d "${llvm_lib_dir}" ]; then
    while IFS= read -r so_path; do
      remove_rpath "${so_path}"
      strip_binary "${so_path}"
    done < <(find "${llvm_lib_dir}" -maxdepth 1 -type f -name '*.so*' | sort)
  fi
}

clone_llvm_source() {
  local target_dir="$1"
  local attempt=1

  rm -rf "${target_dir}"

  if [ -d "${CANN_3RD_LIB_PATH}/llvm-19" ]; then
    cp -r "${CANN_3RD_LIB_PATH}/llvm-19" "${target_dir}"
    return 0
  fi

  while [ "${attempt}" -le "${LLVM_CLONE_RETRY_COUNT}" ]; do
    if git -c http.version=HTTP/1.1 clone \
      --depth 1 \
      --single-branch \
      --branch "${LLVM_GIT_REF}" \
      "${LLVM_GIT_URL}" \
      "${target_dir}"; then
      return 0
    fi

    rm -rf "${target_dir}"

    if [ "${attempt}" -lt "${LLVM_CLONE_RETRY_COUNT}" ]; then
      sleep "${LLVM_CLONE_RETRY_INTERVAL}"
    fi

    attempt=$((attempt + 1))
  done

  print_error "failed to prepare llvm-project source"
  exit 1
}

configure_llvm_build() {
  cmake -C "${HARDENING_CACHE_FILE}" -G Ninja -S llvm -B "${LLVM_BUILD_DIR}" \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="host" \
    "$@"
}

configure_ptoas_build() {
  cmake -C "${HARDENING_CACHE_FILE}" -G Ninja \
    -S . \
    -B build \
    -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
    -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DPython3_FIND_STRATEGY=LOCATION \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_USE_LINKER=lld \
    -DMLIR_PYTHON_PACKAGE_DIR="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core" \
    -DCMAKE_INSTALL_PREFIX="${PTO_INSTALL_DIR}" \
    "$@"
}

checkopts() {
  ENABLE_BUILD_ALL=FALSE
  ENABLE_BUILD_ONLY=FALSE
  ENABLE_PACKAGE=FALSE

  parsed_args=$(getopt -a -o j:hvuO: -l help,pkg,build,cann_3rd_lib_path: -- "$@") || {
  usage
  exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --build)
        shift
        ENABLE_BUILD_ONLY=TRUE
        ;;
      --cann_3rd_lib_path)
        shift
        CANN_3RD_LIB_PATH="$1"
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=TRUE"
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
}

build_only() {
  echo $dotted_line
  echo "build only"
  ensure_hardening_cache
  export LLVM_SOURCE_DIR=$WORKSPACE/llvm-project
  clone_llvm_source "${LLVM_SOURCE_DIR}"
  export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
  export PTO_SOURCE_DIR=$WORKSPACE
  export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

  cd $LLVM_SOURCE_DIR
  rm -rf "${LLVM_BUILD_DIR}"

  if [ -d "$CANN_3RD_LIB_PATH/llvm-19" ]; then
    configure_llvm_build -DLLVM_ENABLE_ZSTD=OFF
  else
    configure_llvm_build
  fi

  ninja -C $LLVM_BUILD_DIR

  cd $PTO_SOURCE_DIR
  export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

  if [ -d "$CANN_3RD_LIB_PATH/llvm-19" ]; then
    configure_ptoas_build
  else
    configure_ptoas_build
  fi

  ninja -C build
  ninja -C build install
  harden_package_artifacts

  export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
  export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
  export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
  export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
  export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH

  bash test/samples/runop.sh --enablebc all
 STAGE="${STAGE:-run}" RUN_MODE='npu' SOC_VERSION='Ascend910' SKIP_CASES='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print' bash test/npu_validation/scripts/run_remote_npu_validation.sh

  echo "execute samples success"
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf ${BUILD_OUT_PATH}
  fi
}

package() {
  echo $dotted_line
  echo "package start"
  ensure_hardening_cache
  clean_build_out
  clean_build
  mkdir $BUILD_PATH
  mkdir $BUILD_OUT_PATH
  cd $BUILD_PATH
  export LLVM_SOURCE_DIR=$BUILD_PATH/llvm-project
  clone_llvm_source "${LLVM_SOURCE_DIR}"
  export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
  export PTO_SOURCE_DIR=$BASE_PATH
  export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

  cd $LLVM_SOURCE_DIR
  rm -rf "${LLVM_BUILD_DIR}"

  if [ -d "$CANN_3RD_LIB_PATH/llvm-19" ]; then
    configure_llvm_build -DLLVM_ENABLE_ZSTD=OFF
  else
    configure_llvm_build
  fi

  ninja -C $LLVM_BUILD_DIR

  cd $PTO_SOURCE_DIR
  export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

  if [ -d "$CANN_3RD_LIB_PATH/llvm-19" ]; then
    configure_ptoas_build ${CMAKE_ARGS}
  else
    configure_ptoas_build ${CMAKE_ARGS}
  fi

  ninja -C build
  ninja -C build install
  harden_package_artifacts
  cd $BUILD_PATH
  ninja package
}

main() {
  checkopts "$@"
  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ]; then
    build_only
  fi
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    package
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
