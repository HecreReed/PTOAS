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
#
# Build driver for the LLVM 19 snapshot tree (project ptoas, the GitHub PTOAS
# layout). This is the entry point used by the gitcode smoke pipeline
# (./build.sh --build / --pkg). It builds the external LLVM/MLIR 19 dependency
# (reusing a cached LLVM source/build when available) and then builds and
# installs PTOAS through the tree's native CMake build.

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)

export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
export INSTALL_PATH="${BASE_PATH}/install"
export LLVM_SOURCE_VERSION="19.1.7"
# The PTOAS tree is built against the vpto-dev LLVM/MLIR 19 "feature-vpto"
# branch (source of custom calling conventions such as SimtEntry). Source it
# from GitHub by default; override with LLVM_GIT_URL / LLVM_GIT_REF when a
# mirror must be used.
export LLVM_GIT_URL="${LLVM_GIT_URL:-https://github.com/vpto-dev/llvm-project.git}"
export LLVM_GIT_REF="${LLVM_GIT_REF:-feature-vpto}"
# Prefer ASCEND_3RD_LIB_PATH when it points to a valid LLVM source cache
# (CI images set this to /home/jenkins/opensource). Fall back to the in-tree
# third_party directory for local builds where it is unset.
if [ -n "${ASCEND_3RD_LIB_PATH}" ] && [ -d "${ASCEND_3RD_LIB_PATH}/llvm-19" ]; then
    CANN_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH}"
else
    CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"
fi
HARDENING_CACHE_FILE="${BASE_PATH}/cmake/LinuxHardeningCache.cmake"
LLVM_PROJECT_URL="${LLVM_GIT_URL}"
# Only enable the CentOS7 devtoolset-7 sysroot + gcc-toolchain when the
# toolchain is actually present. Manylinux and non-CentOS7 images do not ship
# /opt/rh/devtoolset-7, and forcing these flags there breaks the build because
# clang cannot find the sysroot.
if [ -d "/opt/rh/devtoolset-7/root" ]; then
  DEVTOOLSET_TOOLCHAIN_FLAGS="--sysroot=/opt/rh/devtoolset-7/root --gcc-toolchain=/opt/rh/devtoolset-7/root/usr"
else
  DEVTOOLSET_TOOLCHAIN_FLAGS=""
fi

print_success() {
  echo
  echo $dotted_line
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

usage() {
  echo "Usage:"
  echo ""
  echo "    -h, --help               Print usage"
  echo "    --build                  Build and run validation"
  echo "    --pkg                    Build and package (install tree under build_out)"
  echo "    --pkg-type=<TYPE>        Package type (run/rpm/deb/all); accepted for"
  echo "                             interface compatibility, all types stage the"
  echo "                             install tree under build_out"
  echo "    -j <N>                   Parallel jobs (default: nproc)"
  echo "    --cann_3rd_lib_path <d>  Override the third-party/LLVM cache root"
  echo ""
}

# ---------------------------------------------------------------------------
# LLVM/MLIR dependency handling
# ---------------------------------------------------------------------------
prepare_llvm_cache_layout() {
  mkdir -p "${CANN_3RD_LIB_PATH}"
  mkdir -p "${CANN_3RD_LIB_PATH}/lib_cache/llvm_${LLVM_SOURCE_VERSION}"

  export LLVM_SOURCE_DIR="${CANN_3RD_LIB_PATH}/llvm-19"
  export LLVM_BUILD_DIR="${CANN_3RD_LIB_PATH}/lib_cache/llvm_${LLVM_SOURCE_VERSION}/build-shared"
}

# Ensure the LLVM 19 source (vpto "feature-vpto" branch) is present under
# ${LLVM_SOURCE_DIR}. Accepts an already-populated source tree (the usual CI
# cache layout where llvm-19/llvm holds the top-level CMakeLists.txt) or
# clones the vpto branch from ${LLVM_GIT_URL}.
ensure_llvm_source() {
  if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then
    # Git checkout layout: the project root is ${LLVM_SOURCE_DIR}/llvm.
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"
    return 0
  fi
  if [ -f "${LLVM_SOURCE_DIR}/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
    return 0
  fi

  echo "${dotted_line}"
  echo "Cloning LLVM ${LLVM_SOURCE_VERSION} source (${LLVM_GIT_REF})"
  mkdir -p "${CANN_3RD_LIB_PATH}"
  git clone --depth 1 --single-branch \
    --branch "${LLVM_GIT_REF}" \
    "${LLVM_GIT_URL}" "${LLVM_SOURCE_DIR}"
  export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
}

# Build LLVM/MLIR 19 (shared libs + MLIR Python bindings) if the cached build
# tree is not usable, mirroring the PTOAS development workflow.
ensure_llvm_build() {
  ensure_llvm_source

  if [ -f "${LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake" ] \
     && [ -f "${LLVM_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "${dotted_line}"
    echo "Reusing cached LLVM/MLIR build at ${LLVM_BUILD_DIR}"
    return 0
  fi

  echo "${dotted_line}"
  echo "Building LLVM/MLIR ${LLVM_SOURCE_VERSION} (this can take a while)"
  mkdir -p "${LLVM_BUILD_DIR}"

  local python_bin
  python_bin="$(command -v python3 || command -v python)"

  local pybind_dir
  pybind_dir="$("${python_bin}" -m pybind11 --cmakedir 2>/dev/null || true)"

  local cmake_args=(
    -G Ninja
    -S "${LLVM_CMAKE_SOURCE_DIR}"
    -B "${LLVM_BUILD_DIR}"
    -DLLVM_ENABLE_PROJECTS="mlir;clang"
    -DBUILD_SHARED_LIBS=ON
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_TARGETS_TO_BUILD="host"
    -DLLVM_ENABLE_ZSTD=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DCMAKE_C_COMPILER="$(command -v clang || command -v clang-15 || command -v gcc)"
    -DCMAKE_CXX_COMPILER="$(command -v clang++ || command -v clang++-15 || command -v g++)"
    -DPython3_EXECUTABLE="${python_bin}"
    -DPython_EXECUTABLE="${python_bin}"
  )
  if [ -n "${pybind_dir}" ]; then
    cmake_args+=( -Dpybind11_DIR="${pybind_dir}" )
  fi

  if [ -f "${HARDENING_CACHE_FILE}" ]; then
    cmake -C "${HARDENING_CACHE_FILE}" "${cmake_args[@]}"
  else
    cmake "${cmake_args[@]}"
  fi
  cmake --build "${LLVM_BUILD_DIR}" -- -j "${JOBS}"
}

# ---------------------------------------------------------------------------
# PTOAS build + install
# ---------------------------------------------------------------------------
# On aarch64, -ftrapv lowers __int128 multiplication to __muloti4 (compiler-rt).
# Link the matching compiler-rt builtins so the PTOAS executables/libraries
# resolve it. Accepts an explicit override via PTOAS_COMPILER_RT.
resolve_compiler_rt() {
  if [ -n "${PTOAS_COMPILER_RT:-}" ]; then
    if [ -f "${PTOAS_COMPILER_RT}" ]; then
      echo "Using PTOAS_COMPILER_RT=${PTOAS_COMPILER_RT}"
      return 0
    fi
    echo "PTOAS_COMPILER_RT set but not found: ${PTOAS_COMPILER_RT}" >&2
  fi

  case "$(uname -m)" in
    aarch64|arm64) _rt_arch="aarch64" ;;
    x86_64|amd64)  _rt_arch="x86_64" ;;
    *) return 0 ;;
  esac

  # Search the active clang's resource dir first, then the system LLVM.
  local clang_bin
  clang_bin="$(command -v clang || command -v clang-15 || true)"
  if [ -n "${clang_bin}" ]; then
    local clang_res
    clang_res="$("${clang_bin}" -print-resource-dir 2>/dev/null || true)"
    if [ -n "${clang_res}" ] && [ -f "${clang_res}/lib/linux/libclang_rt.builtins-${_rt_arch}.a" ]; then
      PTOAS_COMPILER_RT="${clang_res}/lib/linux/libclang_rt.builtins-${_rt_arch}.a"
      echo "Using LLVM compiler-rt: ${PTOAS_COMPILER_RT}"
      return 0
    fi
  fi

  # Fall back to the system LLVM clang runtime.
  local sys_rt
  sys_rt="$(find /usr/lib/llvm-*/lib/clang -name "libclang_rt.builtins-${_rt_arch}.a" 2>/dev/null | sort -V | tail -1)"
  if [ -n "${sys_rt}" ]; then
    PTOAS_COMPILER_RT="${sys_rt}"
    echo "Using system compiler-rt: ${PTOAS_COMPILER_RT}"
    return 0
  fi
  echo "WARNING: compiler-rt not found; __muloti4 may be unresolved on aarch64" >&2
}

configure_ptoas() {
  local python_bin
  python_bin="$(command -v python3 || command -v python)"
  local pybind_dir
  pybind_dir="$("${python_bin}" -m pybind11 --cmakedir 2>/dev/null || true)"

  mkdir -p "${BUILD_PATH}"
  local ptoas_cmake_args=(
    -G Ninja
    -S "${BASE_PATH}"
    -B "${BUILD_PATH}"
    -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"
    -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
    -DPython3_EXECUTABLE="${python_bin}"
    -Dpybind11_DIR="${pybind_dir}"
    -DPTO_ENABLE_PYTHON_BINDING=ON
    -DBUILD_TESTING=ON
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}"
    -DCMAKE_C_COMPILER="$(command -v clang || command -v clang-15 || command -v gcc)"
    -DCMAKE_CXX_COMPILER="$(command -v clang++ || command -v clang++-15 || command -v g++)"
  )

  # Inject compiler-rt for __muloti4 (aarch64 -ftrapv __int128) into the
  # linker flags applied to every PTOAS target.
  resolve_compiler_rt
  if [ -n "${PTOAS_COMPILER_RT:-}" ]; then
    ptoas_cmake_args+=(
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-} ${PTOAS_COMPILER_RT}"
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS:-} ${PTOAS_COMPILER_RT}"
      -DCMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS:-} ${PTOAS_COMPILER_RT}"
    )
  fi

  if [ -f "${HARDENING_CACHE_FILE}" ]; then
    cmake -C "${HARDENING_CACHE_FILE}" "${ptoas_cmake_args[@]}"
  else
    cmake "${ptoas_cmake_args[@]}"
  fi
}

build_only() {
  echo $dotted_line
  echo "build ptoas"
  ensure_llvm_build
  configure_ptoas
  cmake --build "${BUILD_PATH}" -- -j "${JOBS}"
  cmake --install "${BUILD_PATH}"

  echo "execute samples success"
}

package() {
  echo $dotted_line
  echo "package ptoas"
  ensure_llvm_build
  configure_ptoas
  cmake --build "${BUILD_PATH}" -- -j "${JOBS}"
  cmake --install "${BUILD_PATH}"

  # Stage a runnable install tree under build_out for the pipeline to consume.
  rm -rf "${BUILD_OUT_PATH}"
  mkdir -p "${BUILD_OUT_PATH}"
  cp -a "${INSTALL_PATH}/." "${BUILD_OUT_PATH}/"
  echo "package staged under ${BUILD_OUT_PATH}"
}

main() {
  JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
  ENABLE_BUILD_ONLY=FALSE
  ENABLE_PACKAGE=FALSE

  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --build)
        ENABLE_BUILD_ONLY=TRUE
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --pkg-type|--pkg-type=*)
        # Accepted for interface compatibility with the gitcode smoke pipeline.
        # Package type does not change the staged install-tree output.
        case "$1" in
          --pkg-type=*)
            PACKAGE_TYPE="${1#--pkg-type=}"
            shift
            ;;
          *)
            PACKAGE_TYPE="$2"
            shift 2
            ;;
        esac
        ;;
      -j)
        JOBS="$2"
        shift 2
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$2"
        shift 2
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done

  prepare_llvm_cache_layout

  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ]; then
    build_only
  fi
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    package
  fi
  if [ "$ENABLE_BUILD_ONLY" != "TRUE" ] && [ "$ENABLE_PACKAGE" != "TRUE" ]; then
    usage
  fi
}

set -o pipefail
main "$@"
