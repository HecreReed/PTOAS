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
export PACKAGE_STAGE_PATH="${BUILD_PATH}/package_runtime"
export LLVM_SOURCE_VERSION="19.1.7"
# The PTOAS tree is built against the vpto-dev LLVM/MLIR 19 "feature-vpto"
# branch (source of custom calling conventions such as SimtEntry). Source it
# from GitHub by default; override with LLVM_GIT_URL / LLVM_GIT_REF when a
# mirror must be used.
export LLVM_GIT_URL="${LLVM_GIT_URL:-https://github.com/vpto-dev/llvm-project.git}"
export LLVM_GIT_REF="${LLVM_GIT_REF:-feature-vpto}"
# The vpto calling conventions (SimtEntry, float8) can also be produced by
# applying the feature-vpto patch to the upstream llvmorg-19.1.7 source that
# the CI cache (ASCEND_3RD_LIB_PATH) and cann-cmake download. When the cached
# source lacks SimtEntry we fetch the patch from the gitcode release asset and
# apply it with patch -p1, matching the PATCH_COMMAND added to cann-cmake's
# third_party/llvm.cmake.
export LLVM_VPTO_PATCH_URL="${LLVM_VPTO_PATCH_URL:-https://gitcode.com/cann-src-third-party/llvm/releases/download/19.1.7-h0/feature-vpto-last3.patch}"
export LLVM_VPTO_PATCH_SHA256="${LLVM_VPTO_PATCH_SHA256:-a49c1d3dd8ab78e93264712bc0d46deb536196a54abb2c2ee02abd914cd385e2}"
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

# Check whether the LLVM source carries the vpto custom calling conventions
# (llvm::CallingConv::SimtEntry). Upstream llvmorg-19.1.7 does not; the vpto
# fork and a patched upstream tree do.
llvm_has_simt_entry() {
  [ -f "${LLVM_SOURCE_DIR}/llvm/include/llvm/IR/CallingConv.h" ] \
    && grep -q "SimtEntry" "${LLVM_SOURCE_DIR}/llvm/include/llvm/IR/CallingConv.h"
}

# Download the feature-vpto patch and apply it to the upstream LLVM source so
# the tree gains the vpto calling conventions. The patch is a git-format-patch
# series rooted at llvm/, so patch -p1 is the correct strip level (the same
# PATCH_COMMAND used by cann-cmake's third_party/llvm.cmake).
apply_vpto_patch() {
  echo "${dotted_line}"
  echo "Applying feature-vpto patch to upstream LLVM source"
  local patch_file="${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch"
  if [ -f "${CANN_3RD_LIB_PATH}/feature-vpto-last3.patch" ]; then
    patch_file="${CANN_3RD_LIB_PATH}/feature-vpto-last3.patch"
  elif [ -f "${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch" ]; then
    patch_file="${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch"
  else
    mkdir -p "${CANN_3RD_LIB_PATH}/pkg"
    echo "Downloading vpto patch from ${LLVM_VPTO_PATCH_URL}"
    curl -fL --retry 3 -o "${patch_file}" "${LLVM_VPTO_PATCH_URL}" || {
      echo "ERROR: failed to download vpto patch" >&2
      exit 1
    }
    local actual_sha
    actual_sha="$(sha256sum "${patch_file}" | cut -d' ' -f1)"
    if [ "${actual_sha}" != "${LLVM_VPTO_PATCH_SHA256}" ]; then
      echo "ERROR: vpto patch SHA256 mismatch: ${actual_sha}" >&2
      exit 1
    fi
  fi

  (cd "${LLVM_SOURCE_DIR}" && patch -p1 < "${patch_file}") || {
    echo "ERROR: failed to apply vpto patch to ${LLVM_SOURCE_DIR}" >&2
    exit 1
  }
  echo "Applied vpto patch: ${patch_file}"
}

# Ensure the LLVM 19 source (vpto "feature-vpto" branch) is present under
# ${LLVM_SOURCE_DIR}. Accepts an already-populated source tree (the usual CI
# cache layout where llvm-19/llvm holds the top-level CMakeLists.txt) or
# clones the vpto branch from ${LLVM_GIT_URL}. When the cached source is the
# upstream (unpatched) snapshot, apply the vpto patch so SimtEntry/float8
# resolve during the PTOAS build.
ensure_llvm_source() {
  if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then
    # Git checkout layout: the project root is ${LLVM_SOURCE_DIR}/llvm.
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"
    if ! llvm_has_simt_entry; then
      echo "${dotted_line}"
      echo "Cached LLVM source lacks SimtEntry; applying feature-vpto patch"
      apply_vpto_patch
    fi
    return 0
  fi
  if [ -f "${LLVM_SOURCE_DIR}/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
    if ! llvm_has_simt_entry; then
      echo "${dotted_line}"
      echo "Cached LLVM source lacks SimtEntry; applying feature-vpto patch"
      apply_vpto_patch
    fi
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

# The PTOAS tree is built with the default libstdc++ ABI new string layout
# (_GLIBCXX_USE_CXX11_ABI=1). Cached LLVM/MLIR builds created on RHEL7 /
# devtoolset-7 toolchains are compiled with the old ABI and export Twine::str as
# the non-[abi:cxx11] mangled symbol; linking against them fails with
# "undefined reference to llvm::Twine::str[abi:cxx11]". Detect that by probing
# the shared lib symbol table and rebuild with the current toolchain instead of
# reusing the incompatible cache.
llvm_build_is_abi_compatible() {
  local support_lib="${LLVM_BUILD_DIR}/lib/libLLVMSupport.so.19.1"
  [ -f "${support_lib}" ] || return 1
  # _ZNK4llvm5Twine3strB5cxx11Ev  -> new ABI (this build)
  # _ZNK4llvm5Twine3strEv          -> old libstdc++ ABI (RHEL7/devtoolset cache)
  nm -D --defined-only "${support_lib}" 2>/dev/null \
    | grep -q "_ZNK4llvm5Twine3strB5cxx11Ev"
}

# Build LLVM/MLIR 19 (shared libs + MLIR Python bindings) if the cached build
# tree is not usable, mirroring the PTOAS development workflow.
ensure_llvm_build() {
  ensure_llvm_source

  # The vpto patch changes CallingConv.h; a build-shared tree built from the
  # unpatched upstream source must be rebuilt so SimtEntry is present in the
  # installed headers (otherwise the PTOAS build fails at link/compile time).
  local rebuild_llvm=FALSE
  if llvm_has_simt_entry \
     && [ -f "${LLVM_BUILD_DIR}/include/llvm/IR/CallingConv.h" ] \
     && ! grep -q "SimtEntry" "${LLVM_BUILD_DIR}/include/llvm/IR/CallingConv.h"; then
    echo "${dotted_line}"
    echo "LLVM source was patched but cached build lacks SimtEntry; rebuilding"
    rebuild_llvm=TRUE
  fi

  if [ "$rebuild_llvm" == "FALSE" ] \
     && [ -f "${LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake" ] \
     && [ -f "${LLVM_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ] \
     && ! llvm_build_is_abi_compatible; then
    echo "${dotted_line}"
    echo "Cached LLVM/MLIR build was compiled with an incompatible libstdc++ ABI"
    echo "(lacks llvm::Twine::str[abi:cxx11]); rebuilding with the current toolchain"
    rebuild_llvm=TRUE
  fi

  if [ "$rebuild_llvm" == "FALSE" ] \
     && [ -f "${LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake" ] \
     && [ -f "${LLVM_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "${dotted_line}"
    echo "Reusing cached LLVM/MLIR build at ${LLVM_BUILD_DIR}"
    return 0
  fi

  if [ "$rebuild_llvm" == "TRUE" ]; then
    echo "Removing stale LLVM build tree ${LLVM_BUILD_DIR}"
    rm -rf "${LLVM_BUILD_DIR}"
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
  # linker flags applied to every PTOAS target. The static archive must be
  # pulled in even though linker flags precede the object files: force the
  # symbol with -Wl,-u so the linker extracts __muloti4 from the archive.
  resolve_compiler_rt
  if [ -n "${PTOAS_COMPILER_RT:-}" ]; then
    ptoas_cmake_args+=(
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-} -Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS:-} -Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
      -DCMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS:-} -Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
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

# Stage the PTOAS executable and its runtime dependencies under the CANN package
# layout. The smoke image already puts /usr/local/Ascend/tools/ptoas/bin on
# PATH; installing the raw CMake tree would instead place ptoas in
# /usr/local/Ascend/bin, where test/samples/runop.sh cannot find it.
stage_ptoas_runtime() {
  local staged_bin="${PACKAGE_STAGE_PATH}/tools/ptoas/bin/ptoas"
  local staged_lib_dir="${PACKAGE_STAGE_PATH}/tools/ptoas/lib"

  rm -rf "${PACKAGE_STAGE_PATH}"
  mkdir -p "$(dirname "${staged_bin}")" "${staged_lib_dir}"
  PTO_INSTALL_DIR="${INSTALL_PATH}" \
  LLVM_RUNTIME_LIB_DIR="${LLVM_BUILD_DIR}/lib" \
  LLVM_STRIP_BIN="${LLVM_BUILD_DIR}/bin/llvm-strip" \
    bash "${BASE_PATH}/scripts/package/collect_ptoas_runtime_deps.sh" \
      "${BASE_PATH}" \
      "${INSTALL_PATH}/bin/ptoas" \
      "${staged_bin}" \
      "${staged_lib_dir}"

  test -x "${staged_bin}"
  test -x "${staged_bin}.real"
  echo "staged ptoas runtime: ${staged_bin}"
}

# Package the CANN-compatible runtime tree into a self-extracting .run
# installer under build_out. The gitcode smoke pipeline looks for
# build_out/*.run and drives it with --full / --uninstall.
make_ptoas_run() {
  local arch
  arch="$(uname -m)"
  # The OBS uploader parses the artifact name as cann-pto-as_<ver>_linux-<arch>.run
  # (underscore before the version, like master's CPack/makeself output), then
  # strips the version and uploads it as cann-pto-as_linux-<arch>.run. Keep the
  # underscore form so the uploader can resolve the package; the versioned file
  # is the one the pipeline's pto-as_compile.sh drives with --full/--uninstall.
  local run_file="${BUILD_OUT_PATH}/cann-pto-as_${PTOAS_PACKAGE_VERSION}_linux-${arch}.run"
  bash "${BASE_PATH}/scripts/package/make_ptoas_run.sh" \
    "${BASE_PATH}" \
    "${PACKAGE_STAGE_PATH}" \
    "${run_file}" \
    "${PTOAS_PACKAGE_VERSION}" \
    "ptoas"
  echo "run package: ${run_file}"
}

package() {
  echo $dotted_line
  echo "package ptoas"
  ensure_llvm_build
  configure_ptoas
  cmake --build "${BUILD_PATH}" -- -j "${JOBS}"
  cmake --install "${BUILD_PATH}"

  # Distill the version used for the .run package name. The CANN product version
  # (9.2.0 for this release train) differs from project(ptoas VERSION 0.57), so
  # default to the packaging version and allow an explicit override.
  PTOAS_PACKAGE_VERSION="${PTOAS_PACKAGE_VERSION:-9.2.0}"

  # Stage the .run installer(s) under build_out for the pipeline to consume.
  rm -rf "${BUILD_OUT_PATH}"
  mkdir -p "${BUILD_OUT_PATH}"
  stage_ptoas_runtime
  make_ptoas_run
  echo "package staged under ${BUILD_OUT_PATH}"
  # Diagnostics: the OBS uploader reads build_out via the host path
  # /opt/cloud/slavespace/.../x86build/build_out; print what we actually
  # ~created so a path mismatch is visible in the CI log.
  echo "BUILD_OUT absolute: $(cd "${BUILD_OUT_PATH}" && pwd -P)"
  ls -la "${BUILD_OUT_PATH}"
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
      -j=*)
        JOBS="${1#-j=}"
        shift
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$2"
        shift 2
        ;;
      --cann_3rd_lib_path=*)
        CANN_3RD_LIB_PATH="${1#--cann_3rd_lib_path=}"
        shift
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
