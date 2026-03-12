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
  git clone https://gitcode.com/GitHub_Trending/ll/llvm-project.git -b llvmorg-19.1.7
  export LLVM_SOURCE_DIR=$WORKSPACE/llvm-project
  export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
  export PTO_SOURCE_DIR=$WORKSPACE
  export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

  cd $LLVM_SOURCE_DIR
  cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DBUILD_SHARED_LIBS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python3) \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_TARGETS_TO_BUILD="host"

  ninja -C $LLVM_BUILD_DIR

  cd $PTO_SOURCE_DIR
  export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
  cmake -G Ninja \
      -S . \
      -B build \
      -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
      -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
      -DPython3_EXECUTABLE=$(which python3) \
      -DPython3_FIND_STRATEGY=LOCATION \
      -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
      -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR"

  ninja -C build
  ninja -C build install

  export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
  export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
  export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
  export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
  export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH

  bash test/samples/runop.sh --enablebc all
  STAGE='${STAGE}' RUN_MODE='npu' SOC_VERSION='-Ascend910B' DEVICE_ID='0' SKIP_CASES='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print' bash test/npu_validation/scripts/run_remote_npu_validation.sh

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
  clean_build_out
  clean_build
  mkdir $BUILD_PATH
  mkdir $BUILD_OUT_PATH
  cd $BUILD_PATH

  git clone https://gitcode.com/GitHub_Trending/ll/llvm-project.git -b llvmorg-19.1.7
  export LLVM_SOURCE_DIR=$BUILD_PATH/llvm-project
  export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
  export PTO_SOURCE_DIR=$BASE_PATH
  export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

  cd $LLVM_SOURCE_DIR
  cmake -G Ninja -S llvm -B $LLVM_BUILD_DIR \
      -DLLVM_ENABLE_PROJECTS="mlir;clang" \
      -DBUILD_SHARED_LIBS=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python3) \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_TARGETS_TO_BUILD="host"

  ninja -C $LLVM_BUILD_DIR

  cd $PTO_SOURCE_DIR
  export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

  cmake -G Ninja \
      -S . \
      -B build \
      -DLLVM_DIR=$LLVM_BUILD_DIR/lib/cmake/llvm \
      -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
      -DPython3_EXECUTABLE=$(which python3) \
      -DPython3_FIND_STRATEGY=LOCATION \
      -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DMLIR_PYTHON_PACKAGE_DIR=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core \
      -DCMAKE_INSTALL_PREFIX="$PTO_INSTALL_DIR" \
      ${CMAKE_ARGS}

  ninja -C build
  ninja -C build install
  cd $BUILD_PATH
  make package
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
