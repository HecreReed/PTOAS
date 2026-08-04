# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ptobc lives in-tree inside PTOAS.
#
# NOTE: Do NOT call cmake_minimum_required()/project() or find_package() here.
# PTOAS top-level already configures LLVM/MLIR and exports PTOAS:: targets.

add_mlir_library(ptobc_lib
  OBJECT
  ENABLE_AGGREGATION
  EXCLUDE_FROM_LIBMLIR
  DISABLE_INSTALL

  src/leb128.cpp
  src/ptobc_format.cpp
  src/mlir_helpers.cpp
  src/mlir_encode.cpp
  src/canonical_printer.cpp
  src/ptobc_decode_print.cpp

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  PTOIR
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
)

target_include_directories(ptobc_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
)

# ptobc reports malformed input through C++ exceptions. add_mlir_library
# applies LLVM's default no-exceptions flags, so restore the existing ptobc
# compilation contract on the object target used by the aggregate.
if(MSVC)
  target_compile_options(obj.ptobc_lib PRIVATE /EHsc)
elseif(LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL)
  target_compile_options(obj.ptobc_lib PRIVATE -fexceptions)
endif()

add_executable(ptobc
  src/main.cpp
)

target_link_libraries(ptobc PRIVATE ptobc_lib)

# (Optional) install rules can be added once the layout stabilizes.
