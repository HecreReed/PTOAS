# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(LLVM_LINK_COMPONENTS
  Support
  Core
)

add_llvm_executable(pto-opt
  ptoas.cpp
)

set_target_properties(pto-opt PROPERTIES OUTPUT_NAME "ptoas")
target_compile_definitions(pto-opt PRIVATE
  PTOAS_RELEASE_VERSION="${PTOAS_CLI_VERSION}"
)

target_link_libraries(pto-opt PRIVATE
  PTOIR
  PTOTransforms
  ptobc_lib
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRFuncTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIREmitCDialect
  MLIREmitCTransforms
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSCFTransforms
  MLIRGPUDialect
  MLIRTransforms
  MLIRTargetCpp
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRControlFlowDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)

if(APPLE)
  target_link_options(pto-opt PRIVATE "-Wl,-dead_strip_dylibs")
endif()

add_dependencies(pto-opt
  PTOOpsIncGen
)
