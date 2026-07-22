// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Support/CANNVersion.h"

namespace mlir::pto {

static bool usesCANN900Lowering(const VPTOEmissionOptions &options) {
  const bool isC220 = options.march == "dav-c220-vec" ||
                      options.march == "dav-c220-cube";
  return !isC220 &&
         options.cannVersion >= CANNVersion::release(9, 0, 0);
}

LogicalResult lowerVPTOModuleToLLVMModules(
    ModuleOp module, const VPTOEmissionOptions &options,
    EmittedLLVMModule &cubeModule, EmittedLLVMModule &vectorModule,
    llvm::raw_ostream &diagOS) {
  if (usesCANN900Lowering(options))
    return lowerVPTOModuleToLLVMModulesCANN900(module, options, cubeModule,
                                               vectorModule, diagOS);
  return lowerVPTOModuleToLLVMModulesBeta1(module, options, cubeModule,
                                           vectorModule, diagOS);
}

LogicalResult lowerVPTOModuleToLLVMIRText(
    ModuleOp module, const VPTOEmissionOptions &options, std::string &output,
    llvm::raw_ostream &diagOS) {
  output.clear();

  EmittedLLVMModule cubeModule;
  EmittedLLVMModule vectorModule;
  if (failed(
          lowerVPTOModuleToLLVMModules(module, options, cubeModule, vectorModule,
                                       diagOS)))
    return failure();

  llvm::raw_string_ostream os(output);
  bool printedAny = false;
  if (vectorModule.module) {
    vectorModule.module->print(os, nullptr);
    os << "\n";
    printedAny = true;
  }
  if (cubeModule.module) {
    if (printedAny)
      os << "\n";
    cubeModule.module->print(os, nullptr);
    os << "\n";
  }
  os.flush();
  return success();
}


} // namespace mlir::pto
