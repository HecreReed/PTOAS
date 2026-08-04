// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VFSIMTSizePatcher.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

static std::optional<mlir::pto::VFSIMTSizeFixMode>
parseMode(llvm::StringRef value) {
  if (value == "auto")
    return mlir::pto::VFSIMTSizeFixMode::Auto;
  if (value == "off")
    return mlir::pto::VFSIMTSizeFixMode::Off;
  if (value == "verify")
    return mlir::pto::VFSIMTSizeFixMode::Verify;
  return std::nullopt;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    llvm::errs() << "usage: " << argv[0]
                 << " <module.ll> <raw.o> <patched.o> <auto|off|verify>\n";
    return 2;
  }
  std::optional<mlir::pto::VFSIMTSizeFixMode> mode = parseMode(argv[4]);
  if (!mode) {
    llvm::errs() << "unknown mode: " << argv[4] << "\n";
    return 2;
  }

  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(argv[1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }

  mlir::FailureOr<mlir::pto::VFSIMTSizePatchResult> result =
      mlir::pto::verifyAndPatchVFSIMTSize(*module, argv[2], argv[3], *mode,
                                          llvm::errs());
  if (mlir::failed(result))
    return 1;
  llvm::outs() << "changed=" << result->changed
               << " verified=" << result->verifiedCallSites
               << " patched=" << result->patchedCallSites
               << " object=" << result->objectPath << "\n";
  return 0;
}
