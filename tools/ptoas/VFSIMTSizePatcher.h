// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_VFSIMT_SIZE_PATCHER_H
#define PTOAS_VFSIMT_SIZE_PATCHER_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace mlir::pto {

enum class VFSIMTSizeFixMode {
    Auto,
    Off,
    Verify,
};

struct VFSIMTSizePatchResult {
    bool changed = false;
    unsigned verifiedCallSites = 0;
    unsigned patchedCallSites = 0;
    std::string objectPath;
};

FailureOr<VFSIMTSizePatchResult> verifyAndPatchVFSIMTSize(
    llvm::Module& module, llvm::StringRef rawObjectPath, llvm::StringRef patchedObjectPath, VFSIMTSizeFixMode mode,
    llvm::raw_ostream& diagOS);

} // namespace mlir::pto

#endif
