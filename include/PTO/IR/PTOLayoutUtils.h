// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
//===- PTOLayoutUtils.h - Shared PTO layout inference helpers ---*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOLAYOUTUTILS_H_
#define MLIR_DIALECT_PTO_IR_PTOLAYOUTUTILS_H_

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir::pto {

bool isNZLayout(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
                unsigned elemBytes);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_IR_PTOLAYOUTUTILS_H_
