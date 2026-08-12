// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOLowering.h - VPTO buffer materialization contracts ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOLOWERING_H_
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOLOWERING_H_

#include "PTO/IR/PTO.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

Value materializeBufferPointer(Value value, Type elementType,
                               Attribute memorySpace,
                               PatternRewriter &rewriter, Location loc);
LogicalResult convertVPTOEmissionBoundaryToPtr(
    ModuleOp module, llvm::raw_ostream *diagOS = nullptr);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOLOWERING_H_
