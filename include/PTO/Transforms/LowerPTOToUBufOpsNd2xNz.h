// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_LOWERPTOToUBUFOPSNDto2xNZ_H
#define PTO_TRANSFORMS_LOWERPTOToUBUFOPSNDto2xNZ_H

#include "PTO/IR/PTO.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::pto::detail {

struct PendingTExtractNd2xNz {
  pto::TExtractOp op;
  llvm::SmallVector<int64_t, 4> indices;
  int64_t elemBytes = 0;
  int64_t srcRowStrideElems = 0;
  int64_t srcPhysicalRows = 0;
  int64_t srcPhysicalCols = 0;
  llvm::SmallVector<int64_t, 4> dstPhysical;
  llvm::SmallVector<int64_t, 4> dstValid;
};

std::optional<PendingTExtractNd2xNz>
snapshotTExtractNd2xNz(pto::TExtractOp op);

std::optional<unsigned> selectUnusedHiddenEventId(Operation *anchor,
                                                   MLIRContext *ctx);

LogicalResult expandTExtractNd2xNz(OpBuilder &builder, MLIRContext *ctx,
                                   const PendingTExtractNd2xNz &pending,
                                   pto::EVENT eventId);

} // namespace mlir::pto::detail

#endif // PTO_TRANSFORMS_LOWERPTOToUBUFOPSNDto2xNZ_H
