// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
//===- PTOLayoutUtils.cpp - Shared PTO layout inference helpers -----------===//
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOLayoutUtils.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

using llvm::ArrayRef;
using llvm::SmallVector;

namespace mlir::pto {
namespace {

struct ShapeStride5D {
  SmallVector<int64_t, 5> shape;
  SmallVector<int64_t, 5> stride;
};

std::optional<ShapeStride5D> rightAlignTo5D(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> stride) {
  if (shape.size() != stride.size())
    return std::nullopt;
  if (shape.empty() || shape.size() > 5)
    return std::nullopt;

  ShapeStride5D out;
  out.shape.assign(5, 1);
  out.stride.assign(5, 1);

  const int rank = static_cast<int>(shape.size());
  const int shift = 5 - rank;
  for (int i = 0; i < rank; ++i) {
    out.shape[shift + i] = shape[i];
    out.stride[shift + i] = stride[i];
  }

  for (int i = shift - 1; i >= 0; --i)
    out.stride[i] = out.shape[i + 1] * out.stride[i + 1];

  return out;
}

bool isCanonical2DNZLayout5D(ArrayRef<int64_t> shape5D,
                             ArrayRef<int64_t> stride5D,
                             unsigned elemBytes) {
  if (shape5D.size() != 5 || stride5D.size() != 5 || elemBytes == 0 ||
      32 % elemBytes != 0)
    return false;

  const int64_t c0 = 32 / elemBytes;
  if (shape5D[0] != 1)
    return false;
  // The degenerate [1, 1, 1, 16, c0] form is ambiguous with a plain 2D
  // row-major tensor of shape (16 x c0). Keep that case on the existing ND/DN
  // path unless the user specifies NZ explicitly.
  if (shape5D[1] == 1 && shape5D[2] == 1)
    return false;
  if (shape5D[3] != 16 || shape5D[4] != c0)
    return false;
  if (stride5D[4] != 1 || stride5D[3] != shape5D[4])
    return false;

  for (int i = 2; i >= 0; --i) {
    if (stride5D[i] != shape5D[i + 1] * stride5D[i + 1])
      return false;
  }

  return true;
}

bool isLegacyNZLayout5D(ArrayRef<int64_t> shape5D, ArrayRef<int64_t> stride5D,
                        unsigned elemBytes) {
  if (shape5D.size() != 5 || stride5D.size() != 5 || elemBytes == 0)
    return false;

  int64_t sh3 = shape5D[2], sh4 = shape5D[3], sh5 = shape5D[4];
  int64_t st4 = stride5D[3], st5 = stride5D[4];
  bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
  bool strideMatch = (st5 == 1) && (st4 == sh5);
  return alignMatch && strideMatch;
}

} // namespace

bool isNZLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
                unsigned elemBytes) {
  auto padded = rightAlignTo5D(shape, strides);
  if (!padded)
    return false;
  return isCanonical2DNZLayout5D(padded->shape, padded->stride, elemBytes) ||
         isLegacyNZLayout5D(padded->shape, padded->stride, elemBytes);
}

} // namespace mlir::pto
