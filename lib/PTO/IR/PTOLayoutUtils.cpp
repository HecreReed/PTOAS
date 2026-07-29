// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/IR/PTOLayoutUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr int64_t kNZBlockBytes = 32;
constexpr int64_t kNZInnerRows = 16;

struct ShapeStride5D {
  SmallVector<int64_t, kPTOLayoutRank> shape;
  SmallVector<int64_t, kPTOLayoutRank> stride;
};

static bool isDynamic(int64_t value) { return value == ShapedType::kDynamic; }

static bool isKnownInvalid(int64_t value) {
  return value < 0 && !isDynamic(value);
}

static bool matchesKnown(int64_t actual, int64_t expected) {
  return isDynamic(actual) || actual == expected;
}

static std::optional<int64_t> checkedMultiply(int64_t lhs, int64_t rhs) {
  if (isDynamic(lhs) || isDynamic(rhs))
    return ShapedType::kDynamic;
  int64_t product = 0;
  if (llvm::MulOverflow(lhs, rhs, product))
    return std::nullopt;
  return product;
}

static std::optional<ShapeStride5D> rightAlignTo5D(ArrayRef<int64_t> shape,
                                                   ArrayRef<int64_t> stride) {
  if (shape.empty() || shape.size() != stride.size() ||
      shape.size() > kPTOLayoutRank)
    return std::nullopt;
  if (llvm::any_of(shape, isKnownInvalid) ||
      llvm::any_of(stride, isKnownInvalid))
    return std::nullopt;

  ShapeStride5D result;
  result.shape.assign(kPTOLayoutRank, 1);
  result.stride.assign(kPTOLayoutRank, 1);

  const int64_t shift = kPTOLayoutRank - shape.size();
  for (auto [index, value] : llvm::enumerate(shape))
    result.shape[shift + index] = value;
  for (auto [index, value] : llvm::enumerate(stride))
    result.stride[shift + index] = value;

  for (int64_t index = shift - 1; index >= 0; --index) {
    auto value =
        checkedMultiply(result.shape[index + 1], result.stride[index + 1]);
    if (!value)
      return std::nullopt;
    result.stride[index] = *value;
  }
  return result;
}

static bool allStatic(ArrayRef<int64_t> values) {
  return llvm::none_of(values, isDynamic);
}

static bool matchesNDMinor2D(int64_t rows, int64_t cols, int64_t rowStride,
                             int64_t colStride) {
  if (!isDynamic(cols) && cols != 1 && !isDynamic(colStride) && colStride != 1)
    return false;
  if (!isDynamic(rows) && rows == 1)
    return true;
  if (!isDynamic(cols) && cols == 1)
    return isDynamic(rowStride) || rowStride == 1;
  if (isDynamic(rowStride) || isDynamic(cols))
    return true;
  return rowStride == cols;
}

static bool matchesDNMinor2D(int64_t rows, int64_t cols, int64_t rowStride,
                             int64_t colStride) {
  if (!isDynamic(rows) && rows != 1 && !isDynamic(rowStride) && rowStride != 1)
    return false;
  if (!isDynamic(cols) && cols == 1)
    return true;
  if (!isDynamic(rows) && rows == 1)
    return isDynamic(colStride) || colStride == 1;
  if (isDynamic(colStride) || isDynamic(rows))
    return true;
  return colStride == rows;
}

static std::optional<Layout>
inferMinor2DLayout(int64_t rows, int64_t cols, int64_t rowStride,
                   int64_t colStride, std::optional<Layout> preferredMinor2D,
                   bool *isMinor2DAmbiguous) {
  const bool nd = matchesNDMinor2D(rows, cols, rowStride, colStride);
  const bool dn = matchesDNMinor2D(rows, cols, rowStride, colStride);
  if (!nd && !dn)
    return Layout::ND;
  if (nd && dn) {
    if (isMinor2DAmbiguous)
      *isMinor2DAmbiguous = true;
    if (preferredMinor2D &&
        (*preferredMinor2D == Layout::ND || *preferredMinor2D == Layout::DN))
      return *preferredMinor2D;
    return (!isDynamic(cols) && cols == 1 && (isDynamic(rows) || rows != 1))
               ? Layout::DN
               : Layout::ND;
  }
  return dn ? Layout::DN : Layout::ND;
}

static std::string expectedActual(StringRef name, int64_t expected,
                                  int64_t actual) {
  return ("expected " + name + "==" + Twine(expected) + " (got " +
          Twine(actual) + ")")
      .str();
}

} // namespace

std::optional<int64_t>
mlir::pto::getNZC0StorageElems(unsigned storageElemBytes) {
  if (storageElemBytes == 0 || kNZBlockBytes % storageElemBytes != 0)
    return std::nullopt;
  return kNZBlockBytes / storageElemBytes;
}

bool mlir::pto::hasNZInnerStructure5D(ArrayRef<int64_t> shape5D,
                                      ArrayRef<int64_t> stride5D,
                                      unsigned storageElemBytes) {
  if (shape5D.size() != kPTOLayoutRank || stride5D.size() != kPTOLayoutRank)
    return false;
  auto c0 = getNZC0StorageElems(storageElemBytes);
  if (!c0)
    return false;
  return matchesKnown(shape5D[3], kNZInnerRows) &&
         matchesKnown(shape5D[4], *c0) && matchesKnown(stride5D[4], 1) &&
         matchesKnown(stride5D[3], *c0) &&
         matchesKnown(stride5D[2], kNZInnerRows * *c0);
}

std::optional<std::string>
mlir::pto::getNZViewCompatibilityError(ArrayRef<int64_t> shape5D,
                                       ArrayRef<int64_t> stride5D,
                                       unsigned storageElemBytes) {
  if (shape5D.size() != kPTOLayoutRank || stride5D.size() != kPTOLayoutRank)
    return "NZ layout requires rank-5 shape and stride";

  auto c0 = getNZC0StorageElems(storageElemBytes);
  if (!c0)
    return ("NZ layout requires a non-zero storage element byte size that "
            "divides 32, got " +
            Twine(storageElemBytes))
        .str();

  if (!matchesKnown(shape5D[0], 1))
    return expectedActual("shape[0]", 1, shape5D[0]);
  if (!matchesKnown(shape5D[3], kNZInnerRows))
    return expectedActual("shape[3]", kNZInnerRows, shape5D[3]);
  if (!matchesKnown(shape5D[4], *c0))
    return expectedActual("shape[4]", *c0, shape5D[4]);
  if (!matchesKnown(stride5D[4], 1))
    return expectedActual("stride[4]", 1, stride5D[4]);
  if (!matchesKnown(stride5D[3], *c0))
    return expectedActual("stride[3]", *c0, stride5D[3]);
  if (!matchesKnown(stride5D[2], kNZInnerRows * *c0))
    return expectedActual("stride[2]", kNZInnerRows * *c0, stride5D[2]);

  if (!isDynamic(stride5D[1])) {
    if (stride5D[1] % *c0 != 0)
      return ("expected stride[1] to be C0-aligned (" + Twine(*c0) + "), got " +
              Twine(stride5D[1]))
          .str();
    auto minimum = checkedMultiply(shape5D[2], stride5D[2]);
    if (!minimum)
      return "shape[2] * stride[2] overflows";
    if (!isDynamic(*minimum) && stride5D[1] < *minimum)
      return ("expected stride[1] >= shape[2] * stride[2] (" + Twine(*minimum) +
              "), got " + Twine(stride5D[1]))
          .str();
  }

  if (!isDynamic(stride5D[0])) {
    if (stride5D[0] % *c0 != 0)
      return ("expected stride[0] to be C0-aligned (" + Twine(*c0) + "), got " +
              Twine(stride5D[0]))
          .str();
    auto minimum = checkedMultiply(shape5D[1], stride5D[1]);
    if (!minimum)
      return "shape[1] * stride[1] overflows";
    if (!isDynamic(*minimum) && stride5D[0] < *minimum)
      return ("expected stride[0] >= shape[1] * stride[1] (" + Twine(*minimum) +
              "), got " + Twine(stride5D[0]))
          .str();
  }

  return std::nullopt;
}

bool mlir::pto::isNZViewCompatible5D(ArrayRef<int64_t> shape5D,
                                     ArrayRef<int64_t> stride5D,
                                     unsigned storageElemBytes) {
  return !getNZViewCompatibilityError(shape5D, stride5D, storageElemBytes);
}

std::optional<std::string>
mlir::pto::getNZSubviewCompatibilityError(ArrayRef<int64_t> sourceShape5D,
                                          ArrayRef<int64_t> offsets5D,
                                          ArrayRef<int64_t> sizes5D) {
  if (sourceShape5D.size() != kPTOLayoutRank ||
      offsets5D.size() != kPTOLayoutRank || sizes5D.size() != kPTOLayoutRank)
    return "NZ partition requires rank-5 source, offsets, and sizes";

  for (int64_t index : {0, 3, 4}) {
    int64_t expectedSize = sourceShape5D[index];
    if (isDynamic(offsets5D[index]) || isDynamic(sizes5D[index]) ||
        isDynamic(expectedSize))
      return ("NZ partition must prove that d" + Twine(index) +
              " remains complete at compile time")
          .str();
    if (offsets5D[index] != 0 || sizes5D[index] != expectedSize)
      return ("NZ view cannot be partitioned inside a fractal: d" +
              Twine(index) + " must keep offset=0 and size=" +
              Twine(expectedSize) + " (got offset=" + Twine(offsets5D[index]) +
              ", size=" + Twine(sizes5D[index]) + ")")
          .str();
  }
  return std::nullopt;
}

bool mlir::pto::isCanonicalNZRoot5D(ArrayRef<int64_t> shape5D,
                                    ArrayRef<int64_t> stride5D,
                                    unsigned storageElemBytes) {
  if (!allStatic(shape5D) || !allStatic(stride5D) ||
      !isNZViewCompatible5D(shape5D, stride5D, storageElemBytes))
    return false;
  auto stride1 = checkedMultiply(shape5D[2], stride5D[2]);
  auto stride0 = checkedMultiply(shape5D[1], stride5D[1]);
  return stride1 && stride0 && stride5D[1] == *stride1 &&
         stride5D[0] == *stride0;
}

bool mlir::pto::isLayoutCompatible5D(Layout layout, ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> stride,
                                     unsigned storageElemBytes) {
  auto padded = rightAlignTo5D(shape, stride);
  if (!padded)
    return false;
  switch (layout) {
  case Layout::NZ:
    if (shape.size() != kPTOLayoutRank)
      return false;
    return isNZViewCompatible5D(padded->shape, padded->stride,
                                storageElemBytes);
  case Layout::ND:
  case Layout::DN:
    // ND/DN are logical GlobalTensor interpretations, not compactness
    // promises. Existing inputs legitimately use interleaved and padded
    // outer views whose minor strides do not satisfy a dense 2-D recurrence.
    // Their explicit layout remains authoritative.
    return true;
  case Layout::MX_A_ZZ:
  case Layout::MX_B_NN:
    return true;
  }
  return false;
}

std::optional<Layout>
mlir::pto::inferLayout5D(ArrayRef<int64_t> shape, ArrayRef<int64_t> stride,
                         unsigned storageElemBytes,
                         std::optional<Layout> preferredMinor2D,
                         bool *isMinor2DAmbiguous) {
  if (isMinor2DAmbiguous)
    *isMinor2DAmbiguous = false;
  if (storageElemBytes == 0)
    return std::nullopt;
  auto padded = rightAlignTo5D(shape, stride);
  if (!padded || !allStatic(padded->shape) || !allStatic(padded->stride))
    return std::nullopt;

  if (shape.size() == kPTOLayoutRank &&
      isCanonicalNZRoot5D(padded->shape, padded->stride, storageElemBytes) &&
      padded->shape[0] == 1 && (padded->shape[1] > 1 || padded->shape[2] > 1))
    return Layout::NZ;

  return inferMinor2DLayout(padded->shape[3], padded->shape[4],
                            padded->stride[3], padded->stride[4],
                            preferredMinor2D, isMinor2DAmbiguous);
}
