// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_PTOTYPEUTILS_H
#define PTO_IR_PTOTYPEUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

namespace detail {
template <typename MemRefT>
inline auto getPTOMemRefStridesAndOffsetImpl(
    MemRefT memTy, SmallVectorImpl<int64_t> &strides, int64_t &offset, int)
    -> decltype(memTy.getStridesAndOffset(strides, offset)) {
  return memTy.getStridesAndOffset(strides, offset);
}

template <typename MemRefT>
inline LogicalResult getPTOMemRefStridesAndOffsetImpl(
    MemRefT memTy, SmallVectorImpl<int64_t> &strides, int64_t &offset, long) {
  return getStridesAndOffset(memTy, strides, offset);
}
} // namespace detail

inline LogicalResult getPTOMemRefStridesAndOffset(
    MemRefType memTy, SmallVectorImpl<int64_t> &strides, int64_t &offset) {
  return detail::getPTOMemRefStridesAndOffsetImpl(memTy, strides, offset, 0);
}

bool isPTOFloat8Type(Type t);
bool isPTOFloat8E4M3LikeType(Type t);
bool isPTOFloat8E5M2LikeType(Type t);
bool isPTOHiFloat8Type(Type t);
bool isPTOF8E8M0Type(Type t);
bool isPTOHiFloat8x2Type(Type t);
bool isPTOBF16x2Type(Type t);
bool isPTOFloat4PackedType(Type t);
bool isPTOPackedLdgStgVectorType(Type t);
bool isPTOLowPrecisionType(Type t);

unsigned getPTOStorageElemBitWidth(Type t);
unsigned getPTOStorageElemByteSize(Type t);
unsigned getPTOPackedLdgStgTotalBits(Type t);

// Shared physical-storage sizing for tile buffers (design doc 5.4/12): the
// authoritative allocation (reservation) size used by the planners and the
// post-planning ND-to-2xNz checks. Handles the plain rectangular footprint as
// well as the RowPlusOne compact layout. Returns nullopt when the type is not
// a tile buf or the footprint cannot be computed statically.
std::optional<int64_t> getTileBufStorageByteSize(Type tileBufType);

// The byte offset of the end of the last accessible element (access
// envelope). For plain layouts this equals the allocation size; for
// RowPlusOne compact layouts the trailing per-row gap after the last row is
// not accessed, so the envelope is narrower than the reservation
// (e.g. ColMajor NZ f16 16x32: reservation 1088 B, envelope 1056 B). Alias /
// liveness consumers (InsertSync and post-planning ranges) must use this;
// reservations use getTileBufStorageByteSize.
std::optional<int64_t> getTileBufAccessEndByteSize(Type tileBufType);

// Fold a value to a compile-time integer when it is a constant or a chain of
// constant casts / pure constant addi/subi (the ND-to-2xNZ verifier, the
// post-planning range resolver and the A2/A3 expansion snapshot must agree on
// which expressions are foldable; design doc 5.1/3.2).
std::optional<int64_t> getPTOConstantIntLike(Value value);

} // namespace mlir::pto

#endif // PTO_IR_PTOTYPEUTILS_H
