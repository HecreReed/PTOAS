// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTO.cpp to keep source file nbnc under the codecheck threshold.
// This file is included by lib/PTO/IR/PTO.cpp and intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

LogicalResult pto::TAbsOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy))
    elemTy = tb.getElementType();
  else if (auto mr = dyn_cast<MemRefType>(srcTy))
    elemTy = mr.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<MemRefType, RankedTensorType,
                pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType, MemRefType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto mr = mlir::dyn_cast<MemRefType>(ty)) return mr.getElementType();
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) return tt.getElementType();
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) return tv.getElementType();
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) return tb.getElementType();
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) return tv.getElementType();
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto mr = mlir::dyn_cast<MemRefType>(ty))
    return SmallVector<int64_t,4>(mr.getShape().begin(), mr.getShape().end());
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty))
    return SmallVector<int64_t,4>(tt.getShape().begin(), tt.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty))
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t,4>(tb.getShape().begin(), tb.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty))
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy))
    return rawDim;
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2)
    return dims;

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i)
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  return dims;
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value)
    return {};
  auto valid = getValidShapeVec(value.getType());
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (valid.size() >= 1 && bind.getValidRow())
      valid[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (valid.size() >= 2 && bind.getValidCol())
      valid[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size())
    return shape;

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic)
      shape[i] = valid[i];
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static LogicalResult verifyAsyncFlatContiguous1DGMMemRef(Operation *op,
                                                         Value value,
                                                         StringRef name) {
  auto memTy = dyn_cast<MemRefType>(value.getType());
  if (!memTy)
    return op->emitOpError() << "expects " << name << " to be a memref";
  if (!memTy.hasRank())
    return op->emitOpError() << "expects " << name << " to be a ranked memref";
  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError() << "expects " << name
                             << " to be in GM address space";

  ArrayRef<int64_t> shape = memTy.getShape();
  if (shape.empty())
    return op->emitOpError() << "expects " << name
                             << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memTy, strides, offset)))
    return op->emitOpError() << "expects " << name
                             << " to be a strided memref with a known layout";

  bool hasDynamicLayout =
      offset == ShapedType::kDynamic ||
      llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      });
  if (hasDynamicLayout)
    return success();

  bool packed = !strides.empty() && strides.back() == 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0 && packed; --i)
    packed &= strides[i] == strides[i + 1] * shape[i + 1];
  if (!packed)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D &= shape[i] == 1;
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  return success();
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  Type ty = value.getType();
  if (isa<MemRefType>(ty))
    return verifyAsyncFlatContiguous1DGMMemRef(op, value, name);

  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a memref/tensor_view/partition_view";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D &= shape[i] == 1;
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";

  return success();
}

static bool isCommGlobalLikeType(Type ty) {
  if (auto memTy = dyn_cast<MemRefType>(ty))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty);
}

static LogicalResult verifyPositiveStaticShape(Operation *op, Type ty,
                                               StringRef name) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
  }
  return success();
}

static LogicalResult verifyCommGlobalLike(Operation *op, Value value,
                                          StringRef name) {
  Type ty = value.getType();
  if (!isCommGlobalLikeType(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";
  return verifyPositiveStaticShape(op, ty, name);
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name)))
    return failure();
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(32))
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a tile_buf or memref tile";
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  return verifyPositiveStaticShape(op, ty, name);
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty())
    return op->emitOpError() << "expects at least one " << name << " operand";
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str())))
      return failure();
    if (it.value().getType() != groupTy)
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong)
    return success();
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName)))
    return failure();
  if (ping.getType() != pong.getType())
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return std::nullopt;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0)
    return std::nullopt;

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace()))
      return as.getAddressSpace();
    return std::nullopt;
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(mr.getMemorySpace()))
      return as.getAddressSpace();
    if (!mr.getMemorySpace())
      return pto::AddressSpace::GM;
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (allowBf16 && ty.isBF16())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != 32)
    return false;
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true))
    return true;
  if (!isTargetArchA5(op))
    return false;
  return ty.isFloat8E4M3() || ty.isFloat8E4M3FN() || ty.isFloat8E4M3FNUZ() ||
         ty.isFloat8E4M3B11FNUZ() || ty.isFloat8E5M2() || ty.isFloat8E5M2FNUZ();
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy)
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";

  if (isa<pto::PartitionTensorViewType>(memTy)) {
    if (auto layout = getLogicalViewLayout(memValue)) {
      if (*layout != pto::Layout::ND)
        return op->emitOpError(
            "expects mem partition view to use ND logical layout when layout "
            "can be inferred");
    }
    return success();
  }

  if (auto mr = dyn_cast<MemRefType>(memTy)) {
    auto as = getPTOMemorySpaceEnum(mr);
    if (!as || (*as != pto::AddressSpace::GM &&
                 *as != pto::AddressSpace::Zero))
      return op->emitOpError(
          "expects mem memref to use GM or zero address space");
    if (mr.getRank() == 5) {
      auto shape = mr.getShape();
      bool allStatic = true;
      for (int64_t d : shape)
        if (d == ShapedType::kDynamic)
          allStatic = false;
      if (allStatic && (shape[0] != 1 || shape[1] != 1 || shape[2] != 1))
        return op->emitOpError(
            "expects rank-5 GM memref leading dimensions to be [1,1,1,...] "
            "(GlobalTensor table shape)");
    }
    return success();
  }

  return op->emitOpError(
      "expects mem to be !pto.partition_tensor_view or a GM/ZERO memref");
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != 2 || idxValid.size() != 2)
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile)
    return op->emitOpError("expects idx to be a tile_buf type");

  const bool idxRowMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool idxColMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::ColMajor);

  const bool rowCoalesce1xR =
      idxRowMajor && isKnownUnitExtent(idxValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[0]);
  const bool rowCoalesceRx1 =
      idxColMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownUnitExtent(idxValid[1]);
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);

  if (!(rowCoalesce1xR || rowCoalesceRx1 || elemCoalesce))
    return op->emitOpError()
           << "expects idx valid_shape to be [1, " << dataName
           << ".valid_row], [" << dataName
           << ".valid_row, 1], or match " << dataName << " valid_shape";

  return success();
}

static LogicalResult verifyMGatherMScatterIdxTile(Operation *op, Type ty,
                                                  StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in the vec address space";
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb)
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  if (!srcElem.isF32())
    return false;
  return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         dstElem.isFloat8E4M3() || dstElem.isFloat8E4M3FN() ||
         dstElem.isFloat8E4M3FNUZ() || dstElem.isFloat8E4M3B11FNUZ();
}

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  if (srcElem.isF16())
    return isPTOHiFloat8Type(dstElem);
  if (srcElem.isBF16())
    return isPTOFloat4PackedType(dstElem);
  if (isPTOFloat4PackedType(srcElem))
    return dstElem.isBF16();
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem))
    return dstElem.isF32();
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  return true;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != 2)
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
    Type elemTy = tb.getElementType();
    if (!allowLowPrecision && isPTOLowPrecisionType(elemTy))
      return op->emitOpError() << name << ": dtype " << elemTy
                               << " is not supported by this op yet";
  } else if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (mr.getRank() != 2)
      return op->emitOpError() << "expects " << name << " to be a rank-2 memref";
    if (!allowLowPrecision && isPTOLowPrecisionType(mr.getElementType()))
      return op->emitOpError() << name << ": dtype " << mr.getElementType()
                               << " is not supported by this op yet";
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf or rank-2 memref";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != 2)
    return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic &&
        validShape[i] > shape[i])
      return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i
                               << "] <= shape[" << i << "]";
  }
  return success();
}

static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf or memref";
  if (getElemTy(lhs) != getElemTy(rhs))
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i])
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
  }
  if (lhsValid.size() != rhsValid.size())
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  return success();
}

static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs))
    return success();

  auto lhsExtent = getLogicalTileExtentVec(lhs, compareValidShape);
  auto rhsExtent = getLogicalTileExtentVec(rhs, compareValidShape);
  auto emitMismatch = [&]() -> LogicalResult {
    if (compareValidShape)
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have compatible shapes";
  };
  if (lhsExtent.size() != rhsExtent.size())
    return emitMismatch();

  for (size_t i = 0, e = lhsExtent.size(); i < e; ++i) {
    if (lhsExtent[i] != ShapedType::kDynamic &&
        rhsExtent[i] != ShapedType::kDynamic && lhsExtent[i] != rhsExtent[i])
      return emitMismatch();
  }
  return success();
}

static LogicalResult verifyScaleTileMatchesOperand(Operation *op, Type scaleTy,
                                                   Type operandTy,
                                                   StringRef scaleName,
                                                   StringRef operandName) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";

  auto scaleShape = getShapeVec(scaleTy);
  auto operandShape = getShapeVec(operandTy);
  if (scaleShape.size() != operandShape.size())
    return op->emitOpError() << "expects " << scaleName << " and " << operandName
                             << " to have the same rank";
  for (size_t i = 0; i < scaleShape.size(); ++i) {
    if (scaleShape[i] != ShapedType::kDynamic &&
        operandShape[i] != ShapedType::kDynamic &&
        scaleShape[i] != operandShape[i])
      return op->emitOpError() << "expects " << scaleName << " and " << operandName
                               << " to have the same shape";
  }

  auto scaleValid = getValidShapeVec(scaleTy);
  auto operandValid = getValidShapeVec(operandTy);
  if (scaleValid.size() != operandValid.size())
    return op->emitOpError() << "expects " << scaleName << " and " << operandName
                             << " to have the same valid_shape";
  for (size_t i = 0; i < scaleValid.size(); ++i) {
    if (scaleValid[i] != ShapedType::kDynamic &&
        operandValid[i] != ShapedType::kDynamic &&
        scaleValid[i] != operandValid[i])
      return op->emitOpError() << "expects " << scaleName << " and " << operandName
                               << " to have the same valid_shape";
  }
  return success();
}

static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };
  auto equalsKnown = [](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    for (auto [a, b] : llvm::zip(lhs, rhs)) {
      if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b)
        return false;
    }
    return true;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i]))
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
  }
  if (!equalsKnown(src0Valid, dstValid) && !equalsKnown(src1Valid, dstValid))
    return op->emitOpError(
        "expects at least one of src0/src1 valid_shape to match dst valid_shape");
  return success();
}

[[maybe_unused]] static bool hasKnownZeroValidRegion(Type ty) {
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return false;
  return valid[0] == 0 || valid[1] == 0;
}

static LogicalResult verifyScalarTileOp(Operation *op, Type srcTy, Type dstTy,
                                        StringRef srcName, StringRef dstName,
                                        bool requireValidRowsEqual,
                                        bool requireValidColsEqual) {
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << srcName
                             << " to be in the vec address space";
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << dstName
                             << " to be in the vec address space";
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName)))
    return failure();

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have rank-2 valid_shape";
  if (requireValidRowsEqual &&
      srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[0]";
  if (requireValidColsEqual &&
      srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[1]";
  return success();
}

static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                         Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual) {
  if (failed(verifyScalarTileOp(op, srcTy, dstTy, "src", "dst",
                                requireValidRowsEqual,
                                /*requireValidColsEqual=*/true)))
    return failure();
  if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
    op->emitOpError("scalar must be a scalar type (integer/float)");
    return failure();
  }
  return getElemTy(srcTy);
}

static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                  Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type e0 = getElemTy(src0Ty);
  Type e1 = getElemTy(src1Ty);
  if (!e0 || !e1) {
    op->emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst")))
    return failure();
  return e0;
}

static FailureOr<Type> verifyDistinctRowMajorUnaryTileOpCommon(
    Operation *op, Value src, Value dst, StringRef srcName = "src",
    StringRef dstName = "dst") {
  if (src == dst) {
    op->emitOpError("expects src and dst to use different storage");
    return failure();
  }
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName)))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    op->emitOpError("failed to get element type for src/dst");
    return failure();
  }
  if (srcElem != dstElem) {
    op->emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, srcName, dstName)))
    return failure();
  return srcElem;
}

static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool supported = elemTy.isInteger(32) || elemTy.isInteger(16) ||
                   elemTy.isF16() || elemTy.isF32();
  if (targetArch == PTOArch::A5)
    supported = supported || (allowInt8OnA5 && elemTy.isInteger(8)) ||
                (allowBf16OnA5 && elemTy.isBF16());
  if (supported)
    return success();
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyArithmeticBinaryTileOpWithArchDispatch(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr))
      return failure();
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyArithmeticScalarTileOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error,
    bool requireValidRowsEqualOnA2A3 = true,
    bool requireValidRowsEqualOnA5 = false) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireValidRowsEqual) -> LogicalResult {
    FailureOr<Type> elemOr = verifyNumericScalarTileOpCommon(
        op, srcTy, dstTy, scalarTy, requireValidRowsEqual);
    if (failed(elemOr))
      return failure();
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireValidRowsEqualOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireValidRowsEqualOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyTColReductionElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool ok = elemTy.isF16() || elemTy.isF32() || elemTy.isInteger(16) ||
            elemTy.isInteger(32);
  if (targetArch == PTOArch::A5)
    ok = ok || (allowInt8OnA5 && elemTy.isInteger(8)) ||
         (allowBf16OnA5 && elemTy.isBF16());
  if (ok)
    return success();
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyTColReductionOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, bool requireNonZeroSrcOnA2A3,
    bool requireNonZeroSrcOnA5, bool allowInt8OnA5, bool allowBf16OnA5,
    StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireNonZeroSrc) -> LogicalResult {
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(op, dstTy, "dst")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return op->emitOpError("expects src and dst to have the same element type");
    if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc)))
      return failure();
    Type elem = getElemTy(srcTy);
    return verifyTColReductionElemTypeForArch(op, elem, targetArch, allowInt8OnA5,
                                              allowBf16OnA5, a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireNonZeroSrcOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireNonZeroSrcOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyTColArgReductionOpCommon(Operation *op, Type srcTy,
                                                    Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
    return failure();
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true)))
    return failure();
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32)))
    return op->emitOpError(
        "expects src/tmp element type to be 1, 2, or 4 bytes wide");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

static bool isKnownUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

static LogicalResult verifyVecTileStorage(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  return success();
}

static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto tb = dyn_cast<pto::TileBufType>(ty);
  auto as = getPTOMemorySpaceEnum(ty);
  if (as && *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (tb && tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  return success();
}

static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyVecTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyVecTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyVecTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName,
                                          StringRef dstName,
                                          bool allowBf16,
                                          bool allowInt8) {
  if (failed(verifyVecTileCommon(op, srcTy, srcName)) ||
      failed(verifyVecTileCommon(op, dstTy, dstName)))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName)))
    return failure();
  if (!isSupportedVecElemType(getElemTy(srcTy), allowBf16, allowInt8))
    return op->emitOpError() << "expects vec tile element types to be supported";
  return success();
}

static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::ACC)
    return op->emitOpError() << "expects " << name << " to be in the acc address space";
  return success();
}

static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyAccTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyAccTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyAccTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyMatTileAddressSpaces(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT ||
      *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }
  return success();
}

static LogicalResult verifyMatTileLogicalShapes(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if (lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] ||
      lhsShape[1] != rhsShape[0]) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }
  return success();
}

static LogicalResult verifyMatTileValidSizes(Operation *op, Type lhsTy,
                                             Type rhsTy) {
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() != 2 || rhsValid.size() != 2)
    return success();
  int64_t m = lhsValid[0];
  int64_t k = lhsValid[1];
  int64_t n = rhsValid[1];
  if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
      (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
      (n != ShapedType::kDynamic && (n < 1 || n > 4095))) {
    return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
  }
  return success();
}

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyMatTileAddressSpaces(op, lhsTy, rhsTy, dstTy)) ||
      failed(verifyMatTileLogicalShapes(op, lhsTy, rhsTy, dstTy)) ||
      failed(verifyMatTileValidSizes(op, lhsTy, rhsTy)))
    return failure();
  return success();
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy)))
    return failure();

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb)
    return success();

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor))
    return op->emitOpError("expects dst to use the col_major blayout on A5");

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst")))
    return failure();

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace)
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT)
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1)
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  if (isa<pto::TileBufType>(dstTy) && dstValid[0] != ShapedType::kDynamic &&
      dstValid[0] != 1)
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0])
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1])
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  return success();
}

static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy)))
    return failure();
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias")))
    return failure();
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS)
    return op->emitOpError("expects bias to be in the bias address space");
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1)
    return op->emitOpError("expects bias to have 1 row");
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32())
      return op->emitOpError("expects bias to have element type f32");
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias)))
    return failure();
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError("expects bias to use the row_major blayout on A5");
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
  return failure();
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(32) && isInt8(lhsElemTy) && isInt8(rhsElemTy))
    return success();

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy))
    return success();

  if (isA5 && dstElemTy.isF32() && lhsElemTy == rhsElemTy) {
    if (auto ft = mlir::dyn_cast<FloatType>(lhsElemTy)) {
      unsigned width = ft.getWidth();
      if (width == 8 || width == 16 || width == 32)
        return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", or an A5-supported fp8 pair" : "");
}
