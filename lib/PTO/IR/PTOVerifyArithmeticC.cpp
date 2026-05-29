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

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32()))
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32)))
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, MemRefType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);

    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile))
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    if (!mlir::isa<IntegerType, FloatType>(scalarTy))
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false)))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32())
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<std::pair<Type, std::optional<pto::AddressSpace>>>
verifyTExpandsCommon(TExpandsOp op) {
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!dstSpace ||
      (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects dst to be in the vec or mat address space"),
           failure();
  }
  Type dstElem = getElemTy(dstTy);
  if (op.getScalar().getType() != dstElem)
    return op.emitOpError("expects scalar type == dst element type"), failure();
  return std::make_pair(dstElem, dstSpace);
}

static LogicalResult verifyTExpandsElemType(Operation *op, Type dstElem,
                                            StringRef error, bool allowI8) {
  if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())
    return success();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
    unsigned w = it.getWidth();
    if (w == 16 || w == 32 || (allowI8 && w == 8))
      return success();
  }
  return op->emitOpError(error);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyTExpandsCommon(*this);
    if (failed(common))
      return failure();
    Type dstTy = getDst().getType();
    auto [dstElem, dstSpace] = *common;
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy))
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    return verifyTExpandsElemType(
        getOperation(), dstElem,
        "expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32",
        /*allowI8=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyTExpandsCommon(*this);
    if (failed(common))
      return failure();
    auto [dstElem, dstSpace] = *common;
    (void)dstSpace;
    return verifyTExpandsElemType(
        getOperation(), dstElem,
        "expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32",
        /*allowI8=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

struct IndexedTileTransferCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static bool hasTileBufLayout(pto::TileBufType ty, pto::BLayout bl,
                             pto::SLayout sl) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(bl) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(sl);
}

static bool hasMatExtractSourceLayoutA2A3(pto::TileBufType srcTy) {
  return srcTy.getBLayoutValueI32() ==
             static_cast<int32_t>(pto::BLayout::RowMajor) ||
         (srcTy.getBLayoutValueI32() !=
              static_cast<int32_t>(pto::BLayout::RowMajor) &&
          srcTy.getSLayoutValueI32() ==
              static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool hasMatExtractSourceLayoutA5(pto::TileBufType srcTy,
                                        pto::AddressSpace dstSpace) {
  const bool rowMajorSrc = srcTy.getBLayoutValueI32() ==
                           static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool colMajorView = srcTy.getSLayoutValueI32() ==
                            static_cast<int32_t>(pto::SLayout::ColMajor);
  const bool rowMajorView = srcTy.getSLayoutValueI32() ==
                            static_cast<int32_t>(pto::SLayout::RowMajor);
  if (dstSpace == pto::AddressSpace::LEFT)
    return (rowMajorSrc && colMajorView) || (!rowMajorSrc && rowMajorView) ||
           rowMajorSrc;
  return (rowMajorSrc && colMajorView) || (!rowMajorSrc && rowMajorView);
}

static bool isRowMajorNoneBoxNDTileBuf(pto::TileBufType ty) {
  return hasTileBufLayout(ty, pto::BLayout::RowMajor, pto::SLayout::NoneBox);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA2A3ExtractElemType(Type ty) {
  return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5ExtractElemType(Type ty) {
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8;
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
  return false;
}

static bool isA2A3VecInsertElemType(Type ty) {
  return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5SupportedVecInsertElemType(Type ty) {
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8 || it.getWidth() == 32;
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
  return false;
}

static FailureOr<IndexedTileTransferCommon> verifyIndexedTileTransferCommon(
    Operation *op, Value src, Value dst, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold, bool isInsertOp,
    bool requireSameElementType) {
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb)
    return op->emitOpError("expects src and dst to be !pto.tile_buf"), failure();

  auto verifyBounds =
      isInsertOp ? verifyInsertStaticBoundsCommon : verifyExtractStaticBoundsCommon;
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyNonNegativeIndexRowCol(
          *op, indexRow, indexCol, includeIndexAndIntOpsInConstFold)) ||
      failed(verifyBounds(*op, indexRow, indexCol, srcTy, dstTy,
                          includeIndexAndIntOpsInConstFold))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (requireSameElementType && (!srcElem || !dstElem || srcElem != dstElem))
    return op->emitOpError("expects src and dst to have the same element type"),
           failure();

  return IndexedTileTransferCommon{
      srcTy, dstTy, srcTb, dstTb, srcElem, dstElem,
      getPTOMemorySpaceEnum(srcTy), getPTOMemorySpaceEnum(dstTy)};
}

static LogicalResult verifyTExtractA2A3(const IndexedTileTransferCommon &common,
                                        TExtractOp op) {
  if (!isA2A3ExtractElemType(common.dstElem))
    return op.emitOpError(
        "expects A2/A3 textract element type to be i8/f16/bf16/f32");
  if (common.srcSpace && common.dstSpace &&
      *common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC) {
    return success();
  }
  if (!common.srcSpace || *common.srcSpace != pto::AddressSpace::MAT)
    return op.emitOpError("expects A2/A3 textract src to use loc=mat or vec");
  if (!common.dstSpace || (*common.dstSpace != pto::AddressSpace::LEFT &&
                           *common.dstSpace != pto::AddressSpace::RIGHT)) {
    return op.emitOpError(
        "expects A2/A3 textract dst to use loc=left, loc=right, or loc=vec");
  }
  if (!hasMatExtractSourceLayoutA2A3(common.srcTb))
    return op.emitOpError(
        "expects A2/A3 textract src to use a supported mat blayout/slayout combination");
  if (*common.dstSpace == pto::AddressSpace::LEFT &&
      !hasTileBufLayout(common.dstTb, pto::BLayout::RowMajor,
                        pto::SLayout::RowMajor)) {
    return op.emitOpError(
        "expects A2/A3 left dst to use row_major blayout and row_major slayout");
  }
  if (*common.dstSpace == pto::AddressSpace::RIGHT &&
      !hasTileBufLayout(common.dstTb, pto::BLayout::RowMajor,
                        pto::SLayout::ColMajor)) {
    return op.emitOpError(
        "expects A2/A3 right dst to use row_major blayout and col_major slayout");
  }
  return success();
}

static LogicalResult verifyTExtractA5Pair(const IndexedTileTransferCommon &common,
                                          TExtractOp op) {
  const bool okPair =
      (*common.srcSpace == pto::AddressSpace::MAT &&
       (*common.dstSpace == pto::AddressSpace::LEFT ||
        *common.dstSpace == pto::AddressSpace::RIGHT ||
        *common.dstSpace == pto::AddressSpace::SCALING)) ||
      (*common.srcSpace == pto::AddressSpace::VEC &&
       (*common.dstSpace == pto::AddressSpace::MAT ||
        *common.dstSpace == pto::AddressSpace::VEC));
  if (!okPair) {
    return op.emitOpError(
        "expects A5 textract to use a supported src/dst loc pair");
  }
  return success();
}

static LogicalResult verifyTExtractA5Layouts(const IndexedTileTransferCommon &common,
                                             TExtractOp op) {
  if (*common.srcSpace == pto::AddressSpace::MAT) {
    if (!hasMatExtractSourceLayoutA5(common.srcTb, *common.dstSpace)) {
      return op.emitOpError(
          "expects A5 textract src to use a supported mat blayout/slayout combination");
    }
    if (*common.dstSpace == pto::AddressSpace::LEFT &&
        !hasTileBufLayout(common.dstTb, pto::BLayout::ColMajor,
                          pto::SLayout::RowMajor)) {
      return op.emitOpError(
          "expects A5 left dst to use col_major blayout and row_major slayout");
    }
    if (*common.dstSpace == pto::AddressSpace::RIGHT &&
        !hasTileBufLayout(common.dstTb, pto::BLayout::RowMajor,
                          pto::SLayout::ColMajor)) {
      return op.emitOpError(
          "expects A5 right dst to use row_major blayout and col_major slayout");
    }
    return success();
  }
  if (*common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC &&
      (!isRowMajorNoneBoxNDTileBuf(common.srcTb) ||
       !isRowMajorNoneBoxNDTileBuf(common.dstTb))) {
    return op.emitOpError(
        "expects A5 vec->vec textract src/dst to use ND layout "
        "(blayout=row_major, slayout=none_box)");
  }
  return success();
}

static LogicalResult verifyTExtractA5(const IndexedTileTransferCommon &common,
                                      TExtractOp op) {
  if (!isA5ExtractElemType(common.dstElem))
    return op.emitOpError(
        "expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
  if (!common.srcSpace || !common.dstSpace)
    return op.emitOpError("expects src and dst to have explicit loc");
  if (failed(verifyTExtractA5Pair(common, op)) ||
      failed(verifyTExtractA5Layouts(common, op)))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto common = verifyIndexedTileTransferCommon(
      getOperation(), getSrc(), getDst(), getIndexRow(), getIndexCol(),
      /*includeIndexAndIntOpsInConstFold=*/false, /*isInsertOp=*/false,
      /*requireSameElementType=*/true);
  if (failed(common))
    return failure();

  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTExtractA2A3(*common, *this);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTExtractA5(*common, *this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTInsertA2A3(const IndexedTileTransferCommon &common,
                                       TInsertOp op) {
  if (common.srcSpace && common.dstSpace &&
      *common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC) {
    if (common.srcElem != common.dstElem ||
        !isA2A3VecInsertElemType(common.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
          "(i8/f16/bf16/f32)");
    }
    return success();
  }
  if (!common.srcSpace || !common.dstSpace ||
      *common.srcSpace != pto::AddressSpace::ACC ||
      *common.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");
  }
  if (!isColMajorRowMajorNZTileBuf(common.srcTb))
    return op.emitOpError(
        "expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
  if (!isColMajorRowMajorNZTileBuf(common.dstTb))
    return op.emitOpError(
        "expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
  if (common.dstTb.getSFractalSizeI32() != 512)
    return op.emitOpError("expects A2/A3 tinsert dst fractal size to be 512");
  if (!(common.srcElem.isF32() &&
        (common.dstElem.isF16() || common.dstElem.isBF16()))) {
    return op.emitOpError(
        "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTInsertA5AccToMat(const IndexedTileTransferCommon &common,
                                             TInsertOp op) {
  if (!isColMajorRowMajorNZTileBuf(common.srcTb))
    return op.emitOpError(
        "expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
  if (!isColMajorRowMajorNZTileBuf(common.dstTb))
    return op.emitOpError(
        "expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
  const bool okTypes =
      (common.srcElem.isF32() &&
       (common.dstElem.isF16() || common.dstElem.isBF16() ||
        common.dstElem.isF32())) ||
      (common.srcElem.isInteger(32) && common.dstElem.isInteger(32));
  if (!okTypes) {
    return op.emitOpError(
        "expects A5 acc->mat tinsert element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecToMat(const IndexedTileTransferCommon &common,
                                             TInsertOp op) {
  if (!isColMajorRowMajorNZTileBuf(common.dstTb)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
  }
  const bool srcIsND = isRowMajorNoneBoxNDTileBuf(common.srcTb);
  const bool srcIsNZ = isColMajorRowMajorNZTileBuf(common.srcTb);
  if (!srcIsND && !srcIsNZ) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
  }
  if (common.srcElem != common.dstElem ||
      !isA5SupportedVecInsertElemType(common.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecToVec(const IndexedTileTransferCommon &common,
                                             TInsertOp op) {
  if (!isRowMajorNoneBoxNDTileBuf(common.srcTb) ||
      !isRowMajorNoneBoxNDTileBuf(common.dstTb)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to use ND layout "
        "(blayout=row_major, slayout=none_box)");
  }
  if (common.srcElem != common.dstElem ||
      !isA5SupportedVecInsertElemType(common.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5(const IndexedTileTransferCommon &common,
                                     TInsertOp op) {
  if (!common.srcSpace || !common.dstSpace)
    return op.emitOpError("expects A5 tinsert src/dst to have explicit loc");

  if (*common.srcSpace == pto::AddressSpace::ACC &&
      *common.dstSpace == pto::AddressSpace::MAT)
    return verifyTInsertA5AccToMat(common, op);

  if (*common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::MAT)
    return verifyTInsertA5VecToMat(common, op);

  if (*common.srcSpace == pto::AddressSpace::VEC &&
      *common.dstSpace == pto::AddressSpace::VEC)
    return verifyTInsertA5VecToVec(common, op);

  return op.emitOpError(
      "expects A5 tinsert to use a supported src/dst loc pair: "
      "acc->mat, vec->mat, or vec->vec");
}

mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto common = verifyIndexedTileTransferCommon(
      getOperation(), getSrc(), getDst(), getIndexRow(), getIndexCol(),
      /*includeIndexAndIntOpsInConstFold=*/true, /*isInsertOp=*/true,
      /*requireSameElementType=*/false);
  if (failed(common))
    return failure();

  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTInsertA2A3(*common, *this);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTInsertA5(*common, *this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isA2A3VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(8);
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  return false;
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == 8;
  return false;
}

static bool isA5MxInputType(Type ty) {
  return isA5Fp8LikeType(ty);
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputType(lhsElem) || !isA5MxInputType(rhsElem))
    return op->emitOpError()
           << "expects A5 mx operands " << lhsName << " and " << rhsName
           << " to use fp8 element types";

  if (!dstElem.isF32())
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(8) || isA5Fp8LikeType(dstElem) || dstElem.isF16() ||
           dstElem.isBF16() || dstElem.isF32();
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  return false;
}

static FailureOr<std::tuple<Type, Type, Type, pto::TileBufType, pto::TileBufType,
                            pto::TileBufType, pto::AddressSpace,
                            pto::AddressSpace, pto::AddressSpace>>
verifyVectorPreQuantTransferCommon(Operation *op, Value src, Value fp, Value dst,
                                   Value indexRow, Value indexCol,
                                   bool isInsertOp) {
  Type srcTy = src.getType();
  Type fpTy = fp.getType();
  Type dstTy = dst.getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !fpTb || !dstTb)
    return op->emitOpError("expects src, fp, and dst to be !pto.tile_buf"),
           failure();
  auto verifyBounds = isInsertOp ? verifyInsertStaticBoundsCommon
                                 : verifyExtractStaticBoundsCommon;
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyNonNegativeIndexRowCol(
          *op, indexRow, indexCol,
          /*includeIndexAndIntOpsInConstFold=*/true)) ||
      failed(verifyBounds(*op, indexRow, indexCol, srcTy, dstTy,
                          /*includeIndexAndIntOpsInConstFold=*/true)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto fpSpace = getPTOMemorySpaceEnum(fpTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !fpSpace || !dstSpace)
    return op->emitOpError("expects src, fp, and dst to have explicit loc"),
           failure();
  if (*srcSpace != pto::AddressSpace::ACC)
    return op->emitOpError("expects src to use loc=acc"), failure();
  if (*fpSpace != pto::AddressSpace::SCALING)
    return op->emitOpError("expects fp to use loc=scaling"), failure();
  if (*dstSpace != pto::AddressSpace::MAT)
    return op->emitOpError("expects dst to use loc=mat"), failure();
  if (!isColMajorRowMajorNZTileBuf(srcTb))
    return op->emitOpError(
               "expects src to use blayout=col_major and slayout=row_major"),
           failure();
  if (!isColMajorRowMajorNZTileBuf(dstTb))
    return op->emitOpError(
               "expects dst to use blayout=col_major and slayout=row_major"),
           failure();
  return std::make_tuple(srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, *srcSpace,
                         *fpSpace, *dstSpace);
}

using VectorPreQuantTypePairFn = bool (*)(Type, Type);

static LogicalResult verifyVectorPreQuantTransferOp(
    Operation *op, Value src, Value fp, Value dst, Value indexRow,
    Value indexCol, bool isInsertOp, bool requireDstFractal512,
    VectorPreQuantTypePairFn verifyTypePair, llvm::StringRef message) {
  auto common = verifyVectorPreQuantTransferCommon(op, src, fp, dst, indexRow,
                                                   indexCol, isInsertOp);
  if (failed(common))
    return failure();
  auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
      *common;
  (void)fpTy;
  (void)srcTb;
  (void)fpTb;
  (void)srcSpace;
  (void)fpSpace;
  (void)dstSpace;
  if (requireDstFractal512 && dstTb.getSFractalSizeI32() != 512)
    return op->emitOpError("expects dst fractal size to be 512");
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!verifyTypePair(srcElem, dstElem))
    return op->emitOpError(message);
  return success();
}

mlir::LogicalResult mlir::pto::TExtractFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/false,
        /*requireDstFractal512=*/true, isA2A3VectorPreQuantTypePair,
        "expects A2/A3 textract_fp element types to be (src=f32,dst=i8) "
        "or (src=i32,dst=i8/f16/i16)");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/false,
        /*requireDstFractal512=*/false, isA5VectorPreQuantTypePair,
        "expects A5 textract_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
        "or (src=i32,dst=i8/f16/bf16)");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TInsertFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/true,
        /*requireDstFractal512=*/true, isA2A3VectorPreQuantTypePair,
        "expects A2/A3 tinsert_fp element types to be (src=f32,dst=i8) "
        "or (src=i32,dst=i8/f16/i16)");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyVectorPreQuantTransferOp(
        getOperation(), getSrc(), getFp(), getDst(), getIndexRow(),
        getIndexCol(),
        /*isInsertOp=*/true,
        /*requireDstFractal512=*/false, isA5VectorPreQuantTypePair,
        "expects A5 tinsert_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
        "or (src=i32,dst=i8/f16/bf16)");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static int64_t getFillPadElemBytes(Type type) {
  unsigned elemBytes = getPTOStorageElemByteSize(type);
  return elemBytes == 0 ? -1 : static_cast<int64_t>(elemBytes);
}

static LogicalResult verifyTFillPadMatHomogeneousConstraint(Operation *op,
                                                            Type srcTy,
                                                            Type dstTy,
                                                            llvm::StringRef opName) {
  if (opName != "tfillpad")
    return success();
  auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
  auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!(srcTb && dstTb && srcSpace && dstSpace &&
        *srcSpace == mlir::pto::AddressSpace::MAT &&
        *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb)) {
    return success();
  }

  auto dimToStr = [](int64_t dim) -> std::string {
    return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
  };
  SmallVector<std::string, 4> mismatchFields;
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() == 2 && dstValid.size() == 2) {
    if (srcValid[0] != dstValid[0])
      mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                               dimToStr(dstValid[0]) + ")");
    if (srcValid[1] != dstValid[1])
      mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                               dimToStr(dstValid[1]) + ")");
  }
  if (srcTb.getPadValueI32() != dstTb.getPadValueI32()) {
    mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                             " vs " + std::to_string(dstTb.getPadValueI32()) +
                             ")");
  }

  auto diag = op->emitError()
              << "expects src/dst tile types to be lowerable to TFILLPAD "
                 "for loc=mat";
  if (!mismatchFields.empty())
    diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
  diag << "\n  src: " << srcTy;
  diag << "\n  dst: " << dstTy;
  diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
  return failure();
}

static LogicalResult verifyTFillPadDstPad(Operation *op, Type dstTy,
                                          llvm::StringRef opName) {
  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr =
        mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError() << "expects dst PadVal != Null for " << opName;
  }
  return success();
}

static LogicalResult verifyTFillPadShapeCompatibility(Operation *op,
                                                      ArrayRef<int64_t> srcShape,
                                                      ArrayRef<int64_t> dstShape,
                                                      bool allowDstExpand,
                                                      llvm::StringRef opName) {
  if (!allowDstExpand) {
    if (srcShape != dstShape) {
      return op->emitError()
             << "expects src and dst to have the same static shape for "
             << opName;
    }
    return mlir::success();
  }
  if (srcShape[0] > dstShape[0] || srcShape[1] > dstShape[1]) {
    return op->emitError()
           << "expects dst static shape to be >= src static shape for "
           << opName;
  }
  return mlir::success();
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy, Type dstTy,
                                              bool allowDstExpand,
                                              llvm::StringRef opName) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op->emitError("expects rank-2 shaped types for src/dst");

  int64_t srcB = getFillPadElemBytes(getElemTy(srcTy));
  int64_t dstB = getFillPadElemBytes(getElemTy(dstTy));
  if (srcB < 0 || dstB < 0)
    return op->emitError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");

  if (failed(verifyTFillPadMatHomogeneousConstraint(op, srcTy, dstTy, opName)) ||
      failed(verifyTFillPadDstPad(op, dstTy, opName)) ||
      failed(verifyTFillPadShapeCompatibility(op, srcShape, dstShape,
                                              allowDstExpand, opName))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad");
}

mlir::LogicalResult mlir::pto::TFillPadExpandOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/true, "tfillpad_expand");
}

mlir::LogicalResult mlir::pto::TFillPadInplaceOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad_inplace");
}

struct GatherSrcDstCommon {
  Type srcTy;
  Type dstTy;
  Type srcElem;
  Type dstElem;
};

struct GatherIndexCommon {
  GatherSrcDstCommon base;
  Type idxTy;
  Type tmpTy;
  IntegerType idxElem;
};

struct GatherCompareCommon {
  GatherSrcDstCommon base;
  Type cdstTy;
  Type tmpTy;
  Type cdstElem;
  pto::CmpMode cmpMode;
};

static bool isSupportedGatherElemTypeA5Index(Type ty) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  return false;
}

static LogicalResult verifyTileValidWidthMatchesCols(Operation *op, Type ty,
                                                     StringRef operandName) {
  auto validShape = getValidShapeVec(ty);
  auto shape = getShapeVec(ty);
  if (validShape.size() == 2 && shape.size() == 2 &&
      validShape[1] != ShapedType::kDynamic &&
      shape[1] != ShapedType::kDynamic && validShape[1] != shape[1]) {
    return op->emitOpError() << "expects " << operandName
                             << " valid_shape[1] to equal " << operandName
                             << " cols";
  }
  return success();
}

static FailureOr<GatherSrcDstCommon> verifyGatherSrcDstCommon(TGatherOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return op.emitOpError("failed to get element type for src/dst"), failure();
  return GatherSrcDstCommon{srcTy, dstTy, srcElem, dstElem};
}

static LogicalResult verifyGatherMaskForm(TGatherOp op,
                                          bool allowA5MaskTypes) {
  auto common = verifyGatherSrcDstCommon(op);
  if (failed(common))
    return failure();
  if (!isRowMajorTileBuf(common->srcTy) || !isRowMajorTileBuf(common->dstTy))
    return op.emitOpError("expects src and dst to use row-major layout");

  auto srcSpace = getPTOMemorySpaceEnum(common->srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(common->dstTy);
  if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src and dst to be in the vec address space");
  }

  unsigned srcElemBytes = getPTOStorageElemByteSize(common->srcElem);
  unsigned dstElemBytes = getPTOStorageElemByteSize(common->dstElem);
  if (srcElemBytes == 0 || dstElemBytes == 0)
    return op.emitOpError("failed to get element size for src/dst");
  if (srcElemBytes != dstElemBytes)
    return op.emitOpError("expects src and dst element sizes to match");
  if (failed(verifyTileValidWidthMatchesCols(op.getOperation(), common->dstTy,
                                             "dst"))) {
    return failure();
  }

  if (allowA5MaskTypes) {
    if (!(srcElemBytes == 1 || srcElemBytes == 2 || srcElemBytes == 4)) {
      return op.emitOpError(
          "expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
    }
    if (!isSupportedGatherElemTypeA5(common->srcElem) ||
        !isSupportedGatherElemTypeA5(common->dstElem)) {
      return op.emitOpError(
          "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
    }
    return success();
  }

  if (!(srcElemBytes == 2 || srcElemBytes == 4)) {
    return op.emitOpError(
        "expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
  }
  return success();
}

static FailureOr<GatherIndexCommon> verifyGatherIndexCommon(TGatherOp op) {
  auto base = verifyGatherSrcDstCommon(op);
  if (failed(base))
    return failure();
  Type idxTy = op.getIndices().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, idxTy, "indices")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (base->srcElem != base->dstElem)
    return op.emitOpError("expects src and dst to have the same element type"),
           failure();
  auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
  if (!idxElem)
    return op.emitOpError("indices element type must be integer"), failure();
  return GatherIndexCommon{*base, idxTy, tmpTy, idxElem};
}

static LogicalResult verifyGatherIndexForm(TGatherOp op,
                                           bool allow16BitIndices,
                                           bool allowA5ElemTypes) {
  auto common = verifyGatherIndexCommon(op);
  if (failed(common))
    return failure();
  if (allowA5ElemTypes) {
    if (!isSupportedGatherElemTypeA5Index(common->base.srcElem) ||
        !isSupportedGatherElemTypeA5Index(common->base.dstElem)) {
      return op.emitOpError(
          "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    }
  } else if (!isSupportedGatherElemTypeA2A3(common->base.srcElem) ||
             !isSupportedGatherElemTypeA2A3(common->base.dstElem)) {
    return op.emitOpError(
        "expects gather src/dst element type to be i16/i32/f16/f32");
  }

  unsigned width = common->idxElem.getWidth();
  if (!(width == 32 || (allow16BitIndices && width == 16))) {
    return op.emitOpError() << "expects indices element type to be i32"
                            << (allow16BitIndices ? " or i16" : "");
  }
  if (failed(verifyTileValidWidthMatchesCols(op.getOperation(), common->base.dstTy,
                                             "dst")) ||
      failed(verifyTileValidWidthMatchesCols(op.getOperation(), common->idxTy,
                                             "indices"))) {
    return failure();
  }
  if (!allowA5ElemTypes) {
    if (getElemTy(common->tmpTy) != common->idxElem)
      return op.emitOpError(
          "expects tmp and indices to have the same element type");
    if (failed(verifyTileBufSameValidShape(op, common->idxTy, common->tmpTy,
                                           "indices", "tmp"))) {
      return failure();
    }
  }
  return success();
}

static FailureOr<GatherCompareCommon> verifyGatherCompareCommon(TGatherOp op) {
  auto base = verifyGatherSrcDstCommon(op);
  if (failed(base))
    return failure();
  Type cdstTy = op.getCdst().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, cdstTy, "cdst")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type cdstElem = getElemTy(cdstTy);
  if (!cdstElem)
    return op.emitOpError("failed to get element type for src/dst/cdst"),
           failure();
  auto dstInt = dyn_cast<IntegerType>(base->dstElem);
  if (!dstInt || dstInt.getWidth() != 32)
    return op.emitOpError("expects dst element type to be i32"), failure();
  if (cdstElem != base->dstElem)
    return op.emitOpError("expects cdst to have the same element type as dst"),
           failure();
  if (op.getKValue().getType() != base->srcElem) {
    return op.emitOpError(
               "expects kValue to have the same type as src element type"),
           failure();
  }
  auto cmpAttr = op.getCmpModeAttr();
  pto::CmpMode cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
  if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT) {
    return op.emitOpError(
               "expects compare-form tgather cmpMode to be eq or gt"),
           failure();
  }
  return GatherCompareCommon{*base, cdstTy, tmpTy, cdstElem, cmpMode};
}

static LogicalResult verifyGatherCompareForm(TGatherOp op,
                                             bool allowA5SrcTypes) {
  auto common = verifyGatherCompareCommon(op);
  if (failed(common))
    return failure();
  if (allowA5SrcTypes) {
    if (!(common->base.srcElem.isF16() || common->base.srcElem.isF32() ||
          common->base.srcElem.isInteger(16) ||
          common->base.srcElem.isInteger(32))) {
      return op.emitOpError(
          "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
    }
  } else if (!(common->base.srcElem.isF16() || common->base.srcElem.isF32() ||
               (common->base.srcElem.isInteger(32) &&
                common->cmpMode == pto::CmpMode::EQ))) {
    return op.emitOpError(
        "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
  }

  if (failed(verifyVecTileCommonA2A3(op, common->base.srcTy, "src")) ||
      failed(verifyVecTileCommonA2A3(op, common->base.dstTy, "dst")) ||
      failed(verifyVecTileCommonA2A3(op, common->cdstTy, "cdst")) ||
      failed(verifyVecTileCommonA2A3(op, common->tmpTy, "tmp"))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTGatherForArch(TGatherOp op, bool allowA5Forms) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue())
      return op.emitOpError(
          "mask-pattern tgather only allows src and dst operands");
    return verifyGatherMaskForm(op, /*allowA5MaskTypes=*/allowA5Forms);
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp())
      return op.emitOpError(
          "compare-form tgather expects dst, cdst, kValue, and tmp");
    if (op.getIndices())
      return op.emitOpError("compare-form tgather does not take indices");
    return verifyGatherCompareForm(op, /*allowA5SrcTypes=*/allowA5Forms);
  }
  if (!op.getIndices() || !op.getTmp())
    return op.emitOpError("index-form tgather expects both indices and tmp");
  return verifyGatherIndexForm(op, /*allow16BitIndices=*/allowA5Forms,
                               /*allowA5ElemTypes=*/allowA5Forms);
}

llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTGatherForArch(*this, false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTGatherForArch(*this, true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type srcTy = getSrc().getType();
    Type offTy = getOffsets().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, offTy, "offsets")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto dstElemTy = getElemTy(dstTy);
    if (!srcElemTy || !dstElemTy)
      return emitOpError() << "failed to get element type for src/dst";
    return std::make_pair(srcElemTy, dstElemTy);
  };

  auto getElemBytes = [](Type ty) -> std::optional<unsigned> {
    unsigned elemBytes = getPTOStorageElemByteSize(ty);
    if (elemBytes == 0)
      return std::nullopt;
    return elemBytes;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstTy = getDst().getType();
    Type dstElemTy = elems->second;
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError() << "expects dst to use row-major layout";
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstElemTy = elems->second;
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2)
      return emitOpError("expects src to have rank-2 valid_shape");
    if (valid[0] != ShapedType::kDynamic && valid[0] <= 0)
      return emitOpError("expects src valid_shape[0] to be positive");
    if (valid[1] != ShapedType::kDynamic && valid[1] <= 0)
      return emitOpError("expects src valid_shape[1] to be positive");
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    if (!getSlope().getType().isF32())
      return emitOpError() << "expects slope to have type f32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmax element type to be i32/i16/f16/f32",
      "expects A5 tmax element type to be i32/i16/i8/f16/f32");
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmaxs element type to be i32/i16/f16/f32",
      "expects A5 tmaxs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmin element type to be i32/i16/f16/f32",
      "expects A5 tmin element type to be i32/i16/i8/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmins element type to be i32/i16/f16/f32",
      "expects A5 tmins element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}

struct TMovCommonInfo {
  Type srcTy;
  Type dstTy;
  Value fp;
  TileBufType srcTb;
  TileBufType dstTb;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
  bool hasFp = false;
  bool hasPreQuantScalar = false;
  bool isMatToTile = false;
  bool isVecToVec = false;
  bool isVecToMat = false;
  bool isAccToMat = false;
  bool isAccToVec = false;
};

struct TMovFpCommonInfo {
  Type srcTy;
  Type fpTy;
  Type dstTy;
  Type srcElemTy;
  TileBufType srcTb;
  TileBufType dstTb;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> fpSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TMovCommonInfo> verifyTMovCommon(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasFp && failed(verifyTileBufCommon(op, fp.getType(), "fp")))
    return failure();
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError(
               "expects fp and preQuantScalar forms to be mutually exclusive"),
           failure();
  }

  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace)
    return op.emitOpError("expects src and dst to have explicit address spaces"),
           failure();

  TMovCommonInfo info{
      srcTy, dstTy, fp, dyn_cast<pto::TileBufType>(srcTy),
      dyn_cast<pto::TileBufType>(dstTy), srcSpace, dstSpace, hasFp,
      hasPreQuantScalar};
  info.isMatToTile =
      *srcSpace == pto::AddressSpace::MAT &&
      (*dstSpace == pto::AddressSpace::LEFT ||
       *dstSpace == pto::AddressSpace::RIGHT ||
       *dstSpace == pto::AddressSpace::BIAS ||
       *dstSpace == pto::AddressSpace::SCALING);
  info.isVecToVec = *srcSpace == pto::AddressSpace::VEC &&
                    *dstSpace == pto::AddressSpace::VEC;
  info.isVecToMat = *srcSpace == pto::AddressSpace::VEC &&
                    *dstSpace == pto::AddressSpace::MAT;
  info.isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                    *dstSpace == pto::AddressSpace::MAT;
  info.isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                    *dstSpace == pto::AddressSpace::VEC;
  return info;
}

static LogicalResult verifyTMovShapes(TMovOp op, const TMovCommonInfo &info,
                                      bool isA5) {
  auto srcShape = getShapeVec(info.srcTy);
  auto dstShape = getShapeVec(info.dstTy);
  if (*info.srcSpace == pto::AddressSpace::MAT && srcShape != dstShape)
    return op.emitOpError(
        "expects mat-source tmov to use matching src/dst shapes");
  if (!isA5 && *info.srcSpace != pto::AddressSpace::MAT && srcShape != dstShape)
    return op.emitOpError(
        "expects A2/A3 non-mat tmov to use matching src/dst shapes");
  return success();
}

static LogicalResult verifyTMovAddressSpacePair(TMovOp op,
                                                const TMovCommonInfo &info,
                                                bool isA5) {
  bool okPair = info.isMatToTile || info.isVecToVec || info.isAccToMat ||
                info.isAccToVec || (isA5 && info.isVecToMat);
  if (!okPair)
    return op.emitOpError(
        "expects a supported tmov address-space pair for this target");
  if (op.getAccToVecModeAttr() && !info.isAccToVec) {
    return op.emitOpError(
        "expects accToVecMode to be used only for acc-to-vec tmov");
  }
  return success();
}

static LogicalResult verifyTMovDerivedForms(TMovOp op,
                                            const TMovCommonInfo &info) {
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      !(info.isAccToMat || info.isAccToVec)) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  if (info.hasPreQuantScalar && !(info.isAccToMat || info.isAccToVec)) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (!info.hasFp)
    return success();

  auto fpSpace = getPTOMemorySpaceEnum(info.fp.getType());
  if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
    return op.emitOpError("expects fp to be in the scaling address space");
  auto srcElemTy = getElemTy(info.srcTy);
  auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
  if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32))) {
    return op.emitOpError("expects fp form src to have element type f32, i32");
  }
  if (!(info.isAccToMat || info.isAccToVec))
    return op.emitOpError("expects fp form to use loc=acc src");
  return success();
}

static LogicalResult verifyTMovAccToVecMode(TMovOp op,
                                            const TMovCommonInfo &info) {
  auto accToVecModeAttr = op.getAccToVecModeAttr();
  if (!(info.hasFp || info.hasPreQuantScalar) || !accToVecModeAttr)
    return success();
  switch (accToVecModeAttr.getValue()) {
  case pto::AccToVecMode::SingleModeVec0:
  case pto::AccToVecMode::SingleModeVec1:
    return success();
  case pto::AccToVecMode::DualModeSplitM:
  case pto::AccToVecMode::DualModeSplitN:
    return op.emitOpError(
        "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode");
  }
  return success();
}

static LogicalResult verifyTMovLayouts(TMovOp op, const TMovCommonInfo &info,
                                       bool isA5) {
  if (info.srcTb && *info.srcSpace == pto::AddressSpace::ACC &&
      (info.hasFp || op.getReluPreMode() != pto::ReluPreMode::NoRelu) &&
      !isColMajorRowMajorNZTileBuf(info.srcTb)) {
    return op.emitOpError(
        "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major");
  }
  if (info.srcTb && info.dstTb && info.isAccToMat && !isA5 &&
      info.dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError(
        "expects A2/A3 acc-to-mat tmov destination fractal to be 512");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyImpl = [&](bool isA5) -> LogicalResult {
    auto common = verifyTMovCommon(*this);
    if (failed(common))
      return failure();
    if (failed(verifyTMovShapes(*this, *common, isA5)) ||
        failed(verifyTMovAddressSpacePair(*this, *common, isA5)) ||
        failed(verifyTMovDerivedForms(*this, *common)) ||
        failed(verifyTMovAccToVecMode(*this, *common)) ||
        failed(verifyTMovLayouts(*this, *common, isA5))) {
      return failure();
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyImpl(/*isA5=*/false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyImpl(/*isA5=*/true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<TMovFpCommonInfo> verifyTMovFpCommon(TMovFPOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
  if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32))) {
    return op.emitOpError("expects src to have element type f32, i32"),
           failure();
  }
  auto fpSpace = getPTOMemorySpaceEnum(fpTy);
  if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
    return op.emitOpError("expects fp to be in the scaling address space"),
           failure();
  return TMovFpCommonInfo{
      srcTy, fpTy, dstTy, srcElemTy, dyn_cast<pto::TileBufType>(srcTy),
      dyn_cast<pto::TileBufType>(dstTy), getPTOMemorySpaceEnum(srcTy), fpSpace,
      getPTOMemorySpaceEnum(dstTy)};
}

static LogicalResult verifyTMovFpA2A3(const TMovFpCommonInfo &info,
                                      TMovFPOp op) {
  if (!info.srcSpace || *info.srcSpace != pto::AddressSpace::ACC)
    return op.emitOpError("expects src to be in the acc address space");
  if (!info.dstSpace || *info.dstSpace != pto::AddressSpace::MAT)
    return op.emitOpError("expects dst to be in the mat address space");
  if (info.srcTb && !isColMajorRowMajorNZTileBuf(info.srcTb))
    return op.emitOpError(
        "expects src to use blayout=col_major and slayout=row_major");
  if (info.dstTb && !isColMajorRowMajorNZTileBuf(info.dstTb))
    return op.emitOpError(
        "expects dst to use blayout=col_major and slayout=row_major");
  if (info.dstTb && info.dstTb.getSFractalSizeI32() != 512)
    return op.emitOpError("expects dst to use fractal size 512");
  return success();
}

static LogicalResult verifyTMovFpA5(const TMovFpCommonInfo &info,
                                    TMovFPOp op) {
  if (info.srcTb && !isColMajorRowMajorNZTileBuf(info.srcTb))
    return op.emitOpError(
        "expects src to use blayout=col_major and slayout=row_major");
  return success();
}

mlir::LogicalResult mlir::pto::TMovFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyTMovFpCommon(*this);
    if (failed(common))
      return failure();
    return verifyTMovFpA2A3(*common, *this);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyTMovFpCommon(*this);
    if (failed(common))
      return failure();
    return verifyTMovFpA5(*common, *this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<ShapedType>(t)) return s.getRank();
  if (auto tile = dyn_cast<pto::TileBufType>(t)) return tile.getRank();
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) return view.getRank();
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (ShapedType 或 Tile 类型)
  bool aValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid)
    return op->emitOpError("expects inputs/outputs to be shaped types or PTO tile types");

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank)
      return op->emitOpError("expects a and dst to have the same rank");
    if (bRank != -1 && dRank != -1 && bRank != dRank)
      return op->emitOpError("expects b and dst to have the same rank");
  }

  return success();
}
