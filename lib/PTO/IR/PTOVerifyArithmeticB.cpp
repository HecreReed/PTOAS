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

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();

  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/src2/dst to be memref/tile_buf types");

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd)
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}

static FailureOr<std::pair<Type, Type>> verifyTAxpyCommon(TAxpyOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  if (op.getScalar().getType() != srcElem)
    return op.emitOpError("expects scalar type to match src element type"),
           failure();
  if (getShapeVec(srcTy) != getShapeVec(dstTy))
    return op.emitOpError("expects src and dst to have the same shape"),
           failure();
  return std::make_pair(srcElem, getElemTy(dstTy));
}

static LogicalResult verifyTAxpyTypePair(Operation *op, Type srcElem,
                                         Type dstElem) {
  bool sameType = srcElem == dstElem;
  bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
  if (!(sameType || widenF16ToF32)) {
    return op->emitOpError(
        "expects dst/src element types to match, or dst=f32 and src=f16");
  }
  return success();
}

LogicalResult pto::TAxpyOp::verify() {

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyTAxpyCommon(*this);
    if (failed(common))
      return failure();
    auto [srcElem, dstElem] = *common;
    if (failed(verifyTAxpyTypePair(*this, srcElem, dstElem)))
      return failure();
    if (!(dstElem.isF16() || dstElem.isF32()))
      return emitOpError("expects A2/A3 taxpy dst element type to be f16/f32");
    if (!(srcElem.isF16() || srcElem.isF32()))
      return emitOpError("expects A2/A3 taxpy src element type to be f16/f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyTAxpyCommon(*this);
    if (failed(common))
      return failure();
    auto [srcElem, dstElem] = *common;
    if (failed(verifyTAxpyTypePair(*this, srcElem, dstElem)))
      return failure();
    if (!(dstElem.isF16() || dstElem.isF32() || dstElem.isBF16()))
      return emitOpError("expects A5 taxpy dst element type to be f16/bf16/f32");
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isBF16()))
      return emitOpError("expects A5 taxpy src element type to be f16/bf16/f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd)
    return emitOpError("expects src0/src1/dst to have the same shape");
  return success();
}

template <typename VerifyCommonFn>
static LogicalResult verifyArchIntegerWidthOp(Operation *op,
                                              VerifyCommonFn verifyCommon,
                                              StringRef a2a3Message,
                                              StringRef a5Message) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return op->emitOpError(a2a3Message);
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return op->emitOpError(a5Message);
    return success();
  };

  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyRowMajorBinaryIntWidthOp(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy,
    StringRef a2a3Message, StringRef a5Message) {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
  };
  return verifyArchIntegerWidthOp(op, verifyCommon, a2a3Message, a5Message);
}

static LogicalResult verifyDistinctRowMajorUnaryIntWidthOp(
    Operation *op, Value src, Value dst, StringRef srcName, StringRef dstName,
    StringRef a2a3Message, StringRef a5Message) {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(op, src, dst, srcName,
                                                   dstName);
  };
  return verifyArchIntegerWidthOp(op, verifyCommon, a2a3Message, a5Message);
}

LogicalResult pto::TAndOp::verify() {
  return verifyRowMajorBinaryIntWidthOp(
      getOperation(), getSrc0().getType(), getSrc1().getType(),
      getDst().getType(),
      "expects A2/A3 tand src0, src1, and dst element type to be i8/i16",
      "expects A5 tand src0, src1, and dst element type to be i8/i16/i32");
}

static LogicalResult verifyLocVecType(Operation *op, Type ty, StringRef name) {
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to use loc=vec";
  return success();
}

static LogicalResult verifyConcatElemType(Operation *op, Type elem) {
  if (elem.isF16() || elem.isF32() || elem.isBF16())
    return success();
  auto it = dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
              it.getWidth() != 32)) {
    return op->emitOpError(
        "expects element type to be i8, i16, i32, f16, f32, or bf16");
  }
  return success();
}

static LogicalResult verifyTConcatValidRows(TConcatOp op,
                                            ArrayRef<int64_t> src0Valid,
                                            ArrayRef<int64_t> src1Valid,
                                            ArrayRef<int64_t> dstValid) {
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError(
        "expects src0, src1, and dst to have rank-2 valid_shape");
  }
  if (src0Valid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && src0Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src0 valid row to match dst valid row");
  }
  if (src1Valid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && src1Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src1 valid row to match dst valid row");
  }
  return success();
}

static LogicalResult verifyTConcatDstCols(TConcatOp op, Type dstTy,
                                          ArrayRef<int64_t> src0Valid,
                                          ArrayRef<int64_t> src1Valid) {
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == 2 && dstShape[1] != ShapedType::kDynamic &&
      src0Valid[1] != ShapedType::kDynamic &&
      src1Valid[1] != ShapedType::kDynamic &&
      src0Valid[1] + src1Valid[1] > dstShape[1]) {
    return op.emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
  }
  return success();
}

struct BinaryTileTypeInfo {
  Type src0Ty;
  Type src1Ty;
  Type dstTy;
  Type src0Elem;
  Type src1Elem;
  Type dstElem;
};

template <typename VerifyFn>
static FailureOr<BinaryTileTypeInfo>
verifyBinaryTileTypeInfo(Operation *op, Value src0, Value src1, Value dst,
                         VerifyFn verifyOperand) {
  Type src0Ty = src0.getType();
  Type src1Ty = src1.getType();
  Type dstTy = dst.getType();
  if (failed(verifyOperand(op, src0Ty, "src0")) ||
      failed(verifyOperand(op, src1Ty, "src1")) ||
      failed(verifyOperand(op, dstTy, "dst"))) {
    return failure();
  }
  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !dstElem)
    return op->emitOpError("failed to get element type for operands"), failure();
  return BinaryTileTypeInfo{src0Ty, src1Ty, dstTy, src0Elem, src1Elem, dstElem};
}

static FailureOr<Type> verifyTConcatCommon(TConcatOp op) {
  auto verifyTileBufOperand = [](Operation *op, Type ty, StringRef name) {
    return verifyTileBufCommon(op, ty, name);
  };
  auto infoOr = verifyBinaryTileTypeInfo(op, op.getSrc0(), op.getSrc1(),
                                         op.getDst(), verifyTileBufOperand);
  if (failed(infoOr))
    return failure();
  const auto &info = *infoOr;
  if (info.src0Elem != info.src1Elem || info.src0Elem != info.dstElem) {
    return op.emitOpError(
               "expects src0, src1, and dst to have the same element type"),
           failure();
  }

  auto src0Valid = getValidShapeVec(op.getSrc0());
  auto src1Valid = getValidShapeVec(op.getSrc1());
  auto dstValid = getValidShapeVec(op.getDst());
  if (failed(verifyTConcatValidRows(op, src0Valid, src1Valid, dstValid)) ||
      failed(verifyTConcatDstCols(op, info.dstTy, src0Valid, src1Valid)))
    return failure();
  return info.src0Elem;
}

static LogicalResult verifyTColExpandRowMajorLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (auto tileTy = dyn_cast<TileBufType>(ty); tileTy &&
      tileTy.getBLayoutValueI32() != 0) {
    return op->emitOpError() << "expects " << name << " to use row-major layout";
  }
  return success();
}

static LogicalResult verifyTColExpandSrc1ValidCols(Operation *op, Type t1,
                                                   Type td) {
  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == 2 && dstValid.size() == 2 &&
      src1Valid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic &&
      src1Valid[1] != dstValid[1]) {
    return op->emitOpError(
        "expects src1 valid_shape[1] to equal dst valid_shape[1]");
  }
  return success();
}

static LogicalResult verifyTColExpandShapeAndLayout(Operation *op, Type t0,
                                                    Type t1, Type td) {
  if (getShapeVec(t0) != getShapeVec(td))
    return op->emitOpError("expects src0/dst to have same shape");
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")) ||
      failed(verifyTColExpandRowMajorLayout(op, t0, "src0")) ||
      failed(verifyTColExpandRowMajorLayout(op, t1, "src1")) ||
      failed(verifyTColExpandRowMajorLayout(op, td, "dst")) ||
      failed(verifyTColExpandSrc1ValidCols(op, t1, td)))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto elemOr = verifyTConcatCommon(*this);
  if (failed(elemOr))
    return failure();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    return verifyConcatElemType(getOperation(), *elemOr);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError("expects src0, src1, and dst to use row-major layout");
    }
    return verifyConcatElemType(getOperation(), *elemOr);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyConcatidxElementTypes(Operation *op, Type dataElem,
                                                 Type idxElem) {
  if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
    auto dataInt = dyn_cast<IntegerType>(dataElem);
    if (!dataInt || !dataInt.isSignless() ||
        (dataInt.getWidth() != 8 && dataInt.getWidth() != 16 &&
         dataInt.getWidth() != 32)) {
      return op->emitOpError(
          "expects data element type to be i8, i16, i32, f16, f32, or bf16");
    }
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || !idxInt.isSignless() ||
      (idxInt.getWidth() != 8 && idxInt.getWidth() != 16 &&
       idxInt.getWidth() != 32)) {
    return op->emitOpError(
        "expects index element type to be i8, i16, or i32");
  }
  return success();
}

static LogicalResult verifyTConcatidxValidShapes(TConcatidxOp op) {
  auto src0Valid = getValidShapeVec(op.getSrc0());
  auto src1Valid = getValidShapeVec(op.getSrc1());
  auto src0IdxValid = getValidShapeVec(op.getSrc0Idx());
  auto src1IdxValid = getValidShapeVec(op.getSrc1Idx());
  auto dstValid = getValidShapeVec(op.getDst());
  if (src0Valid.size() != 2 || src1Valid.size() != 2 ||
      src0IdxValid.size() != 2 || src1IdxValid.size() != 2 ||
      dstValid.size() != 2) {
    return op.emitOpError("expects all operands to have rank-2 valid_shape");
  }

  auto checkValidRow = [&](const auto &validShape,
                           StringRef name) -> LogicalResult {
    if (validShape[0] != ShapedType::kDynamic &&
        dstValid[0] != ShapedType::kDynamic && validShape[0] != dstValid[0]) {
      op.emitOpError("expects ")
          << name << " valid row to match dst valid row";
      return failure();
    }
    return success();
  };
  if (failed(checkValidRow(src0Valid, "src0")) ||
      failed(checkValidRow(src1Valid, "src1")) ||
      failed(checkValidRow(src0IdxValid, "src0Idx")) ||
      failed(checkValidRow(src1IdxValid, "src1Idx"))) {
    return failure();
  }
  if (src0IdxValid[1] != ShapedType::kDynamic && src0IdxValid[1] < 1)
    return op.emitOpError("expects src0Idx valid_col >= 1");
  if (src1IdxValid[1] != ShapedType::kDynamic && src1IdxValid[1] < 1)
    return op.emitOpError("expects src1Idx valid_col >= 1");
  return success();
}

static FailureOr<std::pair<Type, Type>> verifyTConcatidxCommon(TConcatidxOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type src0IdxTy = op.getSrc0Idx().getType();
  Type src1IdxTy = op.getSrc1Idx().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, src0IdxTy, "src0Idx")) ||
      failed(verifyTileBufCommon(op, src1IdxTy, "src1Idx")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }

  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !dstElem)
    return op.emitOpError("failed to get element type for data operands"),
           failure();
  if (src0Elem != src1Elem || src0Elem != dstElem) {
    return op.emitOpError(
               "expects src0, src1, and dst to have the same element type"),
           failure();
  }

  Type src0IdxElem = getElemTy(src0IdxTy);
  Type src1IdxElem = getElemTy(src1IdxTy);
  if (!src0IdxElem || !src1IdxElem) {
    return op.emitOpError("failed to get element type for index operands"),
           failure();
  }
  if (src0IdxElem != src1IdxElem) {
    return op.emitOpError(
               "expects src0Idx and src1Idx to have the same element type"),
           failure();
  }

  if (failed(verifyTConcatidxValidShapes(op)))
    return failure();
  return std::make_pair(src0Elem, src0IdxElem);
}

LogicalResult pto::TConcatidxOp::verify() {
  auto elemOr = verifyTConcatidxCommon(*this);
  if (failed(elemOr))
    return failure();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVecType(getOperation(), getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    return verifyConcatidxElementTypes(getOperation(), elemOr->first,
                                       elemOr->second);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyLocVecType(getOperation(), getSrc0().getType(), "src0")) ||
        failed(verifyLocVecType(getOperation(), getSrc1().getType(), "src1")) ||
        failed(verifyLocVecType(getOperation(), getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVecType(getOperation(), getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVecType(getOperation(), getDst().getType(), "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getSrc0Idx().getType()) ||
        !isRowMajorTileBuf(getSrc1Idx().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError("expects all operands to use row-major layout");
    }
    return verifyConcatidxElementTypes(getOperation(), elemOr->first,
                                       elemOr->second);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  return verifyDistinctRowMajorUnaryIntWidthOp(
      getOperation(), getSrc(), getDst(), "src", "dst",
      "expects A2/A3 tands src, scalar, and dst element type to be i8/i16",
      "expects A5 tands src, scalar, and dst element type to be i8/i16/i32");
}

LogicalResult pto::TCIOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy)
    return emitOpError("expects dst element type to be integer");

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32)
    return emitOpError("expects dst element type to be i16/i32");

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy)
    return emitOpError("expects S to be integer");

  if (sTy != elemTy)
    return emitOpError("expects S and dst element type to be exactly the same type");
  auto shape = getShapeVec(dstTy);
  if (shape.size() != 2)
    return emitOpError("expects dst to be rank-2");
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1)
    return emitOpError("expects dst cols to be different from 1");

  return success();
}

LogicalResult pto::TTriOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy)
    return emitOpError("expects diagonal to be an integer operand");

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1)
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false))
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        return success();
      },
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true))
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        return success();
      });
}

static LogicalResult verifyTCmpA2A3(TCmpOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, src0Ty, "src0")) ||
      failed(verifyVecTileStorage(op, src1Ty, "src1")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst"))) {
    return failure();
  }
  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !dstElem)
    return op.emitOpError("failed to get element type for src0/src1/dst");
  if (src0Elem != src1Elem)
    return op.emitOpError("expects src0 and src1 to have the same element type");
  if (!(src0Elem.isInteger(32) || src0Elem.isF16() || src0Elem.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tcmp input element type to be i32/f16/f32");
  }
  if (!dstElem.isInteger(8))
    return op.emitOpError("expects dst element type to be i8");

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError(
        "expects src0, src1, and dst to have rank-2 valid_shape");
  }
  if (!hasCompatibleKnownExtent(src0Valid[0], src1Valid[0]))
    return op.emitOpError("expects src0 and src1 to have the same valid row");
  if (!hasCompatibleKnownExtent(src0Valid[1], src1Valid[1]))
    return op.emitOpError(
        "expects src0 and src1 to have the same valid column");
  if (!hasCompatibleKnownExtent(src0Valid[0], dstValid[0]))
    return op.emitOpError("expects src0 valid row to equal dst valid row");
  return success();
}

static LogicalResult verifyTCmpA5(TCmpOp op) {
  auto verifyTileBufOperand = [](Operation *op, Type ty, StringRef name) {
    return verifyTileBufCommon(op, ty, name);
  };
  auto infoOr = verifyBinaryTileTypeInfo(op, op.getSrc0(), op.getSrc1(),
                                         op.getDst(), verifyTileBufOperand);
  if (failed(infoOr))
    return failure();
  const auto &info = *infoOr;
  if (info.src0Elem != info.src1Elem)
    return op.emitOpError("expects src0 and src1 to have the same element type");
  if (!(info.src0Elem.isF16() || info.src0Elem.isF32() ||
        info.src0Elem.isBF16() || info.src0Elem.isInteger(8) ||
        info.src0Elem.isInteger(16) || info.src0Elem.isInteger(32))) {
    return op.emitOpError(
        "expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(info.dstElem);
  if (!dstInt || dstInt.getWidth() != 8)
    return op.emitOpError("expects dst element type to be i8");
  if (getShapeVec(info.src0Ty) != getShapeVec(info.src1Ty) ||
      getShapeVec(info.src0Ty) != getShapeVec(info.dstTy)) {
    return op.emitOpError("expects src0, src1, and dst to have the same shape");
  }
  return success();
}

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTCmpA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTCmpA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
static LogicalResult verifyTCmpSCommon(TCmpSOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst"))) {
    return failure();
  }
  if (!op.getScalar().getType().isIntOrIndexOrFloat())
    return op.emitOpError("expects scalar to be integer, index, or float");
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (srcValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  return success();
}

static LogicalResult verifyTCmpSA2A3(TCmpSOp op) {
  if (failed(verifyTCmpSCommon(op)))
    return failure();
  Type elemTy = getElemTy(op.getSrc().getType());
  if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
        elemTy.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tcmps input element type to be i16/i32/f16/f32");
  }
  return success();
}

static LogicalResult verifyTCmpSA5(TCmpSOp op) {
  if (failed(verifyTCmpSCommon(op)))
    return failure();
  Type elemTy = getElemTy(op.getSrc().getType());
  if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
        elemTy.isF16() || elemTy.isF32())) {
    return op.emitOpError(
        "expects A5 tcmps input element type to be i8/i16/i32/f16/f32");
  }
  return success();
}

LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTCmpSA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTCmpSA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true))
    return emitOpError("expects tcolexpand element type to be supported");
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for src0/src1/dst");

  auto isSupportedElem = [&](Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32())
      return true;
    if (!allowIntegerTypes)
      return false;
    if (elemTy.isInteger(16) || elemTy.isInteger(32))
      return true;
    return targetArch == PTOArch::A5 && elemTy.isInteger(8);
  };
  if (!isSupportedElem(e0) || !isSupportedElem(e1) || !isSupportedElem(ed)) {
    if (!allowIntegerTypes)
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    if (targetArch == PTOArch::A5)
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (failed(verifyTColExpandShapeAndLayout(op, t0, t1, td)))
    return failure();

  return success();
}
LogicalResult pto::TColExpandMulOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmul",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandAddOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandadd",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    bool allowIntegerTypes = (targetArch == PTOArch::A5);
    return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        targetArch, "tcolexpanddiv",
                                        /*allowIntegerTypes=*/allowIntegerTypes);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandSubOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandsub",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandExpdifOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandexpdif",
                                      /*allowIntegerTypes=*/false);
}
LogicalResult pto::TColExpandMaxOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmax",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandMinOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmin",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColMaxOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmax element type to be f16/f32/i16/i32",
      "expects A5 tcolmax element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTColArgReductionOpCommon(*this, getSrc().getType(),
                                          getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

LogicalResult pto::TColMinOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmin element type to be f16/f32/i16/i32",
      "expects A5 tcolmin element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTColArgReductionOpCommon(*this, getSrc().getType(),
                                          getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}



static ParseResult parseTColSumFormatWithTmp(OpAsmParser &parser,
                                             OperationState &result,
                                             Type &srcTy, Type &tmpTy) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColonType(srcTy) || parser.parseComma() ||
      parser.parseType(tmpTy))
    return failure();
  return success();
}

static ParseResult parseTColSumFormatWithoutTmp(OpAsmParser &parser, Type &srcTy) {
  return parser.parseColonType(srcTy);
}

static ParseResult parseTColSumInsClause(OpAsmParser &parser, OperationState &result,
                                         OpAsmParser::UnresolvedOperand &src,
                                         OpAsmParser::UnresolvedOperand &tmp,
                                         Type &srcTy, Type &tmpTy, bool &hasTmp) {
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();

  if (failed(parser.parseOptionalComma())) {
    if (parseTColSumFormatWithoutTmp(parser, srcTy))
      return failure();
    return success();
  }

  if (parser.parseOperand(tmp))
    return failure();
  hasTmp = true;
  return parseTColSumFormatWithTmp(parser, result, srcTy, tmpTy);
}

static ParseResult parseTColSumOutsClause(OpAsmParser &parser,
                                          OpAsmParser::UnresolvedOperand &dst,
                                          Type &dstTy) {
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  return success();
}

static LogicalResult verifyTColSumCommon(TColSumOp op, bool requireNonZeroSrc,
                                         bool allowInt8, bool allowBf16,
                                         StringRef errorMessage) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst")))
    return failure();
  bool hasTmp = static_cast<bool>(op.getTmp());
  bool hasIsBinary = static_cast<bool>(op.getIsBinaryAttr());
  if (hasTmp != hasIsBinary) {
    if (hasTmp)
      return op.emitOpError("tmp operand requires isBinary attribute");
    return op.emitOpError("isBinary attribute requires tmp operand");
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyNDStyleVecTile(op, tmpTy, "tmp")))
      return failure();
    if (getElemTy(srcTy) != getElemTy(dstTy) ||
        getElemTy(srcTy) != getElemTy(tmpTy))
      return op.emitOpError("expects src/tmp/dst element types to match");
  }
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError("expects src/dst element types to match");
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc)))
    return failure();
  Type elem = getElemTy(srcTy);
  if (!(elem.isF16() || elem.isF32() || (allowBf16 && elem.isBF16()) ||
        elem.isInteger(16) || elem.isInteger(32) ||
        (allowInt8 && elem.isInteger(8))))
    return op.emitOpError(errorMessage);
  return success();
}

ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parseTColSumInsClause(parser, result, src, tmp, srcTy, tmpTy, hasTmp) ||
      parseTColSumOutsClause(parser, dst, dstTy))
    return failure();

  if (!hasTmp && parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, 1> elidedAttrs;
    if (!getIsBinaryAttr() || getIsBinaryAttr().getValue() == false) {
      elidedAttrs.push_back("isBinary");
    }
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : " << getSrc().getType() << ", " << getTmp().getType() << ")";
  } else {
    // Format 1: ins(%src : type) outs(%dst : type)
    p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  }

  p << " outs(" << getDst() << " : " << getDst().getType() << ")";

  // Print remaining attributes for format 1 (excluding isBinary)
  if (!getTmp()) {
    SmallVector<StringRef, 1> elidedAttrs = {"isBinary"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColSumCommon(*this, /*requireNonZeroSrc=*/false,
                               /*allowInt8=*/false, /*allowBf16=*/false,
                               "expects A2/A3 tcolsum element type to be "
                               "f16/f32/i16/i32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColSumCommon(*this, /*requireNonZeroSrc=*/true,
                               /*allowInt8=*/true, /*allowBf16=*/true,
                               "expects A5 tcolsum element type to be "
                               "i8/i16/i32/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColProdOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/false,
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolprod element type to be f16/f32/i16/i32",
      "expects A5 tcolprod element type to be i16/ui16/i32/ui32/f16/bf16/f32");
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true)))
    return failure();
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false)))
    return failure();
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem))
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (shouldBypassDecodedMemrefVerifier(getOperation()))
      return success();

    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32))
      return emitOpError("expects dst element type to be i32 or ui32");

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32)
        return emitOpError() << "expects " << name << " to be i32/ui32";
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3")))
      return failure();

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10)
      return emitOpError("expects rounds to be 7 or 10");

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
