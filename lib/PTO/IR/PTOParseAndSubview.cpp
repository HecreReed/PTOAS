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

static ParseResult parseKeywordedOperand(OpAsmParser &parser, StringRef keyword,
                                         OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseKeyword(keyword) || parser.parseLParen() ||
      parser.parseOperand(operand))
    return failure();
  return success();
}

static ParseResult parseRequiredCommaOperand(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand) {
  if (parser.parseComma() || parser.parseOperand(operand))
    return failure();
  return success();
}

static ParseResult parseOptionalCommaOperand(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    bool &isPresent) {
  if (failed(parser.parseOptionalComma()))
    return success();
  if (parser.parseOperand(operand))
    return failure();
  isPresent = true;
  return success();
}

static ParseResult parseOptionalCommaType(OpAsmParser &parser, bool isPresent,
                                          Type &type) {
  if (!isPresent)
    return success();
  if (parser.parseComma() || parser.parseType(type))
    return failure();
  return success();
}

static ParseResult parseOutsClauseWithOptionalAttrDict(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &dst, Type &dstTy) {
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

static ParseResult resolveOptionalOperand(
    OpAsmParser &parser, bool isPresent, OpAsmParser::UnresolvedOperand &operand,
    Type type, SmallVectorImpl<Value> &operands) {
  if (!isPresent)
    return success();
  return parser.resolveOperand(operand, type, operands);
}

static void addOperandSegmentSizesAttr(OpAsmParser &parser, OperationState &result,
                                       ArrayRef<int32_t> sizes) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(sizes));
}

ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, dstTy, idxTy, tmpTy;
  bool hasTmp = false;

  if (parseKeywordedOperand(parser, "ins", src))
    return failure();
  if (parseRequiredCommaOperand(parser, idx))
    return failure();
  if (parseOptionalCommaOperand(parser, tmp, hasTmp))
    return failure();
  if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(idxTy))
    return failure();
  if (parseOptionalCommaType(parser, hasTmp, tmpTy))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parseOutsClauseWithOptionalAttrDict(parser, result, dst, dstTy))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands))
    return failure();
  if (resolveOptionalOperand(parser, hasTmp, tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  addOperandSegmentSizesAttr(parser, result, {1, 1, hasTmp ? 1 : 0, 1});
  return success();
}

void mlir::pto::TSort32Op::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx();
  if (getTmp()) {
    p << ", " << getTmp();
    p << " : " << getSrc().getType() << ", " << getIdx().getType()
      << ", " << getTmp().getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getIdx().getType() << ")";
  }
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parseKeywordedOperand(parser, "ins", src))
    return failure();
  if (parseOptionalCommaOperand(parser, tmp, hasTmp))
    return failure();
  if (parser.parseColonType(srcTy))
    return failure();
  if (parseOptionalCommaType(parser, hasTmp, tmpTy))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parseOutsClauseWithOptionalAttrDict(parser, result, dst, dstTy))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (resolveOptionalOperand(parser, hasTmp, tmp, tmpTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TRsqrtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp())
    p << ", " << getTmp();
  p << " : " << getSrc().getType();
  if (getTmp())
    p << ", " << getTmp().getType();
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

static ParseResult parseTRowExpandBinaryLikeOp(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;

  if (parseKeywordedOperand(parser, "ins", src0) ||
      parseRequiredCommaOperand(parser, src1))
    return failure();
  if (parseOptionalCommaOperand(parser, tmp, hasTmp))
    return failure();
  if (parser.parseColon())
    return failure();
  if (parser.parseType(src0Ty) || parser.parseComma() || parser.parseType(src1Ty))
    return failure();
  if (parseOptionalCommaType(parser, hasTmp, tmpTy))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parseOutsClauseWithOptionalAttrDict(parser, result, dst, dstTy))
    return failure();

  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands))
    return failure();
  if (resolveOptionalOperand(parser, hasTmp, tmp, tmpTy, result.operands))
    return failure();
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  addOperandSegmentSizesAttr(parser, result, {1, 1, hasTmp ? 1 : 0, 1});
  return success();
}

static void printTRowExpandBinaryLikeOp(OpAsmPrinter &p, Operation *op, Value src0,
                                        Value src1, Value tmp, Value dst) {
  p << " ins(" << src0 << ", " << src1;
  if (tmp) {
    p << ", " << tmp;
    p << " : " << src0.getType() << ", " << src1.getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << src0.getType() << ", " << src1.getType() << ")";
  }
  p << " outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRowExpandDivOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandDivOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMulOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandSubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandSubOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandExpdifOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandExpdifOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMaxOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMaxOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMinOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMinOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

static FailureOr<Type> verifyTRowExpandBinaryCore(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (getElemTy(src0Ty) != getElemTy(src1Ty)) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static bool isTRowExpandBinaryElemSupported(Type elem, PTOArch targetArch,
                                            bool allowA2A3IntegerTypes) {
  if (elem.isF16() || elem.isF32())
    return true;
  if (targetArch == PTOArch::A5)
    return elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32);
  return allowA2A3IntegerTypes &&
         (elem.isInteger(16) || elem.isInteger(32));
}

static LogicalResult verifyTRowExpandBinaryElemType(Operation *op, Type elem,
                                                    PTOArch targetArch,
                                                    bool allowA2A3IntegerTypes,
                                                    StringRef a2a3Message,
                                                    StringRef a5Message) {
  if (isTRowExpandBinaryElemSupported(elem, targetArch,
                                      allowA2A3IntegerTypes))
    return success();
  if (targetArch == PTOArch::A5)
    return op->emitOpError(a5Message);
  return op->emitOpError(a2a3Message);
}

static LogicalResult verifyTRowExpandBinaryLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  bool allowA2A3IntegerTypes,
                                                  StringRef a2a3Message,
                                                  StringRef a5Message) {
  FailureOr<Type> elemOr =
      verifyTRowExpandBinaryCore(op, src0Ty, src1Ty, dstTy, tmpTy, hasTmp);
  if (failed(elemOr))
    return failure();
  return verifyTRowExpandBinaryElemType(op, *elemOr, targetArch,
                                        allowA2A3IntegerTypes, a2a3Message,
                                        a5Message);
}

static LogicalResult verifyTRowExpandAddSrc1Shape(Operation *op, Type src1Ty,
                                                  Type dstTy, Type elem) {
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError(
        "expects src1 and dst to have rank-2 valid_shape");
  if (src1Valid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && src1Valid[0] != dstValid[0])
    return op->emitOpError(
        "expects src1 valid_shape[0] to equal dst valid_shape[0]");
  bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
  int64_t expectedCol =
      elem.isInteger(8) ? 32
                        : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t src1Col = src1Valid[1];
  if (src1IsRowMajor) {
    if (src1Col != ShapedType::kDynamic && src1Col != expectedCol)
      return op->emitOpError(
          "expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    return success();
  }
  if (src1Col != ShapedType::kDynamic && src1Col != 1)
    return op->emitOpError(
        "expects non-row-major src1 valid_shape[1] to be 1");
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    return verifyTRowExpandBinaryLikeOp(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType(),
        getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
        targetArch, /*allowA2A3IntegerTypes=*/false,
        "expects element type to be f16 or f32",
        "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    return verifyTRowExpandBinaryLikeOp(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType(),
        getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
        targetArch, /*allowA2A3IntegerTypes=*/true,
        "expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32",
        "expects A5 trowexpandmul element type to be i8/i16/i32/f16/f32");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    return verifyTRowExpandBinaryLikeOp(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType(),
        getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
        targetArch, /*allowA2A3IntegerTypes=*/true,
        "expects A2/A3 trowexpandsub element type to be i16/i32/f16/f32",
        "expects A5 trowexpandsub element type to be i8/i16/i32/f16/f32");
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr =
        verifyTRowExpandBinaryCore(*this, src0Ty, src1Ty, dstTy, Type{}, false);
    if (failed(elemOr))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (!isRowMajorTileBuf(src0Ty))
      return emitOpError("expects src0 to use row-major layout");
    if (failed(verifyTRowExpandBinaryElemType(
            *this, *elemOr, targetArch, /*allowA2A3IntegerTypes=*/true,
            "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32",
            "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32")))
      return failure();
    return verifyTRowExpandAddSrc1Shape(*this, src1Ty, dstTy, *elemOr);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTRowExpandReduceTypes(Operation *op, Type src0Ty,
                                                 Type src1Ty, Type dstTy,
                                                 Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (!hasTmp)
    return success();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return op->emitOpError()
           << "expects tmp and dst to have the same element type";
  return success();
}

static LogicalResult verifyTRowExpandReduceElementType(Operation *op, Type src0Ty,
                                                       Type src1Ty, Type dstTy,
                                                       PTOArch targetArch,
                                                       StringRef opName,
                                                       bool allowIntegerTypes,
                                                       Type &elem) {
  elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem)
    return op->emitOpError(
        "expects src0, src1, and dst to have the same element type");
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (supported)
    return success();
  if (!allowIntegerTypes)
    return op->emitOpError() << "expects " << opName
                             << " element type to be f16 or f32";
  if (targetArch == PTOArch::A5)
    return op->emitOpError() << "expects A5 " << opName
                             << " element type to be i8/i16/i32/f16/f32";
  return op->emitOpError() << "expects A2/A3 " << opName
                           << " element type to be i16/i32/f16/f32";
}

static bool validShapeMatches(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r)
      return false;
  }
  return true;
}

static LogicalResult verifyNonZeroRank2ValidShape(Operation *op,
                                                  ArrayRef<int64_t> valid,
                                                  StringRef name) {
  if (valid.size() != 2)
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  if (valid[0] != ShapedType::kDynamic && valid[0] == 0)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be non-zero";
  if (valid[1] != ShapedType::kDynamic && valid[1] == 0)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[1] to be non-zero";
  return success();
}

static LogicalResult verifyTRowExpandBroadcastOperand(
    Operation *op, Type elem, Type operandTy, ArrayRef<int64_t> operandValid,
    ArrayRef<int64_t> dstValid, StringRef operandName,
    bool requireNonRowMajor) {
  if (operandValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && operandValid[0] != dstValid[0]) {
    return op->emitOpError() << "expects " << operandName
                             << " valid_shape[0] to equal dst valid_shape[0]";
  }
  int64_t expectedCol =
      elem.isInteger(8) ? 32 : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t operandCol = operandValid[1];
  bool operandIsRowMajor = isRowMajorTileBuf(operandTy);
  if (requireNonRowMajor && operandIsRowMajor) {
    return op->emitOpError()
           << "expects " << operandName
           << " to use a non-row-major layout when tmp is present";
  }
  if (operandIsRowMajor) {
    if (operandCol != ShapedType::kDynamic && operandCol != expectedCol) {
      return op->emitOpError()
             << "expects row-major " << operandName
             << " valid_shape[1] to be 32/sizeof(dtype)";
    }
    return success();
  }
  if (operandCol != ShapedType::kDynamic && operandCol != 1) {
    return op->emitOpError() << "expects non-row-major " << operandName
                             << " valid_shape[1] to be 1";
  }
  return success();
}

static LogicalResult verifyTRowExpandFullAndBroadcast(
    Operation *op, Type elem, ArrayRef<int64_t> dstValid, Type fullTy,
    ArrayRef<int64_t> fullValid, StringRef fullName, Type broadcastTy,
    ArrayRef<int64_t> broadcastValid, StringRef broadcastName,
    bool requireNonRowMajorBroadcast) {
  if (!isRowMajorTileBuf(fullTy))
    return op->emitOpError() << "expects " << fullName
                             << " to use row-major layout when it matches dst";
  if (fullValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && fullValid[0] != dstValid[0])
    return op->emitOpError() << "expects " << fullName
                             << " valid_shape[0] to equal dst valid_shape[0]";
  if (fullValid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic && fullValid[1] != dstValid[1])
    return op->emitOpError() << "expects " << fullName
                             << " valid_shape[1] to equal dst valid_shape[1]";
  return verifyTRowExpandBroadcastOperand(op, elem, broadcastTy, broadcastValid,
                                          dstValid, broadcastName,
                                          requireNonRowMajorBroadcast);
}

static LogicalResult verifyTRowExpandReduceLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (failed(verifyTRowExpandReduceTypes(op, src0Ty, src1Ty, dstTy, tmpTy,
                                         hasTmp)))
    return failure();
  Type elem;
  if (failed(verifyTRowExpandReduceElementType(op, src0Ty, src1Ty, dstTy,
                                               targetArch, opName,
                                               allowIntegerTypes, elem)))
    return failure();
  if (!isRowMajorTileBuf(dstTy))
    return op->emitOpError("expects dst to use row-major layout");

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError(
        "expects src0, src1, and dst to have rank-2 valid_shape");
  if (failed(verifyNonZeroRank2ValidShape(op, dstValid, "dst")))
    return failure();

  const bool src0MatchesDst = validShapeMatches(src0Valid, dstValid);
  const bool src1MatchesDst = validShapeMatches(src1Valid, dstValid);
  if (hasTmp && targetArch == PTOArch::A5)
    return op->emitOpError("expects A5 form to omit tmp");
  const bool requireNonRowMajorBroadcast =
      hasTmp && targetArch == PTOArch::A3;

  if (src0MatchesDst &&
      succeeded(verifyTRowExpandFullAndBroadcast(
          op, elem, dstValid, src0Ty, src0Valid, "src0", src1Ty, src1Valid,
          "src1", requireNonRowMajorBroadcast)))
    return success();
  if (src1MatchesDst &&
      succeeded(verifyTRowExpandFullAndBroadcast(
          op, elem, dstValid, src1Ty, src1Valid, "src1", src0Ty, src0Valid,
          "src0", requireNonRowMajorBroadcast)))
    return success();

  return op->emitOpError()
         << "expects one of src0/src1 to match dst valid_shape"
         << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionNoTmpCommon(*this, getSrc().getType(),
                                          getDst().getType(),
                                          "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowArgReductionCommon(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowArgReductionCommon(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionNoTmpCommon(*this, getSrc().getType(),
                                          getDst().getType(),
                                          "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32()))
    return emitOpError("expects element type to be f16 or f32");
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value())
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    if (tmpElemBytes.value() * tmpNumel.value() < 32)
      return emitOpError("expects tmp to be at least 32 bytes when provided");
  }
  return mlir::success();
}

static bool isScatterAllowedDataElem(Type type) {
  if (type.isF16() || type.isF32() || type.isBF16())
    return true;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32;
  return false;
}

static bool isScatterAllowedIndexElem(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 16 || intTy.getWidth() == 32;
  return false;
}

static unsigned getMaskScatterTimes(MaskPatternAttr pattern) {
  switch (pattern.getValue()) {
  case MaskPattern::P1111:
    return 1;
  case MaskPattern::P0101:
  case MaskPattern::P1010:
    return 2;
  default:
    return 4;
  }
}

static LogicalResult verifyTScatterIndexedForm(TScatterOp op) {
  Type srcTy = op.getSrc().getType();
  Type indexTy = op.getIndexes().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, indexTy, "indexes")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type indexElem = getElemTy(indexTy);
  if (!srcElem || !dstElem || !indexElem)
    return op.emitOpError("failed to get element type for operands");
  if (srcElem != dstElem)
    return op.emitOpError("expects src/dst to have the same element type");
  if (!isScatterAllowedDataElem(srcElem))
    return op.emitOpError(
        "expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  if (!isScatterAllowedIndexElem(indexElem))
    return op.emitOpError("expects indexes element type to be i16/i32");

  auto dataWidth = getPTOStorageElemBitWidth(srcElem);
  auto indexWidth = getPTOStorageElemBitWidth(indexElem);
  if (dataWidth != 8 && dataWidth != 16 && dataWidth != 32)
    return op.emitOpError("unexpected src/dst element bitwidth");

  unsigned dataBytes = dataWidth / 8;
  unsigned expectedIndexBytes = dataBytes == 1 ? 2 : dataBytes;
  if (indexWidth / 8 != expectedIndexBytes) {
    return op.emitOpError(
        "expects indexes element size to match the documented scatter rule");
  }
  return success();
}

static LogicalResult verifyTScatterMaskForm(TScatterOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, dstTy, "dst")))
    return failure();

  auto srcTile = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTile = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTile || !dstTile)
    return op.emitOpError("expects src and dst to be tile_buf types");
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError("expects src and dst to have the same element type");
  if (!isScatterAllowedDataElem(getElemTy(srcTy)))
    return op.emitOpError(
        "expects src/dst element type to be i8/i16/i32/f16/bf16/f32");

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");

  auto pattern = op.getMaskPatternAttr();
  if (!pattern)
    return op.emitOpError(
        "expects mask-pattern tscatter to provide maskPattern");
  const unsigned times = getMaskScatterTimes(pattern);
  if (srcValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && srcValid[0] != dstValid[0])
    return op.emitOpError("expects src and dst to have the same valid rows");
  if (srcValid[1] != ShapedType::kDynamic &&
      dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != static_cast<int64_t>(dstValid[1] * times)) {
    return op.emitOpError(
        "expects src valid cols to equal dst valid cols times the mask expansion factor");
  }

  if (srcTile.getBLayoutValueI32() !=
          static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTile.getBLayoutValueI32() !=
          static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError(
        "expects mask-pattern tscatter to use row_major blayout");
  }
  return success();
}


mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (hasMaskPattern)
      return verifyTScatterMaskForm(*this);
    return verifyTScatterIndexedForm(*this);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (hasMaskPattern)
      return emitOpError("mask-pattern tscatter is not supported on A5 yet");
    return verifyTScatterIndexedForm(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifySelectElementType(Operation *op, Type elem,
                                             PTOArch targetArch,
                                             bool allowBf16,
                                             StringRef a2a3Message,
                                             StringRef a5Message) {
  bool ok = elem.isF16() || elem.isF32() || (allowBf16 && elem.isBF16());
  if (auto intTy = dyn_cast<IntegerType>(elem))
    ok = intTy.getWidth() == 16 || intTy.getWidth() == 32 ||
         (targetArch == PTOArch::A5 && intTy.getWidth() == 8);
  if (ok)
    return success();
  if (targetArch == PTOArch::A5)
    return op->emitOpError(a5Message);
  return op->emitOpError(a2a3Message);
}

static FailureOr<Type> verifyTSelCommon(TSelOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !src1Elem || !dstElem) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (srcElem != src1Elem || srcElem != dstElem) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op.emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return srcElem;
}


mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A3, /*allowBf16=*/true,
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32",
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A5, /*allowBf16=*/true,
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32",
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static FailureOr<Type> verifyTSelSCommon(TSelSOp op) {
  Type maskTy = op.getMask().getType();
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, maskTy, "mask")) ||
      failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type maskElem = getElemTy(maskTy);
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = getElemTy(tmpTy);
  Type dstElem = getElemTy(dstTy);
  if (!maskElem || !srcElem || !tmpElem || !dstElem) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (srcElem != dstElem) {
    op.emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op.emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  return dstElem;
}


mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels:
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelSCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A3, /*allowBf16=*/false,
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32",
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyTSelSCommon(*this);
    if (failed(elemOr))
      return failure();
    return verifySelectElementType(
        *this, *elemOr, PTOArch::A5, /*allowBf16=*/false,
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32",
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx")))
    return failure();
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!(srcElem.isF16() || srcElem.isF32()))
    return emitOpError() << "expects src and dst element type to be f16 or f32";

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32)
    return emitOpError() << "expects idx element type to be i32/u32";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem)))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}

static bool shouldBypassTStoreFPVerifier(TStoreFPOp op) {
  Value src = op.getSrc();
  Value fp = op.getFp();
  return isa<MemRefType>(src.getType()) || isa<MemRefType>(fp.getType()) ||
         src.getDefiningOp<pto::BindTileOp>() ||
         fp.getDefiningOp<pto::BindTileOp>();
}

static LogicalResult verifyTStoreFPDstType(TStoreFPOp op) {
  Type dstTy = op.getDst().getType();
  if (!isa<MemRefType, pto::PartitionTensorViewType>(dstTy))
    return op.emitOpError()
           << "expects dst to be a memref or !pto.partition_tensor_view";
  if (auto dstPart = dyn_cast<pto::PartitionTensorViewType>(dstTy)) {
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        return op.emitOpError()
               << "expects dst shape[" << idx << "] to be positive";
      }
    }
  }
  return success();
}

static LogicalResult verifyTStoreFPTileOperands(TStoreFPOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  if (!isa<pto::TileBufType>(srcTy))
    return op.emitOpError() << "expects src to be a !pto.tile_buf";
  if (!isa<pto::TileBufType>(fpTy))
    return op.emitOpError() << "expects fp to be a !pto.tile_buf";
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")))
    return failure();
  if (failed(verifyTStoreFPDstType(op)))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
    return op.emitOpError() << "expects src to be in the acc address space";
  return success();
}

static LogicalResult verifyTStoreFPA2A3Constraints(TStoreFPOp op) {
  Type srcTy = op.getSrc().getType();
  auto srcElemTy = getElemTy(srcTy);
  auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
  if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32)))
    return op.emitOpError() << "expects src to have element type f32, i32";
  auto srcShape = getShapeVec(srcTy);
  if (srcShape.size() != 2)
    return op.emitOpError() << "expects src to have rank 2";
  if (srcShape[1] != ShapedType::kDynamic &&
      (srcShape[1] < 1 || srcShape[1] > 4095))
    return op.emitOpError() << "expects src.cols to be in the range [1, 4095]";
  auto srcValid = getValidShapeVec(srcTy);
  if (srcValid.size() != 2)
    return op.emitOpError() << "expects src to have a rank-2 valid_shape";
  if (srcValid[1] != ShapedType::kDynamic &&
      (srcValid[1] < 1 || srcValid[1] > 4095)) {
    return op.emitOpError()
           << "expects src.valid_shape[1] to be in the range [1, 4095]";
  }
  return success();
}



mlir::LogicalResult mlir::pto::TStoreFPOp::verify() {
  if (shouldBypassTStoreFPVerifier(*this))
    return success();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTStoreFPTileOperands(*this)))
      return failure();
    return verifyTStoreFPA2A3Constraints(*this);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTStoreFPTileOperands(*this);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}


mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size())
    return emitOpError() << "expects all tensors to have the same rank";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  return mlir::success();
}

struct TTransVerifyState {
  Type srcTy;
  Type dstTy;
  unsigned elemBytes;
};

static bool isSupportedTransposeElemType(Type type, unsigned elemBytes) {
  if (elemBytes == 4)
    return type.isInteger(32) || type.isF32();
  if (elemBytes == 2)
    return type.isInteger(16) || type.isF16() || type.isBF16();
  return type.isInteger(8);
}

static FailureOr<TTransVerifyState>
verifyTTransCommon(TTransOp op, StringRef mismatchMessage) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = getElemTy(tmpTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem ||
      srcElem != tmpElem) {
    op.emitOpError() << mismatchMessage;
    return failure();
  }
  unsigned elemBytes = getPTOStorageElemByteSize(srcElem);
  if (elemBytes == 0) {
    op.emitOpError() << "failed to get transpose element size";
    return failure();
  }
  if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4) {
    op.emitOpError()
        << "expects transpose element size to be 1, 2, or 4 bytes";
    return failure();
  }
  if (!isSupportedTransposeElemType(srcElem, elemBytes)) {
    op.emitOpError()
        << "expects transpose element type to match the supported set for its width";
    return failure();
  }
  return TTransVerifyState{srcTy, dstTy, elemBytes};
}

static LogicalResult verifyTTransA2A3Constraints(TTransOp op,
                                                 const TTransVerifyState &state) {
  auto srcTile = dyn_cast<pto::TileBufType>(state.srcTy);
  if (!srcTile)
    return success();
  if (srcTile.getBLayoutValueI32() !=
      static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError()
           << "expects A2/A3 transpose src to use the row_major blayout";
  }
  return success();
}

static LogicalResult verifyTTransA5MajorAlignment(TTransOp op, Type type,
                                                  unsigned elemBytes,
                                                  StringRef name) {
  auto tile = dyn_cast<pto::TileBufType>(type);
  if (!tile)
    return success();
  auto shape = getShapeVec(type);
  if (shape.size() != 2)
    return success();
  bool rowMajor =
      tile.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[1] : shape[0];
  if (major != ShapedType::kDynamic &&
      (major * static_cast<int64_t>(elemBytes)) % 32 != 0) {
    return op.emitOpError()
           << "expects " << name
           << " major dimension times element size to be 32-byte aligned on A5";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<TTransVerifyState> stateOr =
        verifyTTransCommon(*this, "expects src and dst to have the same element type");
    if (failed(stateOr))
      return failure();
    return verifyTTransA2A3Constraints(*this, *stateOr);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<TTransVerifyState> stateOr = verifyTTransCommon(
        *this, "expects src, tmp, and dst to have the same element type");
    if (failed(stateOr))
      return failure();
    if (failed(verifyTTransA5MajorAlignment(*this, stateOr->srcTy,
                                           stateOr->elemBytes, "src")) ||
        failed(verifyTTransA5MajorAlignment(*this, stateOr->dstTy,
                                           stateOr->elemBytes, "dst")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyBase = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    Type elem = *elemOr;
    if (getElemTy(tmpTy) != elem)
      return emitOpError("expects tmp to have the same element type as src0, src1, and dst");
    if (!isRowMajorTileBuf(tmpTy))
      return emitOpError("expects tmp to use row-major layout");
    if (failed(verifyTileBufSameValidShape(*this, tmpTy, getDst().getType(), "tmp", "dst")))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };
  return verifyArchIntegerWidthOp(
      getOperation(), verifyCommon,
      "expects A2/A3 txors src and dst element type to be i8/i16",
      "expects A5 txors src and dst element type to be i8/i16/i32");
}
mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcType = getSrc().getType();
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError() << "expects printable tile element type";
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!space || *space != pto::AddressSpace::VEC)
      return emitOpError() << "expects printable tile_buf to be in vec address space";
    return success();
  }
  if (mlir::dyn_cast<MemRefType>(srcType) ||
      mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType))
    return mlir::success();
  return emitOpError() << "expects tile_buf, memref, or partition_tensor_view for src";
}



static LogicalResult verifyMatmulShapedCommon(Operation *op, ShapedType lhsTy,
                                              Value rhs, Value biasOpt,
                                              Type maybeDstElemTy,
                                              Type maybeResultElemTy) {
  auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
  if (!rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return op->emitOpError("expects lhs and rhs to be ranked tensors or memrefs");

  if (lhsTy.getElementType() != rhsTy.getElementType()) {
    return op->emitOpError()
           << "expects lhs and rhs to have the same element type, but got lhs="
           << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();
  }

  if (biasOpt) {
    auto biasTy = dyn_cast<ShapedType>(biasOpt.getType());
    if (!biasTy || !biasTy.hasRank())
      return op->emitOpError("expects bias to be a ranked tensor or memref");
    if (biasTy.getElementType() != lhsTy.getElementType()) {
      return op->emitOpError()
             << "expects bias to have the same element type as lhs and rhs, but got bias="
             << biasTy.getElementType() << " vs " << lhsTy.getElementType();
    }
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType()) {
    return op->emitOpError()
           << "expects dst to have the same element type as lhs and rhs, but got dst="
           << maybeDstElemTy << " vs " << lhsTy.getElementType();
  }
  if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType()) {
    return op->emitOpError()
           << "expects result to have the same element type as lhs and rhs, but got result="
           << maybeResultElemTy << " vs " << lhsTy.getElementType();
  }
  return success();
}

static LogicalResult verifyMatmulTileCommon(Operation *op, TileType lhsTile,
                                            Value rhs, Value biasOpt,
                                            Type maybeDstElemTy,
                                            Type maybeResultElemTy) {
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!rhsTile) {
    return op->emitOpError(
        "expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");
  }
  if (lhsTile.getElementType() != rhsTile.getElementType()) {
    return op->emitOpError()
           << "expects lhs and rhs tiles to have the same element type, but got lhs="
           << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();
  }
  if ((int64_t)lhsTile.getShape().size() != 2 ||
      (int64_t)rhsTile.getShape().size() != 2) {
    return op->emitOpError("expects lhs and rhs tiles to be 2D");
  }
  if (lhsTile.getShape()[1] != rhsTile.getShape()[0]) {
    return op->emitOpError()
           << "expects lhs dim1 to equal rhs dim0, but got "
           << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];
  }

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile)
      return op->emitOpError(
          "expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    if (biasTile.getElementType() != lhsTile.getElementType()) {
      return op->emitOpError(
          "expects bias to have the same element type as lhs and rhs");
    }
  }
  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType())
    return op->emitOpError()
           << "expects dst to have the same element type as lhs and rhs";
  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType())
    return op->emitOpError()
           << "expects result to have the same element type as lhs and rhs";
  return success();
}

[[maybe_unused]] static LogicalResult verifyMatmulCommon(Operation *op, Value lhs,
                                                         Value rhs, Value biasOpt,
                                                         Type maybeDstElemTy,
                                                         Type maybeResultElemTy) {
  if (auto lhsTy = dyn_cast<ShapedType>(lhs.getType())) {
    return verifyMatmulShapedCommon(op, lhsTy, rhs, biasOpt, maybeDstElemTy,
                                    maybeResultElemTy);
  }
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  if (!lhsTile) {
    return op->emitOpError(
        "expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");
  }
  return verifyMatmulTileCommon(op, lhsTile, rhs, biasOpt, maybeDstElemTy,
                                maybeResultElemTy);
}

using VerifyMatTileOperandsFn = LogicalResult (*)(Operation *, Type, Type, Type);

static LogicalResult verifyMatmulLikeTileOp(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy,
                                            VerifyMatTileOperandsFn verifyOperands) {
  if (failed(verifyOperands(op, lhsTy, rhsTy, dstTy)))
    return failure();
  if (failed(verifyMatmulTypeTriple(op, getElemTy(lhsTy), getElemTy(rhsTy),
                                    getElemTy(dstTy))))
    return failure();
  return verifyMatmulLike(op, lhsTy, rhsTy, dstTy);
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulLikeTileOp(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), verifyMatTileOperands);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulLikeTileOp(*this, getLhs().getType(), getRhs().getType(),
                                  getDst().getType(), verifyGemvTileOperands);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                   getDst().getType())))
    return failure();
  return success();
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < 2)
    return mlir::Type();

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile)
    return mlir::Type();

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    llvm::SmallVector<int64_t, 2> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2)
    return RankedTensorType();

  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  auto rhsTy = dyn_cast<ShapedType>(operands[1].getType());
  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return RankedTensorType();

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType()))
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    if (auto biasMR = dyn_cast<MemRefType>(operands[2].getType())) {
      if (biasMR.hasStaticShape())
        return RankedTensorType::get(biasMR.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty())
    return RankedTensorType();
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType()))
    return accRT;
  return RankedTensorType();
}

namespace mlir {
namespace pto {

static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess())
    return failure();

  if (parser.parseDimensionList(shape, allowDynamic))
    return failure();

  if (parser.parseType(elementType))
    return failure();

  if (parser.parseGreater())
    return failure();

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic)
      printer << "?";
    else
      printer << d;
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();

  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false)))
    return Type();
  return LocalArrayType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty())
    return emitError() << "'!pto.local_array' requires at least one dimension";
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0)
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
  }
  if (!elementType.isIntOrFloat())
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  return success();
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (add.getKind() == AffineExprKind::Add) {
      flattenAddExpr(add.getLHS(), terms);
      flattenAddExpr(add.getRHS(), terms);
      return;
    }
  }
  terms.push_back(expr);
}

static bool extractStrideFromMulExpr(AffineExpr lhs, AffineExpr rhs,
                                     unsigned &position, int64_t &stride) {
  auto dim = llvm::dyn_cast<AffineDimExpr>(lhs);
  auto constant = llvm::dyn_cast<AffineConstantExpr>(rhs);
  if (!dim || !constant)
    return false;
  position = dim.getPosition();
  stride = constant.getValue();
  return true;
}

static bool extractStrideFromAffineTerm(AffineExpr term, unsigned &position,
                                        int64_t &stride) {
  auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term);
  if (!mul || mul.getKind() != AffineExprKind::Mul)
    return false;
  return extractStrideFromMulExpr(mul.getLHS(), mul.getRHS(), position,
                                  stride) ||
         extractStrideFromMulExpr(mul.getRHS(), mul.getLHS(), position,
                                  stride);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);

  if (map.getNumResults() != 1)
    return;

  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    unsigned position = 0;
    int64_t stride = 0;
    if (extractStrideFromAffineTerm(term, position, stride)) {
      strides[position] = stride;
      continue;
    }
    if (auto dim = llvm::dyn_cast<AffineDimExpr>(term))
      strides[dim.getPosition()] = 1;
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures that the order of AffineExpr addition is:
//   0 + (d0*str0 + d1*str1...) + (s0*str0 + s1*str1...)
// This guarantees bitwise-identical AffineMaps for verification.
static AffineMap buildStrictBitwiseAffineMap(MLIRContext *ctx,
                                             ArrayRef<int64_t> strides,
                                             bool isMultiDimSymbol) {
  unsigned rank = strides.size();

  // Step 1: Initialize with Constant(0)
  AffineExpr totalExpr = getAffineConstantExpr(0, ctx);

  // Step 2: Add Dimensions (d0*str0 + d1*str1...)
  // Strictly in order: 0, 1, 2...
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = getAffineDimExpr(i, ctx);
    auto str = getAffineConstantExpr(strides[i], ctx);
    totalExpr = totalExpr + (dim * str);
  }

  // Step 3: Add Symbols (s0*str0 + s1*str1...)
  // Strictly in order: 0, 1, 2...
  if (isMultiDimSymbol) {
    for (unsigned i = 0; i < rank; ++i) {
      auto sym = getAffineSymbolExpr(i, ctx);
      auto str = getAffineConstantExpr(strides[i], ctx);
      totalExpr = totalExpr + (sym * str);
    }
  }
  // (Optional: handle single dynamic offset case if needed, omitted for clarity)

  // numSymbols is rank if multi-dim (for offsets), else 0
  unsigned numSymbols = isMultiDimSymbol ? rank : 0;
  return AffineMap::get(rank, numSymbols, totalExpr);
}


// =============================================================================
// Parser Implementation
// =============================================================================

// Helper for parsing [64, 1]
static ParseResult parseStrideList(AsmParser &parser, SmallVectorImpl<int64_t> &strides) {
  if (parser.parseLSquare()) return failure();
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) return failure();
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) return failure();
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
[[maybe_unused]] static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) return failure();

  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) return failure();

  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) return failure();

    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) return failure();
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) return failure();
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }

  if (parser.parseGreater()) return failure();

  // 3. Validation
  if (isMultiDim && numSymbols != strides.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "Number of offset symbols must match rank");
  }

  // 4. [CALL SHARED BUILDER]
  // Delegate to the strict builder
  MLIRContext *ctx = parser.getContext();
  AffineMap map = buildStrictBitwiseAffineMap(ctx, strides, isMultiDim);

  layout = AffineMapAttr::get(map);
  return success();
}

// =============================================================================
// Printer Implementation
// =============================================================================

[[maybe_unused]] static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) return;
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) return;

  // 1. [核心修改] 反解 Strides
  SmallVector<int64_t> strides;
  decomposeStridedLayout(map, strides);

  printer << ", strided<[";
  // 2. 打印真实的 strides
  llvm::interleaveComma(strides, printer);
  printer << "]";

  // Print Offset: [?, ?]
  unsigned numSyms = map.getNumSymbols();
  if (numSyms > 0) {
    printer << ", offset: [";
    for (unsigned i = 0; i < numSyms; ++i) {
      printer << "?";
      if (i < numSyms - 1) printer << ", ";
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subview 相关实现

// =============================================================================
// Op Interface Implementation: SubViewOp
// =============================================================================

static ParseResult parseSubViewSourceOffsetsAndSizes(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets) {
  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperandList(offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes")) {
    return failure();
  }
  ArrayAttr sizesAttr;
  return parser.parseAttribute(sizesAttr, "sizes", result.attributes);
}

static ParseResult parseSubViewValids(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &valids) {
  if (failed(parser.parseOptionalKeyword("valid")))
    return success();
  OpAsmParser::UnresolvedOperand rowValid;
  OpAsmParser::UnresolvedOperand colValid;
  if (parser.parseLSquare() || parser.parseOperand(rowValid) ||
      parser.parseComma() || parser.parseOperand(colValid) ||
      parser.parseRSquare()) {
    return failure();
  }
  valids.push_back(rowValid);
  valids.push_back(colValid);
  return success();
}

static ParseResult resolveSubViewSourceAndIndices(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source, Type sourceTy, Type &resultTy,
    bool &hasExplicitResultTy,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &valids) {
  if (parseOptionalArrowTypeAndResolveSource(parser, result, source, sourceTy,
                                             resultTy, hasExplicitResultTy))
    return failure();
  if (resolveIndexOperandsToResult(parser, offsets, result))
    return failure();
  if (!valids.empty() && resolveIndexOperandsToResult(parser, valids, result))
    return failure();
  return success();
}

static ParseResult finalizeSubViewResultTypes(OpAsmParser &parser,
                                              OperationState &result,
                                              Type resultTy,
                                              bool hasExplicitResultTy) {
  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  SmallVector<Type> inferredReturnTypes;
  DictionaryAttr attrs = result.attributes.getDictionary(parser.getContext());
  if (failed(SubViewOp::inferReturnTypes(
          parser.getContext(), std::nullopt, result.operands, attrs, nullptr,
          RegionRange(), inferredReturnTypes))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.subview result type");
  }
  result.addTypes(inferredReturnTypes);
  return success();
}

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (failed(parseSubViewSourceOffsetsAndSizes(parser, result, source, offsets)))
    return failure();
  if (failed(parseSubViewValids(parser, valids)))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy))
    return failure();
  if (failed(resolveSubViewSourceAndIndices(parser, result, source, sourceTy,
                                            resultTy, hasExplicitResultTy,
                                            offsets, valids)))
    return failure();

  int32_t hasValid = valids.empty() ? 0 : 1;
  addOperandSegmentSizesAttr(parser, result,
                             {1, static_cast<int32_t>(offsets.size()), hasValid,
                              hasValid});
  return finalizeSubViewResultTypes(parser, result, resultTy,
                                    hasExplicitResultTy);
}
