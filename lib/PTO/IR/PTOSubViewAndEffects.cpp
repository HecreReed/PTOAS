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

void mlir::pto::SubViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << "[";
  printer.printOperands(getOffsets());
  printer << "] sizes " << getSizes();
  if (getValidRow()) {
    printer << " valid [" << getValidRow() << ", " << getValidCol() << "]";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "sizes"});
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

static std::optional<ArrayAttr> getSubViewSizeAttr(DictionaryAttr attributes,
                                                   OpaqueProperties properties) {
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop && prop->sizes)
      return prop->sizes;
  }
  if (attributes)
    return attributes.getAs<ArrayAttr>("sizes");
  return std::nullopt;
}

static SmallVector<int64_t> collectSubviewShape(ArrayAttr sizeAttr) {
  SmallVector<int64_t> subviewShape;
  for (auto attr : sizeAttr)
    subviewShape.push_back(llvm::cast<IntegerAttr>(attr).getInt());
  return subviewShape;
}

struct SubViewExplicitValidOperands {
  Value row;
  Value col;
};

static void decodeSubviewExplicitValidOperandsFromSegments(
    SubViewExplicitValidOperands &explicitValids, ValueRange operands,
    DictionaryAttr attributes) {
  if (!attributes)
    return;
  auto segAttr = attributes.getAs<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segAttr)
    return;
  ArrayRef<int32_t> segs = segAttr.asArrayRef();
  if (segs.size() != 4)
    return;
  int32_t srcSeg = segs[0];
  int32_t offSeg = segs[1];
  int32_t vRowSeg = segs[2];
  int32_t vColSeg = segs[3];
  if (srcSeg != 1 || offSeg < 0 || (vRowSeg != 0 && vRowSeg != 1) ||
      (vColSeg != 0 && vColSeg != 1))
    return;
  size_t idx = static_cast<size_t>(srcSeg + offSeg);
  if (vRowSeg == 1 && idx < operands.size())
    explicitValids.row = operands[idx++];
  if (vColSeg == 1 && idx < operands.size())
    explicitValids.col = operands[idx];
}

static SubViewExplicitValidOperands decodeSubviewExplicitValidOperands(
    ValueRange operands, DictionaryAttr attributes, int64_t rank) {
  SubViewExplicitValidOperands explicitValids;
  decodeSubviewExplicitValidOperandsFromSegments(explicitValids, operands,
                                                 attributes);
  if (!explicitValids.row && !explicitValids.col && rank == 2) {
    size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
    if (operands.size() >= expectedWithoutValid + 2) {
      explicitValids.row = operands[expectedWithoutValid];
      explicitValids.col = operands[expectedWithoutValid + 1];
    }
  }
  return explicitValids;
}

static SmallVector<int64_t> inferSubviewValidShape(ArrayRef<int64_t> subviewShape,
                                                   Value explicitVRow,
                                                   Value explicitVCol) {
  constexpr int64_t kDynamicValidDim = -1;
  SmallVector<int64_t> validShape;
  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = (i == 0) ? explicitVRow : (i == 1 ? explicitVCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : kDynamicValidDim;
    }
    validShape.push_back(vdim);
  }
  return validShape;
}

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return failure();
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType)
    return failure();

  auto sizeAttr = getSubViewSizeAttr(attributes, properties);
  if (!sizeAttr)
    return failure();
  auto subviewShape = collectSubviewShape(*sizeAttr);
  if (subviewShape.size() != sourceType.getShape().size())
    return failure();

  auto explicitValids = decodeSubviewExplicitValidOperands(
      operands, attributes, static_cast<int64_t>(subviewShape.size()));
  auto validShape = inferSubviewValidShape(subviewShape, explicitValids.row,
                                           explicitValids.col);

  auto cfg = sourceType.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(context);

  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  inferredReturnTypes.push_back(TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg));
  return success();
}

// =============================================================================
// SubViewOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndex(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndex(truncOp.getIn(), out);
  return false;
}

static bool readLayoutI32(Attribute attr, int32_t &out) {
  if (auto layoutAttr = dyn_cast<BLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layoutAttr.getValue());
    return true;
  }
  if (auto layoutAttr = dyn_cast<SLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layoutAttr.getValue());
    return true;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
  }
  return false;
}

static LogicalResult computeBoxedInnerShape(Type elemTy, int32_t fractalSize,
                                            int32_t slayout, int64_t &innerRows,
                                            int64_t &innerCols) {
  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0)
    return failure();
  if (fractalSize == 1024) {
    innerRows = 16;
    innerCols = 16;
    return success();
  }
  if (fractalSize == 32) {
    innerRows = 16;
    innerCols = 2;
    return success();
  }
  if (fractalSize == 512 && slayout == 1) {
    innerRows = 16;
    innerCols = 32 / elemBytes;
    return success();
  }
  if (fractalSize == 512 && slayout == 2) {
    innerRows = 32 / elemBytes;
    innerCols = 16;
    return success();
  }
  return failure();
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  bl = 0;
  sl = 0;
  int32_t fractalSize = 512;
  (void)readLayoutI32(cfg.getBLayout(), bl);
  (void)readLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    fractalSize = static_cast<int32_t>(attr.getInt());

  boxed = sl != 0;
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }
  return computeBoxedInnerShape(elemTy, fractalSize, sl, innerRows, innerCols);
}

struct SubViewVerifyInfo {
  TileBufType srcTy;
  TileBufType dstTy;
  int64_t sizeR;
  int64_t sizeC;
  int64_t offR = 0;
  int64_t offC = 0;
  bool offRConst = false;
  bool offCConst = false;
};

static FailureOr<SubViewVerifyInfo> verifySubviewOperandsAndSizes(SubViewOp op) {
  auto srcTy = llvm::dyn_cast<TileBufType>(op.getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(op.getResult().getType());
  if (!srcTy || !dstTy)
    return op.emitOpError("expects tile_buf src and tile_buf result"), failure();
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return op.emitOpError("expects rank-2 tilebuf for src/dst"), failure();

  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != 2)
    return op.emitOpError("subview expects 2D sizes"), failure();
  int64_t sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (sizeR <= 0 || sizeC <= 0)
    return op.emitOpError("subview sizes must be positive"), failure();
  if (op.getOffsets().size() != 2)
    return op.emitOpError("subview expects 2D offsets"), failure();

  SubViewVerifyInfo info{srcTy, dstTy, sizeR, sizeC};
  info.offRConst = getConstIndex(op.getOffsets()[0], info.offR);
  info.offCConst = getConstIndex(op.getOffsets()[1], info.offC);
  if ((info.offRConst && info.offR < 0) || (info.offCConst && info.offC < 0)) {
    return op.emitOpError("subview offsets must be non-negative"), failure();
  }
  return info;
}

static LogicalResult verifySubviewExplicitValids(SubViewOp op, int64_t sizeR,
                                                 int64_t sizeC) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");
  }
  if (!hasValidRow)
    return success();

  int64_t vRow = 0;
  int64_t vCol = 0;
  if (getConstIndex(op.getValidRow(), vRow)) {
    if (vRow <= 0)
      return op.emitOpError("valid_row must be positive when constant");
    if (vRow > sizeR)
      return op.emitOpError("valid_row must be <= subview row size");
  }
  if (getConstIndex(op.getValidCol(), vCol)) {
    if (vCol <= 0)
      return op.emitOpError("valid_col must be positive when constant");
    if (vCol > sizeC)
      return op.emitOpError("valid_col must be <= subview col size");
  }
  return success();
}

static LogicalResult verifySubviewResultType(SubViewOp op, const SubViewVerifyInfo &info) {
  auto dstShape = info.dstTy.getShape();
  auto srcShape = info.srcTy.getShape();
  if (dstShape.size() != 2)
    return op.emitOpError("expects result to be rank-2");
  if (srcShape.size() != 2)
    return op.emitOpError("expects source to be rank-2");
  if (dstShape[0] != info.sizeR || dstShape[1] != info.sizeC)
    return op.emitOpError("expects result shape to match subview sizes");
  if (info.dstTy.getElementType() != info.srcTy.getElementType())
    return op.emitOpError("expects result element type to match source");
  if (info.dstTy.getMemorySpace() != info.srcTy.getMemorySpace())
    return op.emitOpError("expects result address space to match source");

  auto srcCfg = info.srcTy.getConfigAttr();
  if (!srcCfg)
    srcCfg = TileBufConfigAttr::getDefault(op.getContext());
  auto dstCfg = info.dstTy.getConfigAttr();
  if (!dstCfg)
    dstCfg = TileBufConfigAttr::getDefault(op.getContext());
  if (dstCfg != srcCfg)
    return op.emitOpError("expects result tile config to match source");
  return success();
}

static int64_t getSubviewExpectedValidDim(Value explicitValid, int64_t defaultSize) {
  if (!explicitValid)
    return defaultSize;
  int64_t constantValue = 0;
  if (getConstIndex(explicitValid, constantValue))
    return std::min<int64_t>(constantValue, defaultSize);
  return ShapedType::kDynamic;
}

static LogicalResult verifySubviewResultValidShape(SubViewOp op,
                                                   const SubViewVerifyInfo &info) {
  auto dstValid = info.dstTy.getValidShape();
  if (dstValid.size() != 2)
    return op.emitOpError("expects result to have rank-2 valid_shape");
  int64_t expectedVRow = getSubviewExpectedValidDim(op.getValidRow(), info.sizeR);
  int64_t expectedVCol = getSubviewExpectedValidDim(op.getValidCol(), info.sizeC);
  if (dstValid[0] != expectedVRow)
    return op.emitOpError(
        "expects result valid_shape[0] to match inferred/explicit valid_row");
  if (dstValid[1] != expectedVCol)
    return op.emitOpError(
        "expects result valid_shape[1] to match inferred/explicit valid_col");
  return success();
}

static bool hasStaticRank2Shape(ArrayRef<int64_t> shape) {
  return shape.size() == 2 && shape[0] != ShapedType::kDynamic &&
         shape[1] != ShapedType::kDynamic;
}

static LogicalResult verifyBoxedSubviewMajorConstraint(
    SubViewOp op, const SubViewVerifyInfo &info, ArrayRef<int64_t> srcShape,
    int32_t bl) {
  if (bl == 0) {
    if (info.sizeC != srcShape[1])
      return op.emitOpError("boxed RowMajor subview must keep full cols");
    if (!info.offCConst || info.offC != 0)
      return op.emitOpError(
          "boxed RowMajor subview requires static col offset = 0");
    return success();
  }
  if (bl == 1) {
    if (info.sizeR != srcShape[0])
      return op.emitOpError("boxed ColMajor subview must keep full rows");
    if (!info.offRConst || info.offR != 0)
      return op.emitOpError(
          "boxed ColMajor subview requires static row offset = 0");
  }
  return success();
}

static LogicalResult verifyBoxedSubviewLayout(SubViewOp op,
                                              const SubViewVerifyInfo &info) {
  auto cfg = info.srcTy.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(op.getContext());

  int64_t innerRows = 1;
  int64_t innerCols = 1;
  bool boxed = false;
  int32_t bl = 0;
  int32_t sl = 0;
  if (failed(computeInnerShape(cfg, info.srcTy.getElementType(), innerRows,
                               innerCols, boxed, bl, sl))) {
    return op.emitOpError("unsupported tile layout for subview");
  }
  if (!boxed)
    return success();

  if (info.sizeR % innerRows != 0 || info.sizeC % innerCols != 0) {
    return op.emitOpError(
        "boxed layout subview sizes must be multiples of inner shape");
  }
  if (info.offRConst && info.offR % innerRows != 0)
    return op.emitOpError(
        "boxed layout subview offsets must be multiples of inner shape");
  if (info.offCConst && info.offC % innerCols != 0)
    return op.emitOpError(
        "boxed layout subview offsets must be multiples of inner shape");

  auto srcShape = info.srcTy.getShape();
  if (!hasStaticRank2Shape(srcShape)) {
    return op.emitOpError("boxed layout subview requires static source shape");
  }
  return verifyBoxedSubviewMajorConstraint(op, info, srcShape, bl);
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto infoOr = verifySubviewOperandsAndSizes(*this);
  if (failed(infoOr))
    return failure();
  if (failed(verifySubviewExplicitValids(*this, infoOr->sizeR, infoOr->sizeC)) ||
      failed(verifySubviewResultType(*this, *infoOr)) ||
      failed(verifySubviewResultValidShape(*this, *infoOr)) ||
      failed(verifyBoxedSubviewLayout(*this, *infoOr))) {
    return failure();
  }
  return success();
}

} // namespace pto
} // namespace mlir

// =============================================================================
// Helper Functions
// =============================================================================

[[maybe_unused]] static AddressSpace getAddressSpace(Value val) {
  auto type = llvm::dyn_cast<MemRefType>(val.getType());
  if (!type) return AddressSpace::Zero; // Default

  // 假设你的 AddressSpaceAttr 存储在 MemRef 的 memorySpace 中
  // 需要根据你的 getPTOAddressSpaceAttr 实现来调整
  auto attr = llvm::dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (attr) return attr.getAddressSpace();
  return AddressSpace::Zero;
}

// =============================================================================
// Side Effects Implementation
// =============================================================================

// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value

// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand)
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
}

// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result)
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

void TPrefetchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty())
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty())
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty())
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_QUATERNARY_EFFECTS(OpClass, op0, op1, op2, op3, dstOperand)      \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_READ(op3);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

void THistogramOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TGetScaleAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// SET_VALIDSHAPE: update runtime valid row/col metadata on source tile in-place.
void SetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getSourceMutable());
}

// GET_VALIDSHAPE: read runtime valid row/col metadata from source tile.
void GetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSourceMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TAxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TConcatOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_QUATERNARY_EFFECTS(TConcatidxOp, getSrc0Mutable(), getSrc1Mutable(), getSrc0IdxMutable(), getSrc1IdxMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TTRI: Write(dst) (generates triangular mask)
void TTriOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandExpdifOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColProdOp, getSrcMutable(), getDstMutable())

void TColArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TColArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TCvtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT_FP: Read(src), Read(fp) -> Write(dst)
void TExtractFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT_FP: Read(src), Read(fp) -> Write(dst)
void TInsertFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadInplaceOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty())
    PTO_ADD_WRITE(cdst[0]);
  if (auto indices = getIndicesMutable(); !indices.empty())
    PTO_ADD_READ(indices[0]);
  if (auto tmp = getTmpMutable(); !tmp.empty())
    PTO_ADD_READ(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMovFPOp, getSrcMutable(), getFpMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TPartArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
void TPartArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty())
    PTO_ADD_READ(offsetRange[0]);
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
void TRemOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRemSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

void TRowExpandDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TRowExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty())
      PTO_ADD_READ(idx[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(src0, src1) -> Write(tmp, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp?, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxOp ===
// Read: a, a_scale, b, b_scale, Write: dst
void TGemvMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxAccOp ===
// Read: c_in, a, a_scale, b, b_scale, Write: dst
void TGemvMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxBiasOp ===
// Read: a, a_scale, b, b_scale, bias, Write: dst
void TGemvMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
void TMatmulMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccMxOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
