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

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar load only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects result type to match ptr element type");

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar store only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects value type to match ptr element type");

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr, IntegerAttr modeAttr) {
  if (!opTypeAttr)
    return op->emitOpError("expects 'op_type' attribute");

  auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
  if (failed(opTypeOr)) {
    auto diag =
        op->emitOpError("expects 'op_type' to be pipe_event_type/sync_op_type, got ");
    diag << opTypeAttr;
    return failure();
  }
  pto::PIPE pipe = mapSyncOpTypeToPipe(*opTypeOr);
  if (!isConcreteSyncPipe(pipe))
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");

  if (!bufIdAttr)
    return op->emitOpError("expects 'buf_id' attribute");
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31)
    return op->emitOpError("expects 'buf_id' in range [0, 31]");

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0)
      return op->emitOpError("expects 'mode' to be non-negative");
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}
// ---- TOp ----

static LogicalResult verifyMatmulBiasLikeOp(Operation *op, Type aTy, Type bTy,
                                            Type biasTy, Type dstTy,
                                            bool useGemvOperands) {
  if (useGemvOperands) {
    if (failed(verifyGemvTileOperands(op, aTy, bTy, dstTy)))
      return failure();
  } else {
    if (failed(verifyMatTileOperands(op, aTy, bTy, dstTy)))
      return failure();
  }
  if (failed(verifyMatBiasTile(op, biasTy, dstTy)))
    return failure();
  if (failed(verifyMatmulTypeTriple(op, getElemTy(aTy), getElemTy(bTy),
                                    getElemTy(dstTy))))
    return failure();
  return verifyMatmulLike(op, aTy, bTy, dstTy);
}

template <typename ExtraVerifyFn>
static LogicalResult verifyMatmulMxA2A3LikeOp(Operation *op, Type aScaleTy,
                                              Type bScaleTy, Type aTy, Type bTy,
                                              Type dstTy,
                                              ExtraVerifyFn extraVerify) {
  if (failed(verifyTileBufCommon(op, aScaleTy, "a_scale")) ||
      failed(verifyTileBufCommon(op, bScaleTy, "b_scale")))
    return failure();
  if (failed(extraVerify()))
    return failure();
  return verifyMatmulLike(op, aTy, bTy, dstTy);
}

template <typename VerifyBaseFn>
static LogicalResult verifyMatmulMxA5LikeOp(Operation *op, Type aTy, Type bTy,
                                            Type dstTy,
                                            VerifyBaseFn verifyBase) {
  if (failed(verifyBase()))
    return failure();
  return verifyA5MxTypeTriple(op, aTy, bTy, dstTy, "a", "b", "dst");
}

LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulBiasLikeOp(*this, getA().getType(), getB().getType(),
                                  getBias().getType(), getDst().getType(),
                                  /*useGemvOperands=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2)
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1])
      return emitOpError("expects bias and dst to have the same column shape");
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulBiasLikeOp(*this, getA().getType(), getB().getType(),
                                  getBias().getType(), getDst().getType(),
                                  /*useGemvOperands=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulMxA2A3LikeOp(
        *this, getAScale().getType(), getBScale().getType(), getA().getType(),
        getB().getType(), getDst().getType(),
        []() -> LogicalResult { return success(); });
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyMatmulMxA5LikeOp(*this, getA().getType(), getB().getType(),
                                  getDst().getType(), verifyA2A3);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")))
      return failure();
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA2A3()))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "a", "b", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyMatmulMxA2A3LikeOp(
        *this, getAScale().getType(), getBScale().getType(), getA().getType(),
        getB().getType(), getDst().getType(), [&]() -> LogicalResult {
          if (failed(verifyMatTileOperands(*this, getA().getType(),
                                           getB().getType(),
                                           getDst().getType())) ||
              failed(verifyMatBiasTile(*this, getBias().getType(),
                                       getDst().getType(),
                                       /*requireFloatBias=*/true)))
            return failure();
          return success();
        });
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyMatmulMxA5LikeOp(*this, getA().getType(), getB().getType(),
                                  getDst().getType(), verifyA2A3);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType())
      return emitOpError("expects val type to match dst element type");
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType, MemRefType>(srcTy))
    return emitOpError("expects src to be tile_buf or memref type");

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace =
      isa<pto::TileBufType>(srcTy)
          ? cast<pto::TileBufType>(srcTy).getMemorySpace()
          : cast<MemRefType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT)
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType())
    return emitOpError("expects dst type to match src element type");
  return success();
}

static bool isIntegerTypeWidth(Type ty, unsigned width) {
  auto it = dyn_cast<IntegerType>(ty);
  return it && it.getWidth() == width;
}

static LogicalResult verifyTHistogramShapes(THistogramOp op, Type srcTy,
                                            Type idxTy, Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto idxShape = getShapeVec(idxTy);
  auto dstShape = getShapeVec(dstTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto idxValid = getValidShapeVec(idxTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
      srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError(
        "expects src, idx, and dst to have rank-2 shape and valid_shape");
  }
  if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], idxValid[0])) {
    return op.emitOpError("expects idx rows and valid rows to match src");
  }
  if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
      !hasCompatibleKnownExtent(srcValid[0], dstValid[0])) {
    return op.emitOpError("expects dst rows and valid rows to match src");
  }
  if (!isKnownUnitExtent(idxShape[1]) || !isKnownUnitExtent(idxValid[1]))
    return op.emitOpError("expects idx to have exactly one column");
  if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256)
    return op.emitOpError("expects dst shape[1] to be at least 256");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] < 256)
    return op.emitOpError("expects dst valid_shape[1] to be at least 256");
  return success();
}

static LogicalResult verifyTHistogramA5(THistogramOp op) {
  Type srcTy = op.getSrc().getType();
  Type idxTy = op.getIdx().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, idxTy, "idx")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto idxSpace = getPTOMemorySpaceEnum(idxTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects src to be in the vec address space");
  if (!idxSpace || *idxSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects idx to be in the vec address space");
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects dst to be in the vec address space");

  auto srcTB = dyn_cast<pto::TileBufType>(srcTy);
  auto idxTB = dyn_cast<pto::TileBufType>(idxTy);
  auto dstTB = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTB || !idxTB || !dstTB)
    return op.emitOpError("expects src, idx, and dst to be tile_buf types");
  if (!hasTileBufLayout(srcTB, pto::BLayout::RowMajor, pto::SLayout::NoneBox))
    return op.emitOpError("expects src to use row_major + none_box layout");
  if (!hasTileBufLayout(dstTB, pto::BLayout::RowMajor, pto::SLayout::NoneBox))
    return op.emitOpError("expects dst to use row_major + none_box layout");
  if (!hasTileBufLayout(idxTB, pto::BLayout::ColMajor, pto::SLayout::NoneBox)) {
    return op.emitOpError(
        "expects idx to use DN layout (col_major + none_box)");
  }

  if (!isIntegerTypeWidth(getElemTy(srcTy), 16))
    return op.emitOpError("expects src element type to be ui16");
  if (!isIntegerTypeWidth(getElemTy(idxTy), 8))
    return op.emitOpError("expects idx element type to be ui8");
  if (!isIntegerTypeWidth(getElemTy(dstTy), 32))
    return op.emitOpError("expects dst element type to be ui32");
  return verifyTHistogramShapes(op, srcTy, idxTy, dstTy);
}

LogicalResult THistogramOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTHistogramA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")))
      return failure();
    if (failed(verifyScaleTileMatchesOperand(*this, dstTy, srcTy, "dst", "src")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
LogicalResult MScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  if (!isTargetArchA5(getOperation()))
    return emitOpError("pto.mscatter is only supported on A5 targets");

  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1)
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  if (!srcElem || !idxElem)
    return emitOpError("failed to resolve element types for src or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem))
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src")))
    return failure();

  if (getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
      getScatterOob() != pto::ScatterOOB::Undefined) {
    if (!isTargetArchA5(getOperation()))
      return emitOpError(
          "expects non-default scatterAtomicOp/scatterOob only on A5 targets");
  }

  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, getScatterAtomicOp()))
    return emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src")))
    return failure();

  return success();
}

// ---- MGatherOp ----
LogicalResult MGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  if (!isTargetArchA5(getOperation()))
    return emitOpError("pto.mgather is only supported on A5 targets");

  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();

  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1)
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(*this, dstTy, "dst")) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  if (!dstElem || !idxElem)
    return emitOpError("failed to resolve element types for dst or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), dstElem))
    return emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), dstElem,
                                             "dst")))
    return failure();

  if (getGatherOob() != pto::GatherOOB::Undefined &&
      !isTargetArchA5(getOperation()))
    return emitOpError(
        "expects non-default gatherOob only on A5 targets");

  if (failed(verifyMGatherMScatterTileShape(getOperation(), dstTy, idxTy, "dst")))
    return failure();

  return success();
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs());
  p << " : " << getSrc().getType();
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(srcTy))
    return failure();
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode"))
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  return success();
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    p << ", " << getTmp();
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    p << ", " << getTmp().getType();
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

static ParseResult parseTMrgSortFormat1(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  Type srcTy, blockLenTy, dstTy;
  if (parser.parseType(srcTy) || parser.parseComma() ||
      parser.parseType(blockLenTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand dstOp;
  if (parser.parseOperand(dstOp) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, 1, 0, 0}));
  if (parser.resolveOperand(first, srcTy, result.operands) ||
      parser.resolveOperand(second, blockLenTy, result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
  return success();
}

static ParseResult parseTMrgSortFormat2Sources(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &srcs,
    OpAsmParser::UnresolvedOperand &tmpOp) {
  while (parser.parseOptionalComma().succeeded()) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next))
      return failure();
    srcs.push_back(next);
  }
  if (srcs.size() < 3 || srcs.size() > 5) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "tmrgsort format2 expects 2 to 4 src operands plus one tmp operand");
  }
  tmpOp = srcs.pop_back_val();
  return success();
}

static ParseResult parseTMrgSortFormat2Exhausted(OpAsmParser &parser,
                                                 bool &exhaustedVal) {
  if (failed(parser.parseOptionalLBrace()))
    return success();
  if (parser.parseKeyword("exhausted") || parser.parseEqual())
    return failure();
  StringRef kw;
  if (parser.parseKeyword(&kw) || parser.parseRBrace())
    return failure();
  exhaustedVal = kw == "true";
  return success();
}

static ParseResult parseTMrgSortFormat2Types(
    OpAsmParser &parser, MutableArrayRef<OpAsmParser::UnresolvedOperand> srcs,
    SmallVectorImpl<Type> &srcTypes, Type &tmpTy) {
  if (parser.parseColon())
    return failure();
  Type firstSrcTy;
  if (parser.parseType(firstSrcTy))
    return failure();
  srcTypes.push_back(firstSrcTy);
  while (parser.parseOptionalComma().succeeded()) {
    Type nextTy;
    if (parser.parseType(nextTy))
      return failure();
    srcTypes.push_back(nextTy);
  }
  if (srcTypes.size() != srcs.size() + 1 || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  tmpTy = srcTypes.pop_back_val();
  return success();
}

static ParseResult parseTMrgSortFormat2Outputs(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &dstOp,
    OpAsmParser::UnresolvedOperand &excutedOp, Type &dstTy, Type &excutedTy) {
  if (parser.parseOperand(dstOp) || parser.parseComma() ||
      parser.parseOperand(excutedOp) || parser.parseColon() ||
      parser.parseType(dstTy) || parser.parseComma() ||
      parser.parseType(excutedTy) || parser.parseRParen())
    return failure();
  return success();
}

static ParseResult resolveTMrgSortFormat2Operands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> srcs, ArrayRef<Type> srcTypes,
    OpAsmParser::UnresolvedOperand tmpOp, Type tmpTy,
    OpAsmParser::UnresolvedOperand dstOp, Type dstTy,
    OpAsmParser::UnresolvedOperand excutedOp, Type excutedTy,
    bool exhaustedVal) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(srcs.size()), 0, 1, 1, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      parser.resolveOperand(tmpOp, tmpTy, result.operands) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted")) {
    result.addAttribute("exhausted",
                        parser.getBuilder().getBoolAttr(exhaustedVal));
  }
  return success();
}

static ParseResult parseTMrgSortFormat2(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand first,
    OpAsmParser::UnresolvedOperand second) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs = {first, second};
  OpAsmParser::UnresolvedOperand tmpOp;
  if (failed(parseTMrgSortFormat2Sources(parser, srcs, tmpOp)))
    return failure();
  bool exhaustedVal = false;
  if (failed(parseTMrgSortFormat2Exhausted(parser, exhaustedVal)))
    return failure();
  SmallVector<Type, 4> srcTypes;
  srcTypes.reserve(srcs.size());
  Type tmpTy;
  if (failed(parseTMrgSortFormat2Types(parser, srcs, srcTypes, tmpTy)))
    return failure();
  OpAsmParser::UnresolvedOperand dstOp, excutedOp;
  Type dstTy, excutedTy;
  if (failed(
          parseTMrgSortFormat2Outputs(parser, dstOp, excutedOp, dstTy, excutedTy)))
    return failure();
  return resolveTMrgSortFormat2Operands(parser, result, srcs, srcTypes, tmpOp,
                                        tmpTy, dstOp, dstTy, excutedOp,
                                        excutedTy, exhaustedVal);
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second))
    return failure();

  if (parser.parseOptionalColon().succeeded())
    return parseTMrgSortFormat1(parser, result, first, second);
  return parseTMrgSortFormat2(parser, result, first, second);
}

static LogicalResult verifyTMrgSortFormat1(TMrgSortOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op.emitOpError()
           << "format1 expects PTO shaped-like types for src/dst";
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError() << "expects src/dst to have the same element type";
  if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32())
    return op.emitOpError() << "expects element type to be f16 or f32";
  auto ss = getShapeVec(srcTy);
  auto ds = getShapeVec(dstTy);
  if (ss.size() != 2 || ds.size() != 2)
    return op.emitOpError() << "expects src/dst to be rank-2 tile-shaped";
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
    return op.emitOpError() << "expects src rows == 1";
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
    return op.emitOpError() << "expects dst rows == 1";
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic &&
      ss[1] != ds[1])
    return op.emitOpError() << "expects src/dst cols to match";
  if (auto cstOp = op.getBlockLen().getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
      int64_t v = intAttr.getValue().getSExtValue();
      if (v <= 0 || (v % 64) != 0)
        return op.emitOpError()
               << "expects blockLen > 0 and multiple of 64";
    }
  }
  return mlir::success();
}

static LogicalResult verifyTMrgSortSingleRowTile(Operation *op, Type ty,
                                                 StringRef name) {
  auto shape = getShapeVec(ty);
  if (shape.size() != 2)
    return op->emitOpError() << "format2 expects " << name
                             << " to be rank-2 tile-shaped";
  if (shape[0] != mlir::ShapedType::kDynamic && shape[0] != 1)
    return op->emitOpError() << "format2 expects " << name << " rows == 1";
  return success();
}

static LogicalResult verifyTMrgSortFormat2Executed(Operation *op,
                                                   Value excuted) {
  auto excutedTy = mlir::dyn_cast<mlir::VectorType>(excuted.getType());
  if (!excutedTy || excutedTy.getRank() != 1 ||
      excutedTy.getNumElements() != 4 ||
      !excutedTy.getElementType().isInteger(16))
    return op->emitOpError() << "format2 excuted must be vector<4xi16>";
  return success();
}

static LogicalResult verifyTMrgSortFormat2Src(Operation *op, Value src,
                                              Type elemTy) {
  Type srcTy = src.getType();
  if (failed(verifyTMrgSortSingleRowTile(op, srcTy, "src")))
    return failure();
  if (getElemTy(srcTy) != elemTy)
    return op->emitOpError()
           << "format2 expects src/dst/tmp element types to match";
  return success();
}

static LogicalResult verifyTMrgSortFormat2(TMrgSortOp op) {
  for (Value v : op.getSrcs()) {
    if (!isPTOShapedLike(v.getType()))
      return op.emitOpError()
             << "format2 expects PTO shaped-like type for each src";
  }
  if (op.getSrcs().size() < 2u || op.getSrcs().size() > 4u)
    return op.emitOpError() << "format2 expects 2 to 4 srcs";
  if (op.getDsts().size() != 1u || !op.getTmp() || !op.getExcuted())
    return op.emitOpError()
           << "format2 expects ins(srcs..., tmp), outs(dst), and excuted=vector";
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp().getType();
  if (!isPTOShapedLike(dstTy) || !isPTOShapedLike(tmpTy))
    return op.emitOpError() << "format2 dst/tmp must be PTO shaped-like";
  if (failed(verifyTMrgSortFormat2Executed(op, op.getExcuted())))
    return failure();
  Type elemTy = getElemTy(dstTy);
  if (elemTy != getElemTy(tmpTy))
    return op.emitOpError() << "format2 expects dst/tmp element types to match";
  auto dstShape = getShapeVec(dstTy);
  auto tmpShape = getShapeVec(tmpTy);
  if (failed(verifyTMrgSortSingleRowTile(op, dstTy, "dst")) ||
      failed(verifyTMrgSortSingleRowTile(op, tmpTy, "tmp")))
    return failure();
  if (dstShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < dstShape[1])
    return op.emitOpError() << "format2 expects tmp.cols >= dst.cols";
  for (Value src : op.getSrcs()) {
    if (failed(verifyTMrgSortFormat2Src(op, src, elemTy)))
      return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (isFormat1())
    return verifyTMrgSortFormat1(*this);
  if (isFormat2())
    return verifyTMrgSortFormat2(*this);
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/false);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return emitOpError() << "failed to get element type for src/dst";
  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!mlir::isa<IntegerType>(srcElem))
    return emitOpError() << "expects integral element types";
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0)
    return emitOpError("expects tshls scalar to be non-negative");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32))
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTNegCommon(Operation *op, Type srcTy, Type dstTy) {
  if (failed(verifyVecTileStorage(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")))
    return failure();
  return getElemTy(srcTy);
}

static LogicalResult verifyTNegA2A3(TNegOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTNegCommon(op, srcTy, dstTy);
  if (failed(elemOr) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  Type elemTy = *elemOr;
  if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
        elemTy.isF32())) {
    return op.emitOpError()
           << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
  }
  return success();
}

static LogicalResult verifyTNegA5(TNegOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTNegCommon(op, srcTy, dstTy);
  if (failed(elemOr))
    return failure();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op.emitOpError() << "expects src and dst to have rank-2 valid_shape";
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op.emitOpError()
           << "expects src and dst to have the same valid_shape[1]";
  }
  Type elemTy = *elemOr;
  if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
        elemTy.isF16() || elemTy.isF32() || elemTy.isBF16())) {
    return op.emitOpError()
           << "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTNegA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTNegA5(*this); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy)) {
      emitOpError() << "expects src and dst to have the same element type";
      return failure();
    }
    return elemTy;
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemTy = verifyCommon();
    if (failed(elemTy))
      return failure();
    if (!(*elemTy).isInteger(16))
      return emitOpError() << "expects A2/A3 tnot element type to be i16";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemTy = verifyCommon();
    if (failed(elemTy))
      return failure();
    if (!((*elemTy).isInteger(8) || (*elemTy).isInteger(16) ||
          (*elemTy).isInteger(32)))
      return emitOpError() << "expects A5 tnot element type to be i8/i16/i32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  return verifyRowMajorBinaryIntWidthOp(
      getOperation(), getSrc0().getType(), getSrc1().getType(),
      getDst().getType(),
      "expects A2/A3 tor src0, src1, and dst element type to be i8/i16",
      "expects A5 tor src0, src1, and dst element type to be i8/i16/i32");
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  return verifyDistinctRowMajorUnaryIntWidthOp(
      getOperation(), getSrc(), getDst(), "src", "dst",
      "expects A2/A3 tors src and dst element type to be i8/i16",
      "expects A5 tors src and dst element type to be i8/i16/i32");
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy))
    return op->emitOpError(
               "expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types"),
           failure();
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for operands"), failure();
  if (e0 != e1 || e0 != ed)
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd)
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  return e0;
}

static LogicalResult verifyTPartBinaryLikeOp(Operation *op, Type src0Ty,
                                             Type src1Ty, Type dstTy,
                                             StringRef opName);

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartadd");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

static LogicalResult verifyTPartArgIndexOperands(Operation *op, Type src0Ty,
                                                 Type src1Ty, Type src0IdxTy,
                                                 Type src1IdxTy, Type dstTy,
                                                 Type dstIdxTy) {
  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy))
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy)) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");
  }
  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) || dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  }
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");
  }
  return success();
}

static LogicalResult verifyTPartArgElementType(Operation *op, Type elem,
                                               StringRef opName) {
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32())) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
    }
    return success();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
        elem.isF32())) {
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i32/i16/f16/f32";
  }
  return success();
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElemOr))
    return failure();
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)))
    return failure();

  if (failed(verifyTPartArgIndexOperands(op, src0Ty, src1Ty, src0IdxTy,
                                         src1IdxTy, dstTy, dstIdxTy)))
    return failure();
  return verifyTPartArgElementType(op, *dataElemOr, opName);
}

static LogicalResult verifyTPartBinaryLikeOp(Operation *op, Type src0Ty,
                                             Type src1Ty, Type dstTy,
                                             StringRef opName) {
  FailureOr<Type> elemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(elemOr))
    return failure();
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
    return op->emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  PTOArch arch = getTargetArch(op);
  if (arch != PTOArch::A5 &&
      failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)))
    return failure();
  Type elem = *elemOr;
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return op->emitOpError()
             << "expects A5 " << opName
             << " element type to be i32/i16/i8/f16/bf16/f32";
    return success();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
        elem.isF32()))
    return op->emitOpError()
           << "expects A2/A3 " << opName
           << " element type to be i32/i16/f16/f32";
  return success();
}

mlir::LogicalResult mlir::pto::TPartArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartBinaryLikeOp(getOperation(), getSrc0().getType(),
                                   getSrc1().getType(), getDst().getType(),
                                   "tpartmul");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

struct TPReluCommonInfo {
  Type src0Ty;
  Type src1Ty;
  Type tmpTy;
  Type dstTy;
};

static LogicalResult verifyTPReluElemAndLayout(TPReluOp op, Type src0Ty,
                                               Type src1Ty, Type tmpTy,
                                               Type dstTy) {
  Type src0Elem = getElemTy(src0Ty);
  Type src1Elem = getElemTy(src1Ty);
  Type tmpElem = getElemTy(tmpTy);
  Type dstElem = getElemTy(dstTy);
  if (!src0Elem || !src1Elem || !tmpElem || !dstElem)
    return op.emitOpError("failed to get element type for operands");
  if (src0Elem != src1Elem || src0Elem != dstElem)
    return op.emitOpError("expects dst/src0/src1 to have the same element type");
  if (!(src0Elem.isF16() || src0Elem.isF32())) {
    return op.emitOpError("expects dst/src0/src1 element type to be f16 or f32");
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError("expects src0, src1, and dst to use row-major layout");
  }
  return success();
}

static LogicalResult verifyTPReluMatchingShapes(TPReluOp op, Type src0Ty,
                                                Type src1Ty, Type tmpTy,
                                                Type dstTy) {
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst")))
    return failure();
  if (getShapeVec(src0Ty) != getShapeVec(src1Ty) ||
      getShapeVec(src0Ty) != getShapeVec(tmpTy) ||
      getShapeVec(src0Ty) != getShapeVec(dstTy)) {
    return op.emitOpError("expects src0/src1/tmp/dst to have the same shape");
  }
  return success();
}

static FailureOr<TPReluCommonInfo> verifyTPReluCommon(TPReluOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type tmpTy = op.getTmp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }

  if (failed(verifyTPReluElemAndLayout(op, src0Ty, src1Ty, tmpTy, dstTy)))
    return failure();
  if (failed(verifyTPReluMatchingShapes(op, src0Ty, src1Ty, tmpTy, dstTy)))
    return failure();
  return TPReluCommonInfo{src0Ty, src1Ty, tmpTy, dstTy};
}

static LogicalResult verifyTPReluA2A3(TPReluOp op,
                                      const TPReluCommonInfo &common) {
  Type tmpElem = getElemTy(common.tmpTy);
  auto tmpIntTy = dyn_cast<IntegerType>(tmpElem);
  if (!tmpIntTy || tmpIntTy.getWidth() != 8)
    return op.emitOpError("expects A2/A3 tmp element type to be u8");
  if (!isRowMajorTileBuf(common.tmpTy))
    return op.emitOpError("expects tmp to use row-major layout");
  if (auto arch = getVerifierArchName(op.getOperation());
      arch && arch->equals_insensitive("a3")) {
    if (op.getSrc0() == op.getSrc1() || op.getSrc0() == op.getTmp() ||
        op.getSrc0() == op.getDst() || op.getSrc1() == op.getTmp() ||
        op.getSrc1() == op.getDst() || op.getTmp() == op.getDst()) {
      return op.emitOpError(
          "expects A3 src0, src1, tmp, and dst to use different storage");
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto common = verifyTPReluCommon(*this);
  if (failed(common))
    return failure();
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPReluA2A3(*this, *common); };
  auto verifyA5 = [&]() -> LogicalResult { return success(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTQuantStructural(TQuantOp op) {
  Type dstElemTy = getElemTy(op.getDst().getType());
  auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
  if (op.getQuantType() == mlir::pto::QuantType::INT8_SYM) {
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError(
          "expects dst element type i8/ui8 for INT8_SYM quantization");
    }
    if (op.getOffset()) {
      return op.emitOpError(
          "INT8_SYM quantization must not have an offset operand");
    }
    return success();
  }

  if (!dstIntTy || dstIntTy.getWidth() != 8) {
    return op.emitOpError(
        "expects dst element type i8/ui8 for INT8_ASYM quantization");
  }
  if (!op.getOffset())
    return op.emitOpError("INT8_ASYM quantization requires an offset operand");
  return success();
}

static LogicalResult verifyTQuantCommon(TQuantOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (!getElemTy(srcTy).isF32())
    return op.emitOpError("expects src to have element type f32");
  if (!op.getOffset())
    return success();

  Type offsetTy = op.getOffset().getType();
  if (failed(verifyTileBufCommon(op, offsetTy, "offset")))
    return failure();
  if (!getElemTy(offsetTy).isF32())
    return op.emitOpError("expects offset to have element type f32");
  return success();
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  if (failed(verifyTQuantStructural(*this)))
    return failure();
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTQuantCommon(*this)))
      return failure();
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError("expects A2/A3 src and dst to use row-major layout");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantCommon(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16))
      return emitOpError()
             << "expects src element type i8 or i16";
    if (!getElemTy(getDst().getType()).isF32())
      return emitOpError() << "expects dst element type f32";
    if (!getElemTy(getScale().getType()).isF32())
      return emitOpError() << "expects scale element type f32";
    if (!getElemTy(getOffset().getType()).isF32())
      return emitOpError() << "expects offset element type f32";
    return success();
  };

  if (failed(verifyStructural()))
    return failure();

  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst")))
      return failure();
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType()))
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst())
    return emitOpError("expects A3 trecip src and dst to use different storage");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << errorMessage;
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRemRowMajorTiles(Operation *op, Type src0Ty,
                                             Type src1Ty, Type tmpTy,
                                             Type dstTy) {
  if (isRowMajorTileBuf(src0Ty) && isRowMajorTileBuf(src1Ty) &&
      isRowMajorTileBuf(tmpTy) && isRowMajorTileBuf(dstTy)) {
    return success();
  }
  return op->emitOpError(
      "expects src0, src1, tmp, and dst to use row-major layout");
}

static LogicalResult verifyTRemTmpCoverage(Operation *op, Type tmpTy,
                                           Type dstTy) {
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2)
    return op->emitOpError("expects tmp and dst to be rank-2 tiles");
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1)
    return op->emitOpError("expects tmp to have at least 1 valid row");
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op->emitOpError(
        "expects tmp valid columns to cover dst valid columns");
  }
  return success();
}

static FailureOr<Type> verifyTRemCommon(Operation *op, Type src0Ty, Type src1Ty,
                                        Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return op->emitOpError("expects tmp and dst to have the same element type"),
           failure();
  if (failed(verifyTRemRowMajorTiles(op, src0Ty, src1Ty, tmpTy, dstTy)) ||
      failed(verifyTRemTmpCoverage(op, tmpTy, dstTy)))
    return failure();
  return getElemTy(src0Ty);
}

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  FailureOr<Type> elemOr = verifyTRemCommon(*this, getSrc0().getType(),
                                            getSrc1().getType(),
                                            getTmp().getType(),
                                            getDst().getType());
  if (failed(elemOr))
    return failure();
  Type elem = *elemOr;
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32()))
      return emitOpError("expects A2/A3 trem element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

static FailureOr<Type> verifyTRemScalarCommon(Operation *op, Type srcTy,
                                              Type tmpTy, Type dstTy,
                                              Type scalarTy) {
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return op->emitOpError("expects tmp and dst to have the same element type"),
           failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(tmpTy) ||
      !isRowMajorTileBuf(dstTy)) {
    return op->emitOpError("expects src, tmp, and dst to use row-major layout"),
           failure();
  }
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return op->emitOpError("expects scalar type to match the tile element type"),
           failure();
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2)
    return op->emitOpError("expects tmp and dst to be rank-2 tiles"), failure();
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1)
    return op->emitOpError("expects tmp to have at least 1 valid row"), failure();
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op->emitOpError("expects tmp valid columns to cover dst valid columns"),
           failure();
  }
  return elem;
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  FailureOr<Type> elemOr =
      verifyTRemScalarCommon(*this, getSrc().getType(), getTmp().getType(),
                             getDst().getType(), getScalar().getType());
  if (failed(elemOr))
    return failure();
  Type elem = *elemOr;
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32()))
      return emitOpError("expects A2/A3 trems element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src and dst to use row-major layout");

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return std::nullopt;
    if (d < 0)
      return std::nullopt;
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy)
    return std::nullopt;
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return 2;
    if (ft.isF32())
      return 4;
    if (ft.isF64())
      return 8;
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(1, bits / 8);
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isTileBufOrMemref(Type ty) {
  return mlir::isa<MemRefType, pto::TileBufType>(ty);
}

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value))
    return false;

  if (isa<AllocTileOp, DeclareTileOp, BindTileOp, PointerCastOp,
          MaterializeTileOp>(
          value.getDefiningOp()))
    return true;

  if (auto bitcast = value.getDefiningOp<BitcastOp>())
    return isLocallyBoundTileSource(bitcast.getSrc());
  if (auto reshape = value.getDefiningOp<TReshapeOp>())
    return isLocallyBoundTileSource(reshape.getSrc());

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return cOp.value();
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
    return cInt.value();
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue()))
      return ia.getInt();
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexLike(castOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexLike(truncOp.getIn());
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 tile_buf source");

    ArrayRef<int64_t> validShape = srcTy.getValidShape();
    if (validShape.size() != 2)
      return emitOpError("expects source validShape to be rank-2");
    if (!srcTy.hasDynamicValid())
      return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");

    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

    if (!isLocallyBoundTileSource(getSource()))
      return emitOpError(
          "requires a locally bound tile source; function arguments/results "
          "are unsupported");
  } else if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (!(*this)->hasAttr(kLoweredSetValidShapeAttrName))
      return emitOpError(
          "expects tile_buf source; memref source is only valid for the internal lowered form");
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 memref source after tile lowering");
    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());
  } else {
    return emitOpError("expects tile_buf source (or lowered memref source)");
  }

  auto checkDim = [&](Value operand, unsigned dimIdx,
                      StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal)
      return success();

    if (*constVal < 0)
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic)
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    return success();
  };

  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row")))
    return failure();
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col")))
    return failure();

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 tile_buf source");
    if (srcTy.getValidShape().size() != 2)
      return emitOpError("expects source validShape to be rank-2");
    return success();
  }
  if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 memref source after tile lowering");
    return success();
  }
  return emitOpError("expects tile_buf source (or lowered memref source)");
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb)
    return emitOpError("expects src/result to be !pto.tile_buf types");

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst")))
    return failure();

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace())
    return emitOpError("expects src and dst to use the same loc");

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value())
    return emitOpError("failed to get element byte width for src/dst");

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value())
    return emitOpError("expects static shapes for treshape");

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value())
    return emitOpError("expects src and dst to have the same total byte size");

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed)
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return emitOpError("expects src/result to have the same memorySpace");

  if (srcTy.getElementType() == dstTy.getElementType())
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");

  if (srcTy.getValidShape() != dstTy.getValidShape())
    return emitOpError("expects src/result to have the same validShape");

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg)
    return emitOpError("expects src/result to have the same tile config");

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value())
    return emitOpError("expects static shapes for bitcast");

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value())
    return emitOpError("unsupported element type for bitcast");

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes)
    return emitOpError("bitcast result requires more bytes than source storage");

  return success();
}


static LogicalResult verifyTRowExpandCommon(TRowExpandOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst")))
    return failure();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
    return op.emitOpError("expects src to be in the vec address space");
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getSLayoutValueI32() !=
        static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op.emitOpError("expects src to use the none_box slayout");
    }
  }
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op.emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return op.emitOpError("expects trowexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(op.getSrc());
  auto dstValid = getValidShapeVec(op.getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  auto checkNonZero = [&](ArrayRef<int64_t> valid, StringRef name)
      -> LogicalResult {
    if (valid[0] != ShapedType::kDynamic && valid[0] == 0)
      return op.emitOpError() << "expects " << name
                              << " valid_shape[0] to be non-zero";
    if (valid[1] != ShapedType::kDynamic && valid[1] == 0)
      return op.emitOpError() << "expects " << name
                              << " valid_shape[1] to be non-zero";
    return success();
  };
  if (failed(checkNonZero(srcValid, "src")) ||
      failed(checkNonZero(dstValid, "dst")))
    return failure();
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTRowExpandCommon(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTRowExpandCommon(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
