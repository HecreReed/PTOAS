// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- LowerPTOToUBufOps.cpp - Lower pto.tadd/tsub/tmul/tdiv on a2a3 -----===//
//===----------------------------------------------------------------------===//
//
// Lowers pto.tadd/tsub/tmul/tdiv to pto.ub.vadd/vsub/vmul/vdiv on a3.
// Uses the full CCE dispatch tree from TBinOp.hpp with all modes.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_LOWERPTOTOUBUFOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {
static constexpr int64_t kRepeatMax = 255;
static constexpr int64_t kRepeatStrideMax = 255;
static constexpr int64_t kSmallRptBinOp = 4;
static constexpr int64_t kDefaultRepeatStride = 8;
static constexpr unsigned kMaskLen = 64;
static constexpr unsigned kHalfElementSizeBytes = 2;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static unsigned getElementSize(Type elemTy) {
  if (elemTy.isF16() || elemTy.isBF16()) {
    return kHalfElementSizeBytes;
  }
  if (elemTy.isF32()) {
    return mlir::pto::kValue4;
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    if (width == mlir::pto::kValue16 || width == 32) {
      return width / mlir::pto::kValue8;
    }
  }
  return 0;
}

static Type getStoredElemType(Type ty) {
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    return tbTy.getElementType();
  }
  if (auto mrTy = dyn_cast<MemRefType>(ty)) {
    return mrTy.getElementType();
  }
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    return ptrTy.getElementType();
  }
  return Type();
}

/// Returns true if the type lives in UB (VEC) address space.
static std::optional<bool> isUBMemorySpaceImpl(Type ty) {
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    auto msAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
    if (!msAttr) {
      return false;
    }
    return msAttr.getAddressSpace() == pto::AddressSpace::VEC;
  }
  if (auto mrTy = dyn_cast<MemRefType>(ty)) {
    auto msAttr =
        dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace());
    if (!msAttr) {
      return false;
    }
    return msAttr.getAddressSpace() == pto::AddressSpace::VEC;
  }
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    return ptrTy.getMemorySpace().getAddressSpace() == pto::AddressSpace::VEC;
  }
  return std::nullopt;
}

/// Returns true if the given type is confirmed UB memory space.
static bool isUBMemorySpace(Type ty) {
  auto result = isUBMemorySpaceImpl(ty);
  return result.has_value() && result.value();
}

static bool isRowMajor(pto::TileBufType tbTy) {
  auto config = tbTy.getConfigAttr();
  if (!config) {
    return true;
  }
  return config.getBLayout().getValue() != pto::BLayout::ColMajor;
}

static pto::PtrType getUBPtrType(MLIRContext *ctx, Type elemTy) {
  auto msAttr = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
  return pto::PtrType::get(ctx, elemTy, msAttr);
}

static std::pair<int64_t, int64_t>
computeContMaskValues(unsigned nElements) {
  int64_t mask0 = (nElements >= kMaskLen)
      ? static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)
      : static_cast<int64_t>((1ULL << nElements) - 1ULL);
  int64_t mask1 = (nElements > kMaskLen)
      ? static_cast<int64_t>((1ULL << (nElements - kMaskLen)) - 1ULL)
      : 0LL;
  return {mask0, mask1};
}

struct TileShapeInfo {
  int64_t vRows;
  int64_t vCols;
  int64_t cols;
  int64_t rows;
  unsigned elemSize;
  unsigned elementsPerRepeat;
  unsigned blockSizeElem;
};

struct TileShapeMetadata {
  SmallVector<int64_t, mlir::pto::kValue2> shape;
  SmallVector<int64_t, mlir::pto::kValue2> validShape;
};

using TileShapeMap = DenseMap<Value, TileShapeMetadata>;

static std::optional<TileShapeInfo> extractTileShapeInfoFromValue(
    Value opDst, const TileShapeMap &tileShapes) {
  Type dstTy = opDst.getType();
  if (!isUBMemorySpace(dstTy)) {
    return std::nullopt;
  }

  Type elemTy;
  ArrayRef<int64_t> shape;
  ArrayRef<int64_t> validShape;
  if (auto tbTy = dyn_cast<pto::TileBufType>(dstTy)) {
    elemTy = tbTy.getElementType();
    shape = tbTy.getShape();
    validShape = tbTy.getValidShape();
    if (!isRowMajor(tbTy)) {
      return std::nullopt;
    }
  } else if (auto mrTy = dyn_cast<MemRefType>(dstTy)) {
    elemTy = mrTy.getElementType();
    shape = mrTy.getShape();
  } else if (isa<pto::PtrType>(dstTy)) {
    auto it = tileShapes.find(opDst);
    if (it == tileShapes.end()) {
      return std::nullopt;
    }
    elemTy = cast<pto::PtrType>(dstTy).getElementType();
    shape = llvm::ArrayRef(it->second.shape);
    validShape = llvm::ArrayRef(it->second.validShape);
  } else {
    return std::nullopt;
  }
  unsigned elemSize = getElementSize(elemTy);
  if (elemSize == 0) {
    return std::nullopt;
  }

  if (shape.size() < mlir::pto::kValue2) {
    return std::nullopt;
  }

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  int64_t vRows = (!validShape.empty() &&
                   validShape[0] != ShapedType::kDynamic)
                      ? validShape[0] : rows;
  int64_t vCols = (validShape.size() >= 2 &&
                   validShape[1] != ShapedType::kDynamic)
                      ? validShape[1] : cols;
  if (vRows == ShapedType::kDynamic || vCols == ShapedType::kDynamic ||
      rows == ShapedType::kDynamic || cols == ShapedType::kDynamic) {
    return std::nullopt;
  }

  TileShapeInfo info;
  info.vRows = vRows;
  info.vCols = vCols;
  info.cols = cols;
  info.rows = rows;
  info.elemSize = elemSize;
  info.elementsPerRepeat = mlir::pto::kValue256 / elemSize;
  info.blockSizeElem = mlir::pto::kValue32 / elemSize;
  return info;
}

static std::optional<TileShapeInfo> extractTileShapeInfo(
    Operation *op, const TileShapeMap &tileShapes) {
  return extractTileShapeInfoFromValue(op->getOperand(mlir::pto::kValue2), tileShapes);
}

static bool canLower(Operation *op, const TileShapeMap &tileShapes) {
  return extractTileShapeInfo(op, tileShapes).has_value();
}

//===----------------------------------------------------------------------===//
// ND-to-2xNZ dual-output TEXTRACT expansion (A2/A3 VPTO)
//===----------------------------------------------------------------------===//
// Design doc section 9.3: A2/A3 VPTO cannot reuse the EmitC path. LowerPTOToUbOps
// materializes AllocTileOps to !pto.ptr<..., ub> before any op conversion, so a
// pointer-form pto.textract would otherwise survive into VPTO LLVM emission and
// fail. We snapshot every dual-output TExtractOp *before* allocation
// materialization (tile types are still available), then expand each into two
// scalar loop nests of the existing load_scalar/store_scalar ops with a
// registered V<->S hidden-event barrier (design doc 6.3.1).

struct PendingTExtractNd2xNz {
  pto::TExtractOp op;
  SmallVector<int64_t, 4> indices;      // [row0, col0, row1, col1]
  int64_t elemBytes = 0;
  int64_t srcRowStrideElems = 0;
  int64_t srcPhysicalRows = 0;
  int64_t srcPhysicalCols = 0;
  SmallVector<int64_t, 4> dstPhysical;  // [r0, c0, r1, c1]
  SmallVector<int64_t, 4> dstValid;     // [r0, c0, r1, c1]
};

static std::optional<PendingTExtractNd2xNz> snapshotTExtractNd2xNz(pto::TExtractOp op) {
  if (!op.isNdTo2xNzForm()) {
    return std::nullopt;
  }
  PendingTExtractNd2xNz pd;
  pd.op = op;
  auto indices = op.getIndices();
  if (indices.size() != 4) {
    return std::nullopt;
  }
  for (Value idx : indices) {
    auto c = getConstantIntValue(idx);
    if (!c) {
      return std::nullopt;
    }
    pd.indices.push_back(*c);
  }
  auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  if (!srcTy) {
    return std::nullopt;
  }
  pd.elemBytes = mlir::pto::getPTOStorageElemByteSize(srcTy.getElementType());
  if (pd.elemBytes == 0 || 32 % pd.elemBytes != 0) {
    return std::nullopt;
  }
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2) {
    return std::nullopt;
  }
  pd.srcPhysicalRows = srcShape[0];
  pd.srcPhysicalCols = srcShape[1];
  pd.srcRowStrideElems = srcShape[1];
  for (Value dst : op.getDsts()) {
    auto dstTy = dyn_cast<pto::TileBufType>(dst.getType());
    if (!dstTy) {
      return std::nullopt;
    }
    auto shape = dstTy.getShape();
    auto valid = dstTy.getValidShape();
    if (shape.size() != 2 || valid.size() != 2) {
      return std::nullopt;
    }
    pd.dstPhysical.push_back(shape[0]);
    pd.dstPhysical.push_back(shape[1]);
    pd.dstValid.push_back(valid[0]);
    pd.dstValid.push_back(valid[1]);
  }
  if (pd.dstPhysical.size() != 4 || pd.dstValid.size() != 4) {
    return std::nullopt;
  }
  return pd;
}

static Value buildNd2xNzIndex(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIndexOp>(loc, v);
}

// Load src[row][col] and store dst[valid coordinate] using scalar pointer ops.
static LogicalResult expandTExtractNd2xNzWindow(
    OpBuilder &builder, Location loc, const PendingTExtractNd2xNz &pd,
    unsigned window, Value srcPtr, Value dstPtr) {
  const int64_t c0 = 32 / pd.elemBytes;
  const int64_t rBase = pd.indices[window * 2];
  const int64_t cBase = pd.indices[window * 2 + 1];
  const int64_t validRows = pd.dstValid[window * 2];
  const int64_t validCols = pd.dstValid[window * 2 + 1];
  const int64_t physRows = pd.dstPhysical[window * 2];
  if (validRows <= 0 || validCols <= 0) {
    return failure();
  }

  auto cst = [&](OpBuilder &b, Location l, int64_t v) {
    return buildNd2xNzIndex(b, l, v);
  };
  auto srcTy = cast<pto::PtrType>(srcPtr.getType());
  auto elemTy = srcTy.getElementType();

  // Build a single flat loop over validRows*validCols elements, computing
  // row/col via div/rem; keeps the expansion to two scalar loop nests over
  // the existing load_scalar/store_scalar pointer ops (design doc 9.3).
  int64_t total = validRows * validCols;
  auto flat = builder.create<scf::ForOp>(
      loc, cst(builder, loc, 0), cst(builder, loc, total),
      cst(builder, loc, 1), ValueRange{});
  builder.setInsertionPointToStart(flat.getBody());
  {
    OpBuilder::InsertionGuard guard(builder);
    Value flatIdx = flat.getInductionVar();
    // r = flat / validCols; c = flat % validCols
    Value cIdx = builder.create<arith::RemUIOp>(loc, flatIdx, cst(builder, loc, validCols));
    Value rIdx = builder.create<arith::DivUIOp>(loc, flatIdx, cst(builder, loc, validCols));
    // srcOff = (rBase + r) * srcRowStrideElems + cBase + c
    Value rPlus = builder.create<arith::AddIOp>(loc, cst(builder, loc, rBase), rIdx);
    Value srcRowOff = builder.create<arith::MulIOp>(
        loc, rPlus, cst(builder, loc, pd.srcRowStrideElems));
    Value cPlus = builder.create<arith::AddIOp>(loc, cst(builder, loc, cBase), cIdx);
    Value srcOff = builder.create<arith::AddIOp>(loc, srcRowOff, cPlus);
    // dstOff = (c / c0) * physRows * c0 + r * c0 + (c % c0)
    Value cDiv = builder.create<arith::DivUIOp>(loc, cIdx, cst(builder, loc, c0));
    Value blockOff = builder.create<arith::MulIOp>(
        loc, cDiv, cst(builder, loc, physRows * c0));
    Value rBlock = builder.create<arith::MulIOp>(loc, rIdx, cst(builder, loc, c0));
    Value cRem = builder.create<arith::RemUIOp>(loc, cIdx, cst(builder, loc, c0));
    Value dstOff = builder.create<arith::AddIOp>(
        loc, builder.create<arith::AddIOp>(loc, blockOff, rBlock), cRem);
    Value loaded = builder.create<pto::LoadScalarOp>(loc, elemTy, srcPtr, srcOff);
    builder.create<pto::StoreScalarOp>(loc, dstPtr, dstOff, loaded);
  }
  builder.setInsertionPointAfter(flat);
  return success();
}

// Pick the synchronization event id for the hidden V<->S barrier of one
// ND-to-2xNz scalar expansion. TExtractOp's SyncMacroModel reserves event 0
// (design doc 6.3.1); when the event-id allocator ran (InsertSync), 0 is
// free for us because it is reserved. When it did not run, choose the first
// id not already used by explicit set_flag/wait_flag ops in the enclosing
// scope so the expansion never aliases a user literal event.
static unsigned selectUnusedHiddenEventId(Operation *anchor,
                                          MLIRContext *ctx) {
  llvm::SmallDenseSet<int32_t, 8> used;
  auto scan = [&](Operation *scope) {
    scope->walk([&](Operation *inner) {
      if (auto sf = dyn_cast<pto::SetFlagOp>(inner)) {
        if (auto ev = sf.getEventIdAttr())
          used.insert(static_cast<int32_t>(ev.getEvent()));
      } else if (auto wf = dyn_cast<pto::WaitFlagOp>(inner)) {
        if (auto ev = wf.getEventIdAttr())
          used.insert(static_cast<int32_t>(ev.getEvent()));
      }
    });
  };
  // Scan the enclosing function/module (exclude the op being expanded; it has
  // no flags yet).
  Operation *funcScope = anchor->getParentOp();
  if (funcScope)
    scan(funcScope);
  else
    scan(anchor);
  // kHiddenEventIdNum bounds the hidden-event id space; a straightforward
  // linear scan keeps determinism.
  constexpr unsigned kHiddenEventIdNum = 16;
  for (unsigned id = 0; id < kHiddenEventIdNum; ++id) {
    if (!used.count(static_cast<int32_t>(id)))
      return id;
  }
  return kHiddenEventIdNum - 1;
}

static LogicalResult expandTExtractNd2xNz(
    OpBuilder &builder, MLIRContext *ctx, const PendingTExtractNd2xNz &pd) {
  Operation *rawOp = pd.op;
  Location loc = rawOp->getLoc();
  // After allocation materialization the operands are !pto.ptr<..., ub>;
  // access them by raw operand position (design doc 9.3 step 2).
  Value srcPtr = rawOp->getOperand(0);
  Value dst0Ptr = rawOp->getOperand(5);
  Value dst1Ptr = rawOp->getOperand(6);

  // Registered hidden-event barrier around the scalar expansion (design doc
  // 6.3.1): TExtractOp's SyncMacroModel reserves event 0 for the V<->S hidden
  // pair during event-id allocation (InsertSync path). When the allocator did
  // not run (e.g. default VPTO lowering), pick the first event id not already
  // used explicitly in the enclosing scope so we never collide with a user
  // literal event (design doc 6.3.1: lowering must not register a conflicting
  // literal event).
  builder.setInsertionPoint(rawOp);
  auto pipeV = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
  auto pipeS = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_S);
  auto eventId =
      static_cast<pto::EVENT>(selectUnusedHiddenEventId(rawOp, ctx));
  auto event0 = pto::EventAttr::get(ctx, eventId);
  builder.create<pto::SetFlagOp>(loc, pipeV, pipeS, event0);
  builder.create<pto::WaitFlagOp>(loc, pipeV, pipeS, event0);

  if (failed(expandTExtractNd2xNzWindow(builder, loc, pd, 0, srcPtr, dst0Ptr)) ||
      failed(expandTExtractNd2xNzWindow(builder, loc, pd, 1, srcPtr, dst1Ptr))) {
    return failure();
  }

  builder.setInsertionPoint(rawOp);
  builder.create<pto::SetFlagOp>(loc, pipeS, pipeV, event0);
  builder.create<pto::WaitFlagOp>(loc, pipeS, pipeV, event0);

  rawOp->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerPTOToUBufOpsPass
    : public pto::impl::LowerPTOToUBufOpsBase<LowerPTOToUBufOpsPass> {
  using LowerPTOToUBufOpsBase::LowerPTOToUBufOpsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    auto mod = func->getParentOfType<ModuleOp>();
    if (!mod) {
      return;
    }
    auto archAttr = mod->getAttrOfType<StringAttr>("pto.target_arch");
    if (!archAttr ||
        (archAttr.getValue() != "a2" && archAttr.getValue() != "a3")) {
      return;
    }

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Snapshot dual-output TEXTRACT before allocation materialization loses
    // the tile types (design doc 9.3 step 1).
    SmallVector<PendingTExtractNd2xNz, 4> pendingNd2xNz;
    {
      SmallVector<pto::TExtractOp> extractOps;
      func.walk([&](pto::TExtractOp op) { extractOps.push_back(op); });
      for (pto::TExtractOp op : extractOps) {
        if (auto snapshot = snapshotTExtractNd2xNz(op)) {
          pendingNd2xNz.push_back(*snapshot);
        }
      }
    }

    // A2/A3: consume planned addresses from PTOPlanMemory / PTOMaterializeTileHandles.
    // Each alloc_tile must carry a planned addr operand.
    TileShapeMap tileShapes;
    {
      SmallVector<pto::AllocTileOp> allocOps;
      func.walk([&](pto::AllocTileOp op) { allocOps.push_back(op); });
      for (auto op : allocOps) {
        auto tbTy = cast<pto::TileBufType>(op.getResult().getType());
        if (llvm::any_of(tbTy.getValidShape(), ShapedType::isDynamic)) {
          op.emitError("A2/A3 UB lowering requires static valid_row and "
                       "valid_col");
          signalPassFailure();
          return;
        }
      }
      for (auto op : allocOps) {
        auto tbTy = cast<pto::TileBufType>(op.getResult().getType());
        auto shape = tbTy.getShape();
        Value addr = op.getAddr();
        if (!addr) {
          op.emitError("A3 VPTO UB lowering requires planned alloc_tile "
                       "addresses; run PTOViewToMemref, PTOPlanMemory, "
                       "PTOResolveReservedBuffers, and "
                       "PTOMaterializeTileHandles before LowerPTOToUBufOps");
          signalPassFailure();
          return;
        }
        builder.setInsertionPoint(op);
        auto ptrTy = pto::PtrType::get(
            ctx, tbTy.getElementType(),
            pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC));
        auto pc = builder.create<pto::CastPtrOp>(op.getLoc(), ptrTy, addr);
        tileShapes[pc.getResult()] = {
            SmallVector<int64_t, 2>(shape),
            SmallVector<int64_t, 2>(tbTy.getValidShape())};
        op.getResult().replaceAllUsesWith(pc.getResult());
        op.erase();
      }
    }

    // Expands dual-output TEXTRACT into scalar loop nests after allocation
    // materialization (design doc 9.3 steps 2-5): operands are now
    // !pto.ptr<..., ub>; the expansion consumes the op's registered hidden
    // event reservation and erases the op. Any leftover pointer-form
    // ND-to-2xNz TEXTRACT is rejected at the end of the pass.
    for (auto &pending : pendingNd2xNz) {
      if (failed(expandTExtractNd2xNz(builder, ctx, pending))) {
        pending.op.emitError("failed to expand ND-to-2xNz TEXTRACT for A2/A3 VPTO");
        signalPassFailure();
        return;
      }
    }
    func.walk([&](pto::TExtractOp op) {
      if (op.isNdTo2xNzForm()) {
        op.emitError("ND-to-2xNz TEXTRACT must be expanded before A2/A3 VPTO "
                     "LLVM emission; pointer-form pto.textract cannot survive");
        signalPassFailure();
      }
    });

    // ---- tadd → pto.ub.vadd ----
    {
      SmallVector<pto::TAddOp> ops;
      func.walk([&](pto::TAddOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVaddOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- taddrelu -> pto.ub.vaddrelu ----
    {
      SmallVector<pto::TAddReluOp> ops;
      func.walk([&](pto::TAddReluOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVaddReluOp>(op.getLoc(), builder, dstPtr, src0Ptr,
                                    src1Ptr, ptrType, *info);
        op.erase();
      }
    }

    // ---- tsub → pto.ub.vsub ----
    {
      SmallVector<pto::TSubOp> ops;
      func.walk([&](pto::TSubOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVsubOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tmul → pto.ub.vmul ----
    {
      SmallVector<pto::TMulOp> ops;
      func.walk([&](pto::TMulOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVmulOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tdiv → pto.ub.vdiv ----
    {
      SmallVector<pto::TDivOp> ops;
      func.walk([&](pto::TDivOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVdivOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tmax → pto.ub.vmax ----
    {
      SmallVector<pto::TMaxOp> ops;
      func.walk([&](pto::TMaxOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVmaxOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tmin → pto.ub.vmin ----
    {
      SmallVector<pto::TMinOp> ops;
      func.walk([&](pto::TMinOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVminOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tand → pto.ub.vand ----
    {
      SmallVector<pto::TAndOp> ops;
      func.walk([&](pto::TAndOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVandOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tor → pto.ub.vor ----
    {
      SmallVector<pto::TOrOp> ops;
      func.walk([&](pto::TOrOp op) { ops.push_back(op); });
      for (auto op : ops) {
        if (!canLower(op, tileShapes)) {
          continue;
        }
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, ptrType] = lowerBinaryOpCommon(
            builder, ctx, op, op.getDst(), op.getSrc0(), op.getSrc1(), tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatch<pto::UBVorOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- txor → vor(tmp) + vand(dst) + vnot(dst) + vand(dst,tmp) ----
    // De Morgan: src0 ^ src1 = ~(src0 & src1) & (src0 | src1)
    {
      SmallVector<pto::TXorOp> ops;
      func.walk([&](pto::TXorOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, src0Ptr, src1Ptr, tmpPtr, ptrType] =
            lowerXorOpCommon(builder, ctx, op, op.getDst(), op.getSrc0(),
                             op.getSrc1(), op.getTmp(), tileShapes);
        if (!dstPtr || !tmpPtr) {
          continue;
        }
        auto pipeV = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
        // tmp = src0 | src1
        dispatch<pto::UBVorOp>(op.getLoc(), builder, tmpPtr, src0Ptr, src1Ptr,
                               ptrType, *info);
        builder.create<pto::BarrierOp>(op.getLoc(), pipeV);
        // dst = src0 & src1
        dispatch<pto::UBVandOp>(op.getLoc(), builder, dstPtr, src0Ptr, src1Ptr,
                                ptrType, *info);
        builder.create<pto::BarrierOp>(op.getLoc(), pipeV);
        // dst = ~dst
        dispatchUnary<pto::UBVnotOp>(op.getLoc(), builder, dstPtr, dstPtr,
                                     ptrType, *info);
        builder.create<pto::BarrierOp>(op.getLoc(), pipeV);
        // dst = dst & tmp
        dispatch<pto::UBVandOp>(op.getLoc(), builder, dstPtr, dstPtr, tmpPtr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- tnot → pto.ub.vnot ----
    {
      SmallVector<pto::TNotOp> ops;
      func.walk([&](pto::TNotOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVnotOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                     ptrType, *info);
        op.erase();
      }
    }

    // ---- tabs → pto.ub.vabs ----
    {
      SmallVector<pto::TAbsOp> ops;
      func.walk([&](pto::TAbsOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVabsOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                     ptrType, *info);
        op.erase();
      }
    }

    // ---- trelu → pto.ub.vrelu ----
    {
      SmallVector<pto::TReluOp> ops;
      func.walk([&](pto::TReluOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVreluOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      ptrType, *info);
        op.erase();
      }
    }

    // ---- tneg → pto.ub.vmuls(dst, src, -1) ----
    {
      SmallVector<pto::TNegOp> ops;
      func.walk([&](pto::TNegOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        Type elemTy = ptrType.getElementType();
        Value minusOneScalar;
        if (elemTy.isF32() || elemTy.isF16()) {
          minusOneScalar = builder.create<arith::ConstantOp>(
              op.getLoc(), builder.getFloatAttr(elemTy, -1.0));
        } else {
          minusOneScalar = builder.create<arith::ConstantOp>(
              op.getLoc(), builder.getIntegerAttr(elemTy, -1));
        }
        Value minusOne =
            convertScalarToI64(builder, op.getLoc(), minusOneScalar);
        dispatchShift<pto::UBVmulSOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      minusOne, ptrType, *info);
        op.erase();
      }
    }

    // ---- trecip → vector_dup(dst, 1) + vdiv(dst, dst, src) ----
    {
      SmallVector<pto::TRecipOp> ops;
      func.walk([&](pto::TRecipOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        Type elemTy = ptrType.getElementType();
        Value oneScalar = builder.create<arith::ConstantOp>(
            op.getLoc(), builder.getFloatAttr(elemTy, 1.0));
        Value one = convertScalarToI64(builder, op.getLoc(), oneScalar);
        dispatchDup(op.getLoc(), builder, dstPtr, one, ptrType, *info);
        builder.create<pto::BarrierOp>(op.getLoc(),
                                       pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V));
        dispatch<pto::UBVdivOp>(op.getLoc(), builder, dstPtr, dstPtr, srcPtr,
                                ptrType, *info);
        op.erase();
      }
    }

    // ---- texp → pto.ub.vexp ----
    {
      SmallVector<pto::TExpOp> ops;
      func.walk([&](pto::TExpOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVexpOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                     ptrType, *info);
        op.erase();
      }
    }

    // ---- tlog → pto.ub.vln ----
    {
      SmallVector<pto::TLogOp> ops;
      func.walk([&](pto::TLogOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVlnOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                    ptrType, *info);
        op.erase();
      }
    }

    // ---- tsqrt → pto.ub.vsqrt ----
    {
      SmallVector<pto::TSqrtOp> ops;
      func.walk([&](pto::TSqrtOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVsqrtOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      ptrType, *info);
        op.erase();
      }
    }

    // ---- trsqrt → pto.ub.vrsqrt ----
    {
      SmallVector<pto::TRsqrtOp> ops;
      func.walk([&](pto::TRsqrtOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfoFromValue(op.getDst(), tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchUnary<pto::UBVrsqrtOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                       ptrType, *info);
        op.erase();
      }
    }

    // ---- tadds → pto.ub.vadds ----
    {
      SmallVector<pto::TAddSOp> ops;
      func.walk([&](pto::TAddSOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        Value scalarI64 = convertScalarToI64(builder, op.getLoc(), op.getScalar());
        dispatchShift<pto::UBVaddSOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      scalarI64, ptrType, *info);
        op.erase();
      }
    }

    // ---- tmuls → pto.ub.vmuls ----
    {
      SmallVector<pto::TMulSOp> ops;
      func.walk([&](pto::TMulSOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc0(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        Value scalarI64 = convertScalarToI64(builder, op.getLoc(), op.getScalar());
        dispatchShift<pto::UBVmulSOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      scalarI64, ptrType, *info);
        op.erase();
      }
    }

    // ---- tmaxs → pto.ub.vmaxs ----
    {
      SmallVector<pto::TMaxSOp> ops;
      func.walk([&](pto::TMaxSOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        Value scalarI64 = convertScalarToI64(builder, op.getLoc(), op.getScalar());
        dispatchShift<pto::UBVmaxSOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      scalarI64, ptrType, *info);
        op.erase();
      }
    }

    // ---- tmins → pto.ub.vmins ----
    {
      SmallVector<pto::TMinSOp> ops;
      func.walk([&](pto::TMinSOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        Value scalarI64 = convertScalarToI64(builder, op.getLoc(), op.getScalar());
        dispatchShift<pto::UBVminSOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                      scalarI64, ptrType, *info);
        op.erase();
      }
    }

    // ---- tshls → pto.ub.vshl (scalar shift) ----
    {
      SmallVector<pto::TShlSOp> ops;
      func.walk([&](pto::TShlSOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchShift<pto::UBVshlOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                     op.getScalar(), ptrType, *info);
        op.erase();
      }
    }

    // ---- tshrs → pto.ub.vshr (scalar shift) ----
    {
      SmallVector<pto::TShrSOp> ops;
      func.walk([&](pto::TShrSOp op) { ops.push_back(op); });
      for (auto op : ops) {
        auto info = extractTileShapeInfo(op, tileShapes);
        if (!info) {
          continue;
        }
        auto [dstPtr, srcPtr, ptrType] =
            lowerShiftOpCommon(builder, ctx, op, op.getDst(), op.getSrc(),
                               tileShapes);
        if (!dstPtr) {
          continue;
        }
        dispatchShift<pto::UBVshrOp>(op.getLoc(), builder, dstPtr, srcPtr,
                                     op.getScalar(), ptrType, *info);
        op.erase();
      }
    }

    // ---- tload → mte_gm_ub ----
    SmallVector<pto::TLoadOp> tloadOps;
    func.walk([&](pto::TLoadOp op) { tloadOps.push_back(op); });
    for (auto op : tloadOps) {
      builder.setInsertionPoint(op);
      if (succeeded(lowerTLoad(op, builder, tileShapes))) {
        op.erase();
      }
    }

    // ---- tstore → mte_ub_gm ----
    SmallVector<pto::TStoreOp> tstoreOps;
    func.walk([&](pto::TStoreOp op) { tstoreOps.push_back(op); });
    for (auto op : tstoreOps) {
      builder.setInsertionPoint(op);
      if (succeeded(lowerTStore(op, builder, tileShapes))) {
        op.erase();
      }
    }

    // ---- cleanup dead PTO ops ----
    SmallVector<Operation *> toErase;
    func.walk([&](Operation *op) {
      if (isa<pto::PartitionViewOp, pto::MakeTensorViewOp,
              memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp>(op)) {
        toErase.push_back(op);
      }
    });
    for (auto *op : llvm::reverse(toErase)) {
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

private:
  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  Value i64c(int64_t val, Location loc, OpBuilder &b) {
    return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(val));
  }
  Value idxc(int64_t val, Location loc, OpBuilder &b) {
    return b.create<arith::ConstantOp>(
               loc, b.getIntegerAttr(b.getIndexType(), val))
        .getResult();
  }
  Value i64c0(Location loc, OpBuilder &b) { return i64c(0, loc, b); }
  Value i64c1(Location loc, OpBuilder &b) { return i64c(1, loc, b); }
  Value i64cM1(Location loc, OpBuilder &b) { return i64c(-1, loc, b); }
  Value i64c8(Location loc, OpBuilder &b) { return i64c(kDefaultRepeatStride, loc, b); }
  Value idxc0(Location loc, OpBuilder &b) { return idxc(0, loc, b); }
  Value idxc1(Location loc, OpBuilder &b) { return idxc(1, loc, b); }

  template <typename UBop>
  void emitUBBinOp(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   Value repeat, Value repStride) {
    b.create<UBop>(loc, dst, s0, s1, repeat,
                   i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                   repStride, repStride, i64c0(loc, b));
  }

  void vadd(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVaddOp>(loc, b, dst, s0, s1, repeat, repStride);
  }
  void vsub(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVsubOp>(loc, b, dst, s0, s1, repeat, repStride);
  }
  void vmul(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVmulOp>(loc, b, dst, s0, s1, repeat, repStride);
  }
  void vdiv(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
            Value repeat, Value repStride) {
    emitUBBinOp<pto::UBVdivOp>(loc, b, dst, s0, s1, repeat, repStride);
  }

  std::tuple<Value, Value, Value, pto::PtrType>
  lowerBinaryOpCommon(OpBuilder &builder, MLIRContext *ctx, Operation *op,
                      Value dstVal, Value src0Val, Value src1Val,
                      const TileShapeMap &tileShapes) {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);
    Type elemTy = getStoredElemType(dstVal.getType());
    auto ptrType = getUBPtrType(ctx, elemTy);

    auto emitAddr = [&](Value tile) -> Value {
      if (isa<pto::PtrType>(tile.getType())) {
        return tile;
      }
      auto addrOp = builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
      return addrOp.getDst();
    };

    Value dstPtr = emitAddr(dstVal);
    Value src0Ptr = emitAddr(src0Val);
    Value src1Ptr = emitAddr(src1Val);
    return {dstPtr, src0Ptr, src1Ptr, ptrType};
  }

  std::tuple<Value, Value, Value, Value, pto::PtrType>
  lowerXorOpCommon(OpBuilder &builder, MLIRContext *ctx, Operation *op,
                   Value dstVal, Value src0Val, Value src1Val, Value tmpVal,
                   const TileShapeMap &tileShapes) {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);
    Type elemTy = getStoredElemType(dstVal.getType());
    auto ptrType = getUBPtrType(ctx, elemTy);

    auto emitAddr = [&](Value tile) -> Value {
      if (isa<pto::PtrType>(tile.getType())) {
        return tile;
      }
      auto addrOp = builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
      return addrOp.getDst();
    };

    Value dstPtr = emitAddr(dstVal);
    Value src0Ptr = emitAddr(src0Val);
    Value src1Ptr = emitAddr(src1Val);
    Value tmpPtr = emitAddr(tmpVal);
    return {dstPtr, src0Ptr, src1Ptr, tmpPtr, ptrType};
  }

  Value convertScalarToI64(OpBuilder &builder, Location loc, Value scalar) {
    if (scalar.getType().isF32() || scalar.getType().isF16()) {
      unsigned width = scalar.getType().isF32() ? 32 : 16;
      auto intTy = builder.getIntegerType(width);
      Value asInt = builder.create<arith::BitcastOp>(loc, intTy, scalar);
      return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), asInt);
    }
    if (scalar.getType().isInteger(mlir::pto::kValue64)) {
      return scalar;
    }
    return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), scalar);
  }

  std::tuple<Value, Value, pto::PtrType>
  lowerShiftOpCommon(OpBuilder &builder, MLIRContext *ctx, Operation *op,
                     Value dstVal, Value srcVal, const TileShapeMap &tileShapes) {
    Location loc = op->getLoc();
    builder.setInsertionPoint(op);
    Type elemTy = getStoredElemType(dstVal.getType());
    auto ptrType = getUBPtrType(ctx, elemTy);

    auto emitAddr = [&](Value tile) -> Value {
      if (isa<pto::PtrType>(tile.getType())) {
        return tile;
      }
      auto addrOp = builder.create<pto::TileBufAddrOp>(loc, ptrType, tile);
      return addrOp.getDst();
    };

    Value dstPtr = emitAddr(dstVal);
    Value srcPtr = emitAddr(srcVal);
    return {dstPtr, srcPtr, ptrType};
  }

  template <typename UBop>
  void dispatchShift(Location loc, OpBuilder &b, Value dst, Value src,
                     Value scalar, pto::PtrType ptrTy,
                     const TileShapeInfo &info) {
    if (info.vRows > 1 && info.vCols != info.cols) {
      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(info.vRows, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value off = b.create<arith::MulIOp>(
          loc, forOp.getInductionVar(), idxc(info.cols, loc, b));
      TileShapeInfo rowInfo = info;
      rowInfo.rows = 1;
      rowInfo.vRows = 1;
      dispatchShift<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
                          addPtr(loc, b, src, ptrTy, off), scalar, ptrTy,
                          rowInfo);
      b.setInsertionPointAfter(forOp);
      return;
    }
    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    auto emitShift = [this, loc, &b, scalar](Value d, Value s) {
      Value scalarI64 = scalar;
      if (scalarI64.getType() != b.getI64Type()) {
        scalarI64 = b.create<arith::ExtSIOp>(
            loc, b.getI64Type(), scalar);
      }
      b.create<UBop>(loc, d, s, scalarI64,
                     i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                     i64c8(loc, b), i64c8(loc, b));
    };

    if (headRepeats > 1 || tailElements > 0) {
      if (headRepeats > 0) {
        auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                          idxc(headRepeats, loc, b), idxc1(loc, b));
        b.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value off = b.create<arith::MulIOp>(loc, iv, idxc(epr, loc, b)).getResult();
        Value rd = addPtr(loc, b, dst, ptrTy, off);
        Value r0 = addPtr(loc, b, src, ptrTy, off);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(epr, loc, b), i64c0(loc, b));
        emitShift(rd, r0);
        b.create<pto::UBSetMaskNormOp>(loc);
        b.setInsertionPointAfter(forOp);
      }
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, loc, b);
        Value td = addPtr(loc, b, dst, ptrTy, offT);
        Value ts0 = addPtr(loc, b, src, ptrTy, offT);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(tailElements, loc, b), i64c0(loc, b));
        emitShift(td, ts0);
        b.create<pto::UBSetMaskNormOp>(loc);
      }
      fullMask(loc, b);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emitShift(dst, src);
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  template <typename UBop>
  void dispatchUnary(Location loc, OpBuilder &b, Value dst, Value src,
                     pto::PtrType ptrTy, const TileShapeInfo &info) {
    if (info.vRows > 1 && info.vCols != info.cols) {
      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(info.vRows, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value off = b.create<arith::MulIOp>(
          loc, forOp.getInductionVar(), idxc(info.cols, loc, b));
      TileShapeInfo rowInfo = info;
      rowInfo.rows = 1;
      rowInfo.vRows = 1;
      dispatchUnary<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
                          addPtr(loc, b, src, ptrTy, off), ptrTy, rowInfo);
      b.setInsertionPointAfter(forOp);
      return;
    }
    auto emit = [this, loc, &b](Value rd, Value rs) {
      b.create<UBop>(loc, rd, rs,
                     i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                     i64c8(loc, b), i64c8(loc, b));
    };
    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 1 || tailElements > 0) {
      if (headRepeats > 0) {
        auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                          idxc(headRepeats, loc, b), idxc1(loc, b));
        b.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value off = b.create<arith::MulIOp>(loc, iv, idxc(epr, loc, b)).getResult();
        Value rd = addPtr(loc, b, dst, ptrTy, off);
        Value rs = addPtr(loc, b, src, ptrTy, off);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(epr, loc, b), i64c0(loc, b));
        emit(rd, rs);
        b.create<pto::UBSetMaskNormOp>(loc);
        b.setInsertionPointAfter(forOp);
      }
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, loc, b);
        Value td = addPtr(loc, b, dst, ptrTy, offT);
        Value ts = addPtr(loc, b, src, ptrTy, offT);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(tailElements, loc, b), i64c0(loc, b));
        emit(td, ts);
        b.create<pto::UBSetMaskNormOp>(loc);
      }
      fullMask(loc, b);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emit(dst, src);
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  void dispatchDup(Location loc, OpBuilder &b, Value dst, Value scalar,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    if (info.vRows > 1 && info.vCols != info.cols) {
      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(info.vRows, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value off = b.create<arith::MulIOp>(
          loc, forOp.getInductionVar(), idxc(info.cols, loc, b));
      TileShapeInfo rowInfo = info;
      rowInfo.rows = 1;
      rowInfo.vRows = 1;
      dispatchDup(loc, b, addPtr(loc, b, dst, ptrTy, off), scalar, ptrTy,
                  rowInfo);
      b.setInsertionPointAfter(forOp);
      return;
    }
    auto emit = [this, loc, &b, scalar](Value rd) {
      Value scalarI64 = scalar;
      if (scalarI64.getType() != b.getI64Type()) {
        scalarI64 = b.create<arith::ExtSIOp>(loc, b.getI64Type(), scalar);
      }
      b.create<pto::UBVdupOp>(loc, rd, scalarI64,
                              i64c1(loc, b), i64c1(loc, b), i64c1(loc, b),
                              i64c8(loc, b), i64c0(loc, b));
    };

    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 1 || tailElements > 0) {
      if (headRepeats > 0) {
        auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                          idxc(headRepeats, loc, b), idxc1(loc, b));
        b.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value off = b.create<arith::MulIOp>(loc, iv, idxc(epr, loc, b)).getResult();
        Value rd = addPtr(loc, b, dst, ptrTy, off);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(epr, loc, b), i64c0(loc, b));
        emit(rd);
        b.create<pto::UBSetMaskNormOp>(loc);
        b.setInsertionPointAfter(forOp);
      }
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, loc, b);
        Value td = addPtr(loc, b, dst, ptrTy, offT);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(tailElements, loc, b), i64c0(loc, b));
        emit(td);
        b.create<pto::UBSetMaskNormOp>(loc);
      }
      fullMask(loc, b);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emit(dst);
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  template <typename UBop>
  void modeNorm1L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t totalV = info.vRows * info.vCols;
    int64_t headRepeats = totalV / epr;
    int64_t tailElements = totalV % epr;

    if (headRepeats > 1 || tailElements > 0) {
      if (headRepeats > 0) {
        auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                          idxc(headRepeats, loc, b), idxc1(loc, b));
        b.setInsertionPointToStart(forOp.getBody());
        Value iv = forOp.getInductionVar();
        Value off = b.create<arith::MulIOp>(loc, iv, idxc(epr, loc, b)).getResult();
        Value rd = addPtr(loc, b, dst, ptrTy, off);
        Value r0 = addPtr(loc, b, s0, ptrTy, off);
        Value r1 = addPtr(loc, b, s1, ptrTy, off);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(epr, loc, b), i64c0(loc, b));
        emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c1(loc, b), i64c8(loc, b));
        b.create<pto::UBSetMaskNormOp>(loc);
        b.setInsertionPointAfter(forOp);
      }
      if (tailElements > 0) {
        Value offT = idxc(headRepeats * epr, loc, b);
        Value td = addPtr(loc, b, dst, ptrTy, offT);
        Value ts0 = addPtr(loc, b, s0, ptrTy, offT);
        Value ts1 = addPtr(loc, b, s1, ptrTy, offT);
        b.create<pto::UBSetMaskCountOp>(loc);
        b.create<pto::UBSetMaskOp>(loc, i64c(tailElements, loc, b), i64c0(loc, b));
        emitUBBinOp<UBop>(loc, b, td, ts0, ts1, i64c1(loc, b), i64c8(loc, b));
        b.create<pto::UBSetMaskNormOp>(loc);
      }
      fullMask(loc, b);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(totalV, loc, b), i64c0(loc, b));
    emitUBBinOp<UBop>(loc, b, dst, s0, s1, i64c1(loc, b), i64c8(loc, b));
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  void setMask(Location loc, OpBuilder &b, unsigned n) {
    auto [m0, m1] = computeContMaskValues(n);
    b.create<pto::UBSetMaskOp>(loc, i64c(m0, loc, b), i64c(m1, loc, b));
  }

  void fullMask(Location loc, OpBuilder &b) {
    b.create<pto::UBSetMaskOp>(loc, i64cM1(loc, b), i64cM1(loc, b));
  }

  Value addPtr(Location loc, OpBuilder &b, Value base, pto::PtrType ptrTy,
                Value off) {
    return b.create<pto::AddPtrOp>(loc, ptrTy, base, off);
  }

  //===--------------------------------------------------------------------===//
  // tload → mte_gm_ub / tstore → mte_ub_gm
  //===--------------------------------------------------------------------===//

  struct DmaViewInfo {
    Value gmPtr;
    Value linearOffset;
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    SmallVector<Value> offsets;
  };

  static bool hasUnitInnermostStride(Value view) {
    while (Operation *def = view.getDefiningOp()) {
      if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
        auto strides = subview.getStaticStrides();
        if (strides.empty() || strides.back() != 1) {
          return false;
        }
        view = subview.getSource();
        continue;
      }
      if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
        auto strides = reinterpret.getConstifiedMixedStrides();
        if (strides.empty()) {
          return false;
        }
        auto stride = getConstantIntValue(strides.back());
        return stride && *stride == 1;
      }
      if (auto cast = dyn_cast<memref::CastOp>(def)) {
        view = cast.getSource();
        continue;
      }
      break;
    }
    auto memTy = dyn_cast<MemRefType>(view.getType());
    if (!memTy) {
      return false;
    }
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(pto::getPTOMemRefStridesAndOffset(memTy, strides, offset)))
      return false;
    return !strides.empty() && strides.back() == 1;
  }

  static FailureOr<DmaViewInfo> extractDmaViewInfo(pto::TLoadOp op) {
    auto pvOp = op.getSrc().getDefiningOp<pto::PartitionViewOp>();
    if (pvOp) {
      auto mtvOp = pvOp.getSource().getDefiningOp<pto::MakeTensorViewOp>();
      if (!mtvOp) {
        return failure();
      }
      DmaViewInfo info;
      info.gmPtr = mtvOp.getPtr();
      info.sizes.assign(pvOp.getSizes().begin(), pvOp.getSizes().end());
      info.strides.assign(mtvOp.getStrides().begin(),
                          mtvOp.getStrides().end());
      if (info.strides.empty() ||
          !matchPattern(info.strides.back(), m_One())) {
        op.emitError("A2/A3 DMA lowering requires a unit innermost stride");
        return failure();
      }
      info.offsets.assign(pvOp.getOffsets().begin(), pvOp.getOffsets().end());
      return info;
    }
    return extractDmaMemRefViewInfo(op.getLoc(), op.getSrc(), op.getContext());
  }

  static FailureOr<DmaViewInfo> extractDmaViewInfo(pto::TStoreOp op) {
    auto pvOp = op.getDst().getDefiningOp<pto::PartitionViewOp>();
    if (pvOp) {
      auto mtvOp = pvOp.getSource().getDefiningOp<pto::MakeTensorViewOp>();
      if (!mtvOp) {
        return failure();
      }
      DmaViewInfo info;
      info.gmPtr = mtvOp.getPtr();
      info.sizes.assign(pvOp.getSizes().begin(), pvOp.getSizes().end());
      info.strides.assign(mtvOp.getStrides().begin(), mtvOp.getStrides().end());
      if (info.strides.empty() ||
          !matchPattern(info.strides.back(), m_One())) {
        op.emitError("A2/A3 DMA lowering requires a unit innermost stride");
        return failure();
      }
      info.offsets.assign(pvOp.getOffsets().begin(), pvOp.getOffsets().end());
      return info;
    }
    return extractDmaMemRefViewInfo(op.getLoc(), op.getDst(), op.getContext());
  }

  static FailureOr<DmaViewInfo> extractDmaMemRefViewInfo(Location loc, Value view,
                                                         MLIRContext *ctx) {
    auto memTy = dyn_cast<MemRefType>(view.getType());
    if (!memTy) {
      return failure();
    }
    auto msAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memTy.getMemorySpace());
    if (!msAttr || msAttr.getAddressSpace() != pto::AddressSpace::GM) {
      return failure();
    }
    ArrayRef<int64_t> shape = memTy.getShape();
    if (shape.size() < mlir::pto::kValue2) {
      return failure();
    }
    if (!hasUnitInnermostStride(view)) {
      emitError(loc) << "A2/A3 DMA lowering requires a unit innermost stride";
      return failure();
    }

    OpBuilder b(ctx);
    b.setInsertionPointAfterValue(view);
    DmaViewInfo info;
    auto ptrTy = pto::PtrType::get(ctx, memTy.getElementType(), msAttr);
    auto metadata = b.create<memref::ExtractStridedMetadataOp>(loc, view);
    info.gmPtr = b.create<pto::CastPtrOp>(loc, ptrTy, traceRootMemRef(view));
    info.linearOffset = metadata.getOffset();
    info.sizes.assign(metadata.getSizes().begin(), metadata.getSizes().end());
    info.strides.assign(metadata.getStrides().begin(),
                        metadata.getStrides().end());
    return info;
  }

  static Value traceRootMemRef(Value value) {
    while (Operation *def = value.getDefiningOp()) {
      if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
        value = subview.getSource();
        continue;
      }
      if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
        value = reinterpret.getSource();
        continue;
      }
      if (auto cast = dyn_cast<memref::CastOp>(def)) {
        value = cast.getSource();
        continue;
      }
      break;
    }
    return value;
  }

  Value computeGMByteOffset(Location loc, OpBuilder &b,
                            const DmaViewInfo &viewInfo, unsigned elemSize) {
    if (viewInfo.linearOffset) {
      Value offset = b.create<arith::IndexCastOp>(
          loc, b.getI64Type(), viewInfo.linearOffset);
      return b.create<arith::MulIOp>(loc, offset,
                                     i64c(elemSize, loc, b));
    }
    Value totalOff = idxc0(loc, b);
    for (size_t i = 0; i < viewInfo.offsets.size() &&
                        i < viewInfo.strides.size(); ++i) {
      APInt constOff;
      if (matchPattern(viewInfo.offsets[i], m_ConstantInt(&constOff)) &&
          constOff.isZero()) {
        continue;
      }
      Value dimOff = b.create<arith::MulIOp>(loc, viewInfo.offsets[i],
                                             viewInfo.strides[i]).getResult();
      totalOff = b.create<arith::AddIOp>(loc, totalOff, dimOff).getResult();
    }
    if (elemSize > 1) {
      totalOff = b.create<arith::MulIOp>(loc, totalOff,
                                         idxc(elemSize, loc, b)).getResult();
    }
    return totalOff;
  }

  Value offsetGMPtrByBytes(Location loc, OpBuilder &b, Value gmPtr,
                           Value byteOff) {
    APInt constOff;
    if (matchPattern(byteOff, m_ConstantInt(&constOff)) && constOff.isZero()) {
      return gmPtr;
    }
    auto origPtrTy = cast<pto::PtrType>(gmPtr.getType());
    auto bytePtrTy = pto::PtrType::get(b.getContext(), b.getI8Type(),
                                       origPtrTy.getMemorySpace());
    Value bytePtr = b.create<pto::CastPtrOp>(loc, bytePtrTy, gmPtr);
    Value offIdx = byteOff;
    if (!offIdx.getType().isIndex()) {
      offIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), byteOff)
                   .getResult();
    }
    Value offsetBytePtr =
        b.create<pto::AddPtrOp>(loc, bytePtrTy, bytePtr, offIdx);
    return b.create<pto::CastPtrOp>(loc, origPtrTy, offsetBytePtr);
  }

  Value i64Cast(Location loc, OpBuilder &b, Value indexVal) {
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), indexVal)
        .getResult();
  }

  LogicalResult emitMteGmUb(Location loc, OpBuilder &b, Value gmPtr,
                             Value ubPtr, const DmaViewInfo &viewInfo,
                             Type elemTy, ArrayRef<int64_t> tileShape) {
    if (tileShape.size() < mlir::pto::kValue2) {
      return failure();
    }
    int64_t ubCols = tileShape[1];
    unsigned elemSize = getElementSize(elemTy);
    unsigned nd = viewInfo.sizes.size();
    if (nd < mlir::pto::kValue2) {
      return failure();
    }

    Value nburstCount = viewInfo.sizes[nd - 2];
    Value lenBurstElts = viewInfo.sizes[nd - 1];
    Value lenBurst = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(), lenBurstElts)
            .getResult(), i64c(elemSize, loc, b)).getResult();
    Value nburstSrcStride = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.strides[nd - 2]).getResult(),
        i64c(elemSize, loc, b)).getResult();
    Value ubRowStride = b.create<arith::MulIOp>(loc, i64c(ubCols, loc, b),
                                                i64c(elemSize, loc, b)).getResult();
    pto::DmaLoopConfig nburst{i64Cast(loc, b, nburstCount),
                              nburstSrcStride, ubRowStride};

    SmallVector<pto::DmaLoopConfig> loops;
    for (int i = nd - 3; i >= 0; --i) {
      Value count = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
          viewInfo.sizes[i]).getResult();
      Value srcStride = b.create<arith::MulIOp>(loc,
          b.create<arith::IndexCastOp>(loc, b.getI64Type(),
              viewInfo.strides[i]).getResult(),
          i64c(elemSize, loc, b)).getResult();
      Value innerElems = i64c1(loc, b);
      for (int j = i + 1; j < static_cast<int>(nd); ++j) {
        Value dimSize = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.sizes[j]).getResult();
        innerElems = b.create<arith::MulIOp>(loc, innerElems, dimSize).getResult();
      }
      Value dstStride = b.create<arith::MulIOp>(loc, innerElems,
          i64c(elemSize, loc, b)).getResult();
      loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteGmUbOp>(loc, gmPtr, ubPtr, i64c0(loc, b), lenBurst,
        nburst, llvm::ArrayRef(loops), std::nullopt);
    return success();
  }

  LogicalResult emitMteUbGm(Location loc, OpBuilder &b, Value ubPtr,
                             Value gmPtr, const DmaViewInfo &viewInfo,
                             Type elemTy, ArrayRef<int64_t> tileShape) {
    if (tileShape.size() < mlir::pto::kValue2) {
      return failure();
    }
    int64_t ubCols = tileShape[1];
    unsigned elemSize = getElementSize(elemTy);
    unsigned nd = viewInfo.sizes.size();
    if (nd < mlir::pto::kValue2) {
      return failure();
    }

    Value nburstCount = viewInfo.sizes[nd - 2];
    Value lenBurstElts = viewInfo.sizes[nd - 1];
    Value lenBurst = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(), lenBurstElts)
            .getResult(), i64c(elemSize, loc, b)).getResult();
    Value nburstSrcStride = b.create<arith::MulIOp>(loc,
        i64c(ubCols, loc, b), i64c(elemSize, loc, b)).getResult();
    Value nburstDstStride = b.create<arith::MulIOp>(loc,
        b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.strides[nd - 2]).getResult(),
        i64c(elemSize, loc, b)).getResult();
    pto::DmaLoopConfig nburst{i64Cast(loc, b, nburstCount),
                              nburstSrcStride, nburstDstStride};

    SmallVector<pto::DmaLoopConfig> loops;
    for (int i = nd - 3; i >= 0; --i) {
      Value count = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
          viewInfo.sizes[i]).getResult();
      Value innerElems = i64c1(loc, b);
      for (int j = i + 1; j < static_cast<int>(nd); ++j) {
        Value dimSize = b.create<arith::IndexCastOp>(loc, b.getI64Type(),
            viewInfo.sizes[j]).getResult();
        innerElems = b.create<arith::MulIOp>(loc, innerElems, dimSize).getResult();
      }
      Value srcStride = b.create<arith::MulIOp>(loc, innerElems,
          i64c(elemSize, loc, b)).getResult();
      Value dstStride = b.create<arith::MulIOp>(loc,
          b.create<arith::IndexCastOp>(loc, b.getI64Type(),
              viewInfo.strides[i]).getResult(),
          i64c(elemSize, loc, b)).getResult();
      loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteUbGmOp>(loc, ubPtr, gmPtr, lenBurst, nburst, Value{},
                             llvm::ArrayRef(loops));
    return success();
  }

  LogicalResult lowerTLoad(pto::TLoadOp op, OpBuilder &b,
                           const TileShapeMap &tileShapes) {
    Location loc = op.getLoc();
    auto viewInfo = extractDmaViewInfo(op);
    if (failed(viewInfo)) {
      return failure();
    }
    Type dstType = op.getDst().getType();
    if (!isUBMemorySpace(dstType)) {
      return failure();
    }
    Type elemTy = getStoredElemType(dstType);
    if (!elemTy) {
      return failure();
    }
    unsigned elemSize = getElementSize(elemTy);
    if (elemSize == 0) {
      return failure();
    }

    auto it = tileShapes.find(op.getDst());
    if (it == tileShapes.end()) {
      return failure();
    }

    Value byteOff = computeGMByteOffset(loc, b, *viewInfo, elemSize);
    Value gmPtr = offsetGMPtrByBytes(loc, b, viewInfo->gmPtr, byteOff);
    return emitMteGmUb(loc, b, gmPtr, op.getDst(), *viewInfo, elemTy,
                       llvm::ArrayRef(it->second.shape));
  }

  LogicalResult lowerTStore(pto::TStoreOp op, OpBuilder &b,
                            const TileShapeMap &tileShapes) {
    Location loc = op.getLoc();
    Type srcType = op.getSrc().getType();
    if (!isUBMemorySpace(srcType)) {
      return failure();
    }
    Type elemTy = getStoredElemType(srcType);
    if (!elemTy) {
      return failure();
    }
    unsigned elemSize = getElementSize(elemTy);
    if (elemSize == 0) {
      return failure();
    }

    auto it = tileShapes.find(op.getSrc());
    if (it == tileShapes.end()) {
      return failure();
    }

    auto viewInfo = extractDmaViewInfo(op);
    if (failed(viewInfo)) {
      return failure();
    }

    Value byteOff = computeGMByteOffset(loc, b, *viewInfo, elemSize);
    Value gmPtr = offsetGMPtrByBytes(loc, b, viewInfo->gmPtr, byteOff);
    return emitMteUbGm(loc, b, op.getSrc(), gmPtr, *viewInfo, elemTy,
                       llvm::ArrayRef(it->second.shape));
  }

  //===--------------------------------------------------------------------===//
  // CCE dispatch tree — mirrors TBinOp.hpp BinaryInstr
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void dispatch(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t cols = info.cols;
    int64_t rows = info.rows;
    int64_t vRows = info.vRows;
    int64_t vCols = info.vCols;

    // 1. Small tile
    if (rows <= kRepeatMax && cols < static_cast<int64_t>(epr)) {
      modeSmall<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      return;
    }

    // 2. Continuous at compile time
    if (vCols == cols || vRows == 1) {
      int64_t totalV = vRows * vCols;
      int64_t totalRpts = (totalV + epr - 1) / epr;

      if (totalRpts > kRepeatMax) {
        modeCount1L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      } else {
        modeNorm1L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      }
      return;
    }

    // 3. Non-continuous
    int64_t normColRepeat = cols / epr;
    if (normColRepeat > 1 && vRows * normColRepeat < kSmallRptBinOp) {
      modeCount2L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
    } else if (vRows < normColRepeat + 1) {
      if (vCols % epr > 0) {
        modeCount2L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      } else {
        modeColVLAlign<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      }
    } else {
      modeRowRpt<UBop>(loc, b, dst, s0, s1, ptrTy, info);
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin1LNormModeSmall
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeSmall(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                 pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t rs = info.cols / static_cast<int64_t>(info.blockSizeElem);

    if (info.vRows > 1) {
      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                         idxc(info.vRows, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value off = b.create<arith::MulIOp>(loc, iv, idxc(info.cols, loc, b)).getResult();
      Value rd = addPtr(loc, b, dst, ptrTy, off);
      Value r0 = addPtr(loc, b, s0, ptrTy, off);
      Value r1 = addPtr(loc, b, s1, ptrTy, off);
      b.create<pto::UBSetMaskCountOp>(loc);
      b.create<pto::UBSetMaskOp>(loc, i64c(info.vCols, loc, b), i64c0(loc, b));
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c1(loc, b), i64c(rs, loc, b));
      b.create<pto::UBSetMaskNormOp>(loc);
      b.setInsertionPointAfter(forOp);
      fullMask(loc, b);
      return;
    }

    b.create<pto::UBSetMaskCountOp>(loc);
    b.create<pto::UBSetMaskOp>(loc, i64c(info.vCols, loc, b), i64c0(loc, b));
    emitUBBinOp<UBop>(loc, b, dst, s0, s1, i64c1(loc, b), i64c(rs, loc, b));
    b.create<pto::UBSetMaskNormOp>(loc);
    fullMask(loc, b);
  }

  //===--------------------------------------------------------------------===//
  // Bin1LCountMode
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeCount1L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    modeNorm1L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeColVLAlign
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeColVLAlign(Location loc, OpBuilder &b, Value dst, Value s0,
                      Value s1, pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t epr = info.elementsPerRepeat;
    int64_t headRepeats = info.vCols / epr;
    int64_t rowStride = info.cols;

    if (headRepeats > kRepeatMax) {
      modeCount2L<UBop>(loc, b, dst, s0, s1, ptrTy, info);
      return;
    }

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b))
                    .getResult();
    Value rd = addPtr(loc, b, dst, ptrTy, off);
    Value rs0 = addPtr(loc, b, s0, ptrTy, off);
    Value rs1 = addPtr(loc, b, s1, ptrTy, off);
    emitUBBinOp<UBop>(loc, b, rd, rs0, rs1, i64c(headRepeats, loc, b), i64c8(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LCountMode – row-by-row count mode
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeCount2L(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                   pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t rowStride = info.cols;

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value off = b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b))
                    .getResult();
    Value rd = addPtr(loc, b, dst, ptrTy, off);
    Value rs0 = addPtr(loc, b, s0, ptrTy, off);
    Value rs1 = addPtr(loc, b, s1, ptrTy, off);
    TileShapeInfo rowInfo = info;
    rowInfo.rows = 1;
    rowInfo.vRows = 1;
    modeNorm1L<UBop>(loc, b, rd, rs0, rs1, ptrTy, rowInfo);
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeRowRpt
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void modeRowRpt(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info) {
    int64_t be = info.blockSizeElem;
    int64_t rowStride = info.cols;
    int64_t rs = rowStride / be;
    bool condRowRpt = (info.vRows <= kRepeatMax) && (rs <= kRepeatStrideMax);

    if (condRowRpt) {
      rowRptFast<UBop>(loc, b, dst, s0, s1, ptrTy, info, rs);
    } else {
      rowRptChunked<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rs);
    }
  }

  template <typename UBop>
  void rowRptFast(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                  pto::PtrType ptrTy, const TileShapeInfo &info, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = info.vCols / epr;
    int64_t tailElements = info.vCols % epr;

    for (int64_t i = 0; i < numLoop; i++) {
      Value rd = addPtr(loc, b, dst, ptrTy, idxc(i * epr, loc, b));
      Value r0 = addPtr(loc, b, s0, ptrTy, idxc(i * epr, loc, b));
      Value r1 = addPtr(loc, b, s1, ptrTy, idxc(i * epr, loc, b));
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
    }

    if (tailElements > 0) {
      Value off = idxc(numLoop * epr, loc, b);
      Value rd = addPtr(loc, b, dst, ptrTy, off);
      Value r0 = addPtr(loc, b, s0, ptrTy, off);
      Value r1 = addPtr(loc, b, s1, ptrTy, off);
      setMask(loc, b, tailElements);
      emitUBBinOp<UBop>(loc, b, rd, r0, r1, i64c(info.vRows, loc, b), i64c(rs, loc, b));
      fullMask(loc, b);
    }
  }

  template <typename UBop>
  void rowRptChunked(Location loc, OpBuilder &b, Value dst, Value s0,
                     Value s1, pto::PtrType ptrTy, const TileShapeInfo &info,
                     int64_t rowStride, int64_t rs) {
    int64_t epr = info.elementsPerRepeat;
    int64_t rptPerLine = info.vCols / epr;
    int64_t remainElem = info.vCols % epr;

    if (info.vRows > static_cast<int64_t>(epr)) {
      if (rptPerLine > 0) {
        headRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rptPerLine);
      }
      if (remainElem > 0) {
        Value off = idxc(rptPerLine * epr, loc, b);
        tailRows<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
                 addPtr(loc, b, s0, ptrTy, off),
                 addPtr(loc, b, s1, ptrTy, off), ptrTy, info, rowStride, rs,
                 remainElem);
      }
    } else {
      if (remainElem == 0) {
        headRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride,
                 info.vCols / epr);
      } else if (rptPerLine > 0) {
        headRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rptPerLine);
        Value off = idxc(rptPerLine * epr, loc, b);
        tailRows<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
                 addPtr(loc, b, s0, ptrTy, off),
                 addPtr(loc, b, s1, ptrTy, off), ptrTy, info, rowStride, rs,
                 remainElem);
      } else {
        tailRows<UBop>(loc, b, dst, s0, s1, ptrTy, info, rowStride, rs, remainElem);
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeHead – chunked per-row head
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void headRows(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info,
                int64_t rowStride, int64_t rptPerLine) {
    int64_t epr = info.elementsPerRepeat;
    int64_t numLoop = rptPerLine / kRepeatMax;
    int64_t remain = rptPerLine % kRepeatMax;
    int64_t chunkElems = kRepeatMax * epr;

    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(info.vRows, loc, b),
                                      idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value iv = forOp.getInductionVar();
    Value rowBase =
        b.create<arith::MulIOp>(loc, iv, idxc(rowStride, loc, b)).getResult();

    if (numLoop > 0) {
      auto inner = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(numLoop, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(inner.getBody());
      Value jv = inner.getInductionVar();
      Value co = b.create<arith::MulIOp>(loc, jv, idxc(chunkElems, loc, b))
                     .getResult();
      Value off = b.create<arith::AddIOp>(loc, rowBase, co).getResult();
      emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
           addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
           i64c(kRepeatMax, loc, b), i64c8(loc, b));
      b.setInsertionPointAfter(inner);
    }

    if (remain > 0) {
      Value co = idxc(numLoop * chunkElems, loc, b);
      Value off = b.create<arith::AddIOp>(loc, rowBase, co).getResult();
      emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
           addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
           i64c(remain, loc, b), i64c8(loc, b));
    }
    b.setInsertionPointAfter(forOp);
  }

  //===--------------------------------------------------------------------===//
  // Bin2LNormModeTail – masked per-row tail
  //===--------------------------------------------------------------------===//

  template <typename UBop>
  void tailRows(Location loc, OpBuilder &b, Value dst, Value s0, Value s1,
                pto::PtrType ptrTy, const TileShapeInfo &info,
                int64_t rowStride, int64_t rs, unsigned remainPerLine) {
    bool strideOver =
        (rowStride / info.blockSizeElem > kRepeatStrideMax);
    setMask(loc, b, remainPerLine);

    int64_t numLoop = 0;
    int64_t remainAfterLoop = info.vRows;
    if (info.vRows > kRepeatMax) {
      numLoop = info.vRows / kRepeatMax;
      remainAfterLoop = info.vRows % kRepeatMax;

      auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                        idxc(numLoop, loc, b), idxc1(loc, b));
      b.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      if (strideOver) {
        tailStrideOverChunk<UBop>(loc, b, iv, dst, s0, s1, ptrTy, rowStride);
      } else {
        tailStrideOkChunk<UBop>(loc, b, iv, dst, s0, s1, ptrTy, rowStride, rs);
      }
      b.setInsertionPointAfter(forOp);
    }

    if (remainAfterLoop > 0) {
      if (strideOver) {
        tailStrideOverRemain<UBop>(loc, b, dst, s0, s1, ptrTy, rowStride, numLoop,
                             remainAfterLoop);
      } else {
        tailStrideOkRemain<UBop>(loc, b, dst, s0, s1, ptrTy, rowStride, rs, numLoop,
                           remainAfterLoop);
      }
    }

    fullMask(loc, b);
  }

  template <typename UBop>
  void tailStrideOverChunk(Location loc, OpBuilder &b, Value iv, Value dst,
                           Value s0, Value s1, pto::PtrType ptrTy,
                           int64_t rowStride) {
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b),
                                      idxc(kRepeatMax, loc, b), idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = b.create<arith::MulIOp>(
        loc, iv, idxc(kRepeatMax * rowStride, loc, b)).getResult();
    Value rowOff =
        b.create<arith::MulIOp>(loc, jv, idxc(rowStride, loc, b)).getResult();
    Value off = b.create<arith::AddIOp>(loc, baseOff, rowOff).getResult();
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c1(loc, b), i64c1(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  template <typename UBop>
  void tailStrideOkChunk(Location loc, OpBuilder &b, Value iv, Value dst,
                         Value s0, Value s1, pto::PtrType ptrTy,
                         int64_t rowStride, int64_t rs) {
    Value off = b.create<arith::MulIOp>(
        loc, iv, idxc(kRepeatMax * rowStride, loc, b)).getResult();
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c(kRepeatMax, loc, b), i64c(rs, loc, b));
  }

  template <typename UBop>
  void tailStrideOverRemain(Location loc, OpBuilder &b, Value dst, Value s0,
                            Value s1, pto::PtrType ptrTy, int64_t rowStride,
                            int64_t numLoop, int64_t remain) {
    auto forOp = b.create<scf::ForOp>(loc, idxc0(loc, b), idxc(remain, loc, b),
                                      idxc1(loc, b));
    b.setInsertionPointToStart(forOp.getBody());
    Value jv = forOp.getInductionVar();
    Value baseOff = idxc(numLoop * kRepeatMax * rowStride, loc, b);
    Value rowOff =
        b.create<arith::MulIOp>(loc, jv, idxc(rowStride, loc, b)).getResult();
    Value off = b.create<arith::AddIOp>(loc, baseOff, rowOff).getResult();
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c1(loc, b), i64c1(loc, b));
    b.setInsertionPointAfter(forOp);
  }

  template <typename UBop>
  void tailStrideOkRemain(Location loc, OpBuilder &b, Value dst, Value s0,
                          Value s1, pto::PtrType ptrTy, int64_t rowStride,
                          int64_t rs, int64_t numLoop, int64_t remain) {
    Value off = idxc(numLoop * kRepeatMax * rowStride, loc, b);
    emitUBBinOp<UBop>(loc, b, addPtr(loc, b, dst, ptrTy, off),
         addPtr(loc, b, s0, ptrTy, off), addPtr(loc, b, s1, ptrTy, off),
         i64c(remain, loc, b), i64c(rs, loc, b));
  }
};
} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createLowerPTOToUBufOpsPass() {
  return std::make_unique<LowerPTOToUBufOpsPass>();
}
} // namespace pto
} // namespace mlir
