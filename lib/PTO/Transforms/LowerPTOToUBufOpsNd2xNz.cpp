// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/LowerPTOToUBufOpsNd2xNz.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/SmallSet.h"

#include <limits>

using namespace mlir;

namespace mlir::pto::detail {

static bool snapshotNd2xNzIndices(pto::TExtractOp op,
                                   PendingTExtractNd2xNz &pending) {
  auto indices = op.getIndices();
  bool hasFourIndices = indices.size() == 4;
  if (!hasFourIndices) {
    return false;
  }
  for (Value index : indices) {
    auto constant = mlir::pto::getPTOConstantIntLike(index);
    if (!constant) {
      return false;
    }
    pending.indices.push_back(*constant);
  }
  return true;
}

static bool snapshotNd2xNzSource(pto::TExtractOp op,
                                  PendingTExtractNd2xNz &pending) {
  auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  if (!srcTy) {
    return false;
  }
  pending.elemBytes =
      mlir::pto::getPTOStorageElemByteSize(srcTy.getElementType());
  if (pending.elemBytes == 0 || 32 % pending.elemBytes != 0) {
    return false;
  }
  auto shape = srcTy.getShape();
  const bool invalidSourceRank = shape.size() != 2;
  if (invalidSourceRank) {
    return false;
  }
  pending.srcPhysicalRows = shape[0];
  pending.srcPhysicalCols = shape[1];
  pending.srcRowStrideElems = shape[1];
  return true;
}

static bool snapshotNd2xNzDestinations(pto::TExtractOp op,
                                        PendingTExtractNd2xNz &pending) {
  for (Value dst : op.getDsts()) {
    auto dstTy = dyn_cast<pto::TileBufType>(dst.getType());
    if (!dstTy) {
      return false;
    }
    auto shape = dstTy.getShape();
    auto valid = dstTy.getValidShape();
    bool invalidShape = shape.size() != 2 || valid.size() != 2;
    if (invalidShape) {
      return false;
    }
    pending.dstPhysical.push_back(shape[0]);
    pending.dstPhysical.push_back(shape[1]);
    pending.dstValid.push_back(valid[0]);
    pending.dstValid.push_back(valid[1]);
  }
  return pending.dstPhysical.size() == 4 && pending.dstValid.size() == 4;
}

std::optional<PendingTExtractNd2xNz>
snapshotTExtractNd2xNz(pto::TExtractOp op) {
  if (!op.isNdTo2xNzForm()) {
    return std::nullopt;
  }
  PendingTExtractNd2xNz pending;
  pending.op = op;
  bool snapshotFailed = !snapshotNd2xNzIndices(op, pending) ||
                        !snapshotNd2xNzSource(op, pending) ||
                        !snapshotNd2xNzDestinations(op, pending);
  if (snapshotFailed) {
    return std::nullopt;
  }
  return pending;
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
  // Static loop-bound/offset products must be checked (overflow would wrap
  // the footprint and corrupt loads/stores).
  int64_t total = 0;
  bool totalOverflow = __builtin_mul_overflow(validRows, validCols, &total);
  if (totalOverflow || total <= 0) {
    return failure();
  }
  int64_t physRowsTimesC0 = 0;
  bool footprintOverflow =
      __builtin_mul_overflow(physRows, c0, &physRowsTimesC0);
  if (footprintOverflow || physRowsTimesC0 <= 0) {
    return failure();
  }
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
        loc, cDiv, cst(builder, loc, physRowsTimesC0));
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
// id not already used by explicit set_flag/wait_flag ops in the whole
// enclosing function (not just the immediate region, so a nested scf.for
// sees every explicit event in the function). Only EVENT_ID0..7 are legal
// compiler event ids (SyncEventIdAllocation::kTotalEventIdNum == 8; 14/15
// are hardware block-sync ids); if all eight are in use this expansion must
// fail rather than silently reuse a reserved/literal event.
std::optional<unsigned> selectUnusedHiddenEventId(Operation *anchor,
                                                         MLIRContext *ctx) {
  llvm::SmallDenseSet<int32_t, 8> used;
  // Dynamic set/wait flags take a runtime event id; a constant dynamic id
  // folds into `used`, while a genuinely runtime id could take any legal
  // value, so we can no longer prove a static id is safe.
  bool hasUnknownDynEvent = false;
  auto scan = [&](Operation *scope) {
    scope->walk([&](Operation *inner) {
      if (auto sf = dyn_cast<pto::SetFlagOp>(inner)) {
        if (auto ev = sf.getEventIdAttr()) {
          used.insert(static_cast<int32_t>(ev.getEvent()));
        }
        return;
      }
      if (auto wf = dyn_cast<pto::WaitFlagOp>(inner)) {
        if (auto ev = wf.getEventIdAttr()) {
          used.insert(static_cast<int32_t>(ev.getEvent()));
        }
        return;
      }
      if (auto sd = dyn_cast<pto::SetFlagDynOp>(inner)) {
        if (auto c = mlir::pto::getPTOConstantIntLike(sd.getEventId())) {
          used.insert(static_cast<int32_t>(*c));
        } else {
          hasUnknownDynEvent = true;
        }
        return;
      }
      if (auto wd = dyn_cast<pto::WaitFlagDynOp>(inner)) {
        if (auto c = mlir::pto::getPTOConstantIntLike(wd.getEventId())) {
          used.insert(static_cast<int32_t>(*c));
        } else {
          hasUnknownDynEvent = true;
        }
      }
    });
  };
  func::FuncOp func = anchor->getParentOfType<func::FuncOp>();
  if (func) {
    scan(func);
  } else {
    scan(anchor->getParentOp() ? anchor->getParentOp() : anchor);
  }
  // kTotalEventIdNum == 8 bounds the legal compiler event-id space
  // (SyncEventIdAllocation.h); a straightforward linear scan keeps
  // determinism. With an unresolvable dynamic event id in the function we
  // cannot prove any static id is disjoint (a runtime id may take 0..7), so
  // expansion refuses to guess (design doc 6.3.1: do not emit unregistered
  // literal events).
  constexpr unsigned kCompilerEventIdNum = 8;
  if (hasUnknownDynEvent) {
    return std::nullopt;
  }
  for (unsigned id = 0; id < kCompilerEventIdNum; ++id) {
    if (!used.count(static_cast<int32_t>(id))) {
      return id;
    }
  }
  return std::nullopt;
}

LogicalResult expandTExtractNd2xNz(
    OpBuilder &builder, MLIRContext *ctx, const PendingTExtractNd2xNz &pd,
    pto::EVENT eventId) {
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
  // One event id is chosen per function (see the caller): sequential barrier
  // pairs on a shared id are safe because every set is matched by wait before
  // the next op's set, and the id avoids ids used by explicit user flags.
  auto event0 = pto::EventAttr::get(ctx, eventId);
  builder.create<pto::SetFlagOp>(loc, pipeV, pipeS, event0);
  builder.create<pto::WaitFlagOp>(loc, pipeV, pipeS, event0);

  LogicalResult firstWindow =
      expandTExtractNd2xNzWindow(builder, loc, pd, 0, srcPtr, dst0Ptr);
  LogicalResult secondWindow =
      expandTExtractNd2xNzWindow(builder, loc, pd, 1, srcPtr, dst1Ptr);
  const bool windowExpansionFailed = failed(firstWindow) || failed(secondWindow);
  if (windowExpansionFailed) {
    return failure();
  }

  builder.setInsertionPoint(rawOp);
  builder.create<pto::SetFlagOp>(loc, pipeS, pipeV, event0);
  builder.create<pto::WaitFlagOp>(loc, pipeS, pipeV, event0);

  rawOp->erase();
  return success();
}

} // namespace mlir::pto::detail
