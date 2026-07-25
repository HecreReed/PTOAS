// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOSOFTPOSTUPDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

// Per-op-type descriptor: how to extract address operands and check post-update.
// base/strideOperand indices are operand positions; updatedBaseResultIdx is the
// result index for updated_base (or -1 if no such result exists).
struct PostUpdateOpInfo {
  int baseOperandIdx;
  int strideOperandIdx;
  int64_t weight;              // 1 for standard, 32 for block-stride
  unsigned minResultsForPost;  // numResults > this means already post-update
};

using PostUpdateTable = llvm::StringMap<PostUpdateOpInfo>;

static const PostUpdateTable &getPostUpdateTable() {
  static const PostUpdateTable table = [] {
    PostUpdateTable t;
    //                       base  strideOp  weight  minResultsForPost
    t["pto.vlds"]         = { 0,    1,        1,      1 };
    t["pto.vsts"]         = { 1,    2,        1,      0 };
    t["pto.vsstb"]        = { 1,    3,        32,     0 };
    return t;
  }();
  return table;
}

static const PostUpdateOpInfo *getPostUpdateInfo(Operation *op) {
  auto it = getPostUpdateTable().find(op->getName().getStringRef());
  if (it == getPostUpdateTable().end())
    return nullptr;
  return &it->second;
}

// Extract base and stride operand from a candidate op using table info.
static void extractBaseAndStrideOperand(Operation *op,
                                        const PostUpdateOpInfo &info,
                                        Value &base, Value &strideOperand) {
  base = op->getOperand(info.baseOperandIdx);
  strideOperand = op->getOperand(info.strideOperandIdx);
}

// Check if op already has an updated_base result.
static bool isAlreadyPostUpdate(Operation *op, const PostUpdateOpInfo &info) {
  return op->getNumResults() > info.minResultsForPost;
}

// Check if op is directly inside the scf.for body (not nested in scf.if etc).
static bool isDirectlyInForBody(Operation *op, scf::ForOp forOp) {
  return op->getParentOp() == forOp.getOperation();
}

//===----------------------------------------------------------------------===//
// Delta Analysis
//===----------------------------------------------------------------------===//

// Compute the per-iteration delta of value `v` within `forOp`.
// Returns the delta as a loop-invariant Value, or nullptr if unknown.
static Value computeDelta(Value v, scf::ForOp forOp, OpBuilder &builder) {
  // IV: delta = step
  if (v == forOp.getInductionVar())
    return forOp.getStep();

  // Constant or loop-invariant: delta = 0
  if (forOp.isDefinedOutsideOfLoop(v)) {
    builder.setInsertionPoint(forOp);
    return builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
  }

  // Block argument from iter_args: check yield = arg + c
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value yieldVal = yieldOp.getOperand(idx);
      if (auto addOp = yieldVal.getDefiningOp<arith::AddIOp>()) {
        Value other;
        if (addOp.getLhs() == blockArg)
          other = addOp.getRhs();
        else if (addOp.getRhs() == blockArg)
          other = addOp.getLhs();
        if (other && forOp.isDefinedOutsideOfLoop(other))
          return other;
      }
      return nullptr;
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return nullptr;

  // arith.addi(a, b): delta = delta(a) + delta(b)
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    Value da = computeDelta(addOp.getLhs(), forOp, builder);
    Value db = computeDelta(addOp.getRhs(), forOp, builder);
    if (!da || !db)
      return nullptr;
    // Optimize: if either is constant 0, return the other
    if (auto ca = getConstantIntValue(da); ca && *ca == 0)
      return db;
    if (auto cb = getConstantIntValue(db); cb && *cb == 0)
      return da;
    builder.setInsertionPoint(forOp);
    return builder.create<arith::AddIOp>(forOp.getLoc(), da, db);
  }

  // arith.subi(a, b): delta = delta(a) - delta(b)
  if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    Value da = computeDelta(subOp.getLhs(), forOp, builder);
    Value db = computeDelta(subOp.getRhs(), forOp, builder);
    if (!da || !db)
      return nullptr;
    if (auto cb = getConstantIntValue(db); cb && *cb == 0)
      return da;
    builder.setInsertionPoint(forOp);
    return builder.create<arith::SubIOp>(forOp.getLoc(), da, db);
  }

  // arith.muli(a, b) where one is loop-invariant:
  //   delta = invariant * delta(other)
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();
    for (auto [invariant, variant] :
         {std::pair{rhs, lhs}, std::pair{lhs, rhs}}) {
      if (forOp.isDefinedOutsideOfLoop(invariant)) {
        Value dv = computeDelta(variant, forOp, builder);
        if (!dv)
          continue;
        if (auto cv = getConstantIntValue(dv); cv && *cv == 0) {
          builder.setInsertionPoint(forOp);
          return builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);
        }
        if (auto cv = getConstantIntValue(dv); cv && *cv == 1)
          return invariant;
        builder.setInsertionPoint(forOp);
        return builder.create<arith::MulIOp>(forOp.getLoc(), invariant, dv);
      }
    }
    return nullptr;
  }

  // arith.index_castui / arith.index_cast: delta = delta(input)
  if (auto castOp = dyn_cast<arith::IndexCastUIOp>(defOp))
    return computeDelta(castOp.getIn(), forOp, builder);
  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp))
    return computeDelta(castOp.getIn(), forOp, builder);

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Rewrite: create new ForOp with additional iter_arg
//===----------------------------------------------------------------------===//

// Compute the value of `v` at the first iteration (IV = lower bound) by
// cloning the def-chain with IV replaced by the lower bound.  Returns nullptr
// if `v` cannot be materialized outside the loop.
static Value materializeAtLoopEntry(Value v, scf::ForOp forOp,
                                    OpBuilder &builder) {
  // IV → lower bound
  if (v == forOp.getInductionVar())
    return forOp.getLowerBound();

  // Already defined outside the loop — use directly.
  if (forOp.isDefinedOutsideOfLoop(v))
    return v;

  // iter_arg → its init value
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (blockArg.getOwner() == forOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      unsigned idx = blockArg.getArgNumber() - 1;
      return forOp.getInitArgs()[idx];
    }
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp || !forOp->isAncestor(defOp))
    return nullptr;

  // Clone the defining op with operands materialized at loop entry.
  SmallVector<Value> newOperands;
  for (Value operand : defOp->getOperands()) {
    Value materialized = materializeAtLoopEntry(operand, forOp, builder);
    if (!materialized)
      return nullptr;
    newOperands.push_back(materialized);
  }
  builder.setInsertionPoint(forOp);
  Operation *cloned = builder.clone(*defOp);
  for (auto [i, operand] : llvm::enumerate(newOperands))
    cloned->setOperand(i, operand);
  return cloned->getResult(0);
}

// Compute the initial pointer: base + weight * strideOperand_at_iter0.
// weight=1 for vlds/vsts (offset in elements), weight=32 for vsstb/vsldb.
static Value computeInitialPtr(Value base, Value strideOperand,
                               int64_t weight, scf::ForOp forOp,
                               OpBuilder &builder) {
  if (!strideOperand)
    return base;

  Value soAtEntry = materializeAtLoopEntry(strideOperand, forOp, builder);
  if (!soAtEntry)
    return nullptr;

  if (auto constVal = getConstantIntValue(soAtEntry);
      constVal && *constVal == 0)
    return base;

  builder.setInsertionPoint(forOp);
  Value scaledOffset = soAtEntry;
  if (weight != 1) {
    Value soIndex = soAtEntry;
    if (soAtEntry.getType() != builder.getIndexType())
      soIndex = builder.create<arith::IndexCastUIOp>(
          forOp.getLoc(), builder.getIndexType(), soAtEntry);
    Value weightVal =
        builder.create<arith::ConstantIndexOp>(forOp.getLoc(), weight);
    scaledOffset =
        builder.create<arith::MulIOp>(forOp.getLoc(), soIndex, weightVal);
  }
  return builder.create<pto::AddPtrOp>(forOp.getLoc(), base, scaledOffset);
}

// Information about a post-update transformation to apply.
struct PostUpdateRewrite {
  Operation *op;
  Value base;
  Value stride;  // loop-invariant stride value (stride_new for block-stride ops)
  Value initPtr; // base + weight * strideOperand_at_iter0
};

// A unique key for grouping rewrites that can share an iter_arg.
// Same base + same stride (by Value identity) = same group.
using IterArgGroupKey = std::pair<Value, Value>;

static IterArgGroupKey getGroupKey(const PostUpdateRewrite &rw) {
  return {rw.base, rw.stride};
}

// Apply post-update rewrites to a single scf.for.
// Returns the new ForOp if any rewrites were applied, null otherwise.
static scf::ForOp applyPostUpdateRewrites(
    scf::ForOp forOp, ArrayRef<PostUpdateRewrite> rewrites,
    OpBuilder &builder) {
  if (rewrites.empty())
    return nullptr;

  // Group rewrites by (base, stride). Ops in the same group share one iter_arg
  // and all use the pre-update pointer. Only one updated_base per group is
  // yielded. This avoids redundant iter_args for same-address ops (e.g. vlds
  // + vsts both accessing %base[%iv]).
  DenseMap<IterArgGroupKey, unsigned> groupToIdx; // group key -> iter_arg index
  SmallVector<unsigned> rwGroupIdx(rewrites.size()); // rewrite -> group index
  SmallVector<Value> groupInitPtrs; // initial pointer per group (base + offset_at_iter0)

  for (auto [i, rw] : llvm::enumerate(rewrites)) {
    auto key = getGroupKey(rw);
    auto [it, inserted] = groupToIdx.try_emplace(key, groupInitPtrs.size());
    if (inserted)
      groupInitPtrs.push_back(rw.initPtr);
    rwGroupIdx[i] = it->second;
  }

  unsigned numGroups = groupInitPtrs.size();

  // Build new init args: original + one new pointer per group.
  SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                forOp.getInitArgs().end());
  for (Value ptr : groupInitPtrs)
    newInitArgs.push_back(ptr);

  unsigned origIterArgCount = forOp.getInitArgs().size();

  // Create new ForOp.
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());

  // Map old block args to new: IV + original iter_args.
  IRMapping mapping;
  Block *oldBody = forOp.getBody();
  Block *newBody = newForOp.getBody();
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < origIterArgCount; ++i)
    mapping.map(oldBody->getArgument(i + 1), newBody->getArgument(i + 1));

  // Clone the body, tracking old->new op correspondence.
  DenseMap<Operation *, Operation *> opMapping;
  builder.setInsertionPointToStart(newBody);
  for (auto &op : oldBody->without_terminator()) {
    Operation *cloned = builder.clone(op, mapping);
    opMapping[&op] = cloned;
  }

  // Apply rewrites. All ops in a group use the same pre-update pointer (block
  // arg). Track the last updated_base per group for yielding.
  SmallVector<Value> groupYieldPtrs(numGroups);
  for (unsigned g = 0; g < numGroups; ++g)
    groupYieldPtrs[g] = newBody->getArgument(origIterArgCount + 1 + g);

  for (auto [rwIdx, rw] : llvm::enumerate(rewrites)) {
    auto it = opMapping.find(rw.op);
    if (it == opMapping.end())
      continue;
    Operation *clonedOp = it->second;
    unsigned gIdx = rwGroupIdx[rwIdx];
    Value ptr = newBody->getArgument(origIterArgCount + 1 + gIdx);
    Value strideNew = mapping.lookupOrDefault(rw.stride);

    builder.setInsertionPoint(clonedOp);

    const PostUpdateOpInfo *info = getPostUpdateInfo(clonedOp);
    if (!info)
      continue;

    // Build the post-update op generically: replace base and strideOperand,
    // keep all other operands, append updated_base to result types.
    OperationState state(clonedOp->getLoc(), clonedOp->getName());
    for (auto [i, operand] : llvm::enumerate(clonedOp->getOperands())) {
      if (static_cast<int>(i) == info->baseOperandIdx)
        state.addOperands(ptr);
      else if (static_cast<int>(i) == info->strideOperandIdx)
        state.addOperands(strideNew);
      else
        state.addOperands(operand);
    }
    for (Type t : clonedOp->getResultTypes())
      state.addTypes(t);
    state.addTypes(ptr.getType()); // updated_base (appended last)
    state.addAttributes(clonedOp->getAttrs());

    Operation *newOp = builder.create(state);

    // Replace old results with new and update the mapping so that later
    // yield construction via mapping.lookupOrDefault sees the new results
    // instead of dangling pointers to the erased clonedOp.
    for (unsigned r = 0; r < clonedOp->getNumResults(); ++r) {
      clonedOp->getResult(r).replaceAllUsesWith(newOp->getResult(r));
      mapping.map(rw.op->getResult(r), newOp->getResult(r));
    }

    // updated_base is the last result.
    groupYieldPtrs[gIdx] = newOp->getResult(newOp->getNumResults() - 1);
    clonedOp->erase();
  }

  // Build yield: original yields + one pointer per group.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYields;
  for (Value v : oldYield.getOperands())
    newYields.push_back(mapping.lookupOrDefault(v));
  for (Value ptr : groupYieldPtrs)
    newYields.push_back(ptr);

  builder.setInsertionPointToEnd(newBody);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYields);

  // Replace original ForOp results (only the original ones).
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));

  forOp.erase();
  return newForOp;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct VPTOSoftPostUpdatePass
    : public pto::impl::VPTOSoftPostUpdateBase<VPTOSoftPostUpdatePass> {
  using pto::impl::VPTOSoftPostUpdateBase<
      VPTOSoftPostUpdatePass>::VPTOSoftPostUpdateBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    module.walk([&](pto::VecScopeOp vecscope) {
      processVecScope(vecscope, builder);
    });
  }

private:
  void processVecScope(pto::VecScopeOp vecscope, OpBuilder &builder) {
    // Collect scf.for ops inside this vecscope (inner-to-outer order).
    SmallVector<scf::ForOp> forOps;
    vecscope.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

    // Process inner-to-outer (walk gives us pre-order, reverse for post-order).
    std::reverse(forOps.begin(), forOps.end());

    for (scf::ForOp forOp : forOps)
      processForOp(forOp, builder);
  }

  void processForOp(scf::ForOp forOp, OpBuilder &builder) {
    SmallVector<PostUpdateRewrite> rewrites;

    for (Operation &op : *forOp.getBody()) {
      const PostUpdateOpInfo *info = getPostUpdateInfo(&op);
      if (!info)
        continue;
      if (isAlreadyPostUpdate(&op, *info))
        continue;
      if (!isDirectlyInForBody(&op, forOp))
        continue;

      Value base, strideOperand;
      extractBaseAndStrideOperand(&op, *info, base, strideOperand);

      // Delta analysis.
      Value deltaBase = computeDelta(base, forOp, builder);
      Value deltaOffset = computeDelta(strideOperand, forOp, builder);

      if (!deltaBase && !deltaOffset)
        continue;

      // Compute stride = delta(base) + weight * delta(strideOperand).
      // For vlds/vsts (weight=1): stride = delta(base) + delta(offset).
      // For vsstb (weight=32): stride = delta(dest) + 32*delta(rs).
      int64_t weight = info->weight;

      // Scale deltaOffset by weight if needed.
      Value weightedDeltaOffset = deltaOffset;
      if (deltaOffset && weight != 1) {
        if (auto co = getConstantIntValue(deltaOffset); co && *co != 0) {
          builder.setInsertionPoint(forOp);
          weightedDeltaOffset = builder.create<arith::ConstantIndexOp>(
              forOp.getLoc(), *co * weight);
        } else if (auto co = getConstantIntValue(deltaOffset); co && *co == 0) {
          // 0 * weight = 0, keep as is.
        } else {
          builder.setInsertionPoint(forOp);
          Value weightVal =
              builder.create<arith::ConstantIndexOp>(forOp.getLoc(), weight);
          weightedDeltaOffset =
              builder.create<arith::MulIOp>(forOp.getLoc(), deltaOffset, weightVal);
        }
      }

      Value stride;
      if (deltaBase && weightedDeltaOffset) {
        auto cb = getConstantIntValue(deltaBase);
        auto co = getConstantIntValue(weightedDeltaOffset);
        if (cb && *cb == 0)
          stride = weightedDeltaOffset;
        else if (co && *co == 0)
          stride = deltaBase;
        else {
          builder.setInsertionPoint(forOp);
          stride = builder.create<arith::AddIOp>(forOp.getLoc(),
                                                  deltaBase, weightedDeltaOffset);
        }
      } else if (deltaBase) {
        stride = deltaBase;
      } else {
        stride = weightedDeltaOffset;
      }

      if (!stride)
        continue;

      // Skip zero stride.
      if (auto constTotal = getConstantIntValue(stride);
          constTotal && *constTotal == 0)
        continue;

      // stride_new = stride / weight.
      // weight=1: stride_new = stride (always valid).
      // weight>1: stride must be a constant divisible by weight.
      Value strideNew;
      if (weight != 1) {
        auto constTotal = getConstantIntValue(stride);
        if (!constTotal || *constTotal % weight != 0)
          continue;
        builder.setInsertionPoint(forOp);
        unsigned bitWidth =
            strideOperand.getType().getIntOrFloatBitWidth();
        strideNew = builder.create<arith::ConstantIntOp>(
            forOp.getLoc(), *constTotal / weight, bitWidth);
      } else {
        strideNew = stride;
      }

      // Stride must be loop-invariant.
      if (!forOp.isDefinedOutsideOfLoop(strideNew))
        continue;

      Value initPtr =
          computeInitialPtr(base, strideOperand, weight, forOp, builder);
      if (!initPtr)
        continue;

      rewrites.push_back({&op, base, strideNew, initPtr});
    }

    if (!rewrites.empty())
      applyPostUpdateRewrites(forOp, rewrites, builder);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOSoftPostUpdatePass() {
  return std::make_unique<VPTOSoftPostUpdatePass>();
}
