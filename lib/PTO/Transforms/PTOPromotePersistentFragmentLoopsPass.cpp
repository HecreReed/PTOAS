// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPromotePersistentFragmentLoopsPass.cpp --------------------------===//
//
// Promote loops that access persistent SIMT fragment buffers to forced full
// unrolling.
//
// Persistent fragment materialization (pto-materialize-simt-persistent-
// -fragment) requires the loops touching a persistent buffer to be fully
// unrolled, so that every access resolves to a stable resident slot.  This
// pass is the discovery half of that pipeline: it finds the loops and marks
// them for pto-unroll-loops; it does NOT do slot assignment, keep/resume
// generation, SIMT outlining, or LLVM metadata lowering.
//
// Discovery: every llvm.alloca carrying {pto.persistent} is an explicit
// entry point (no structural re-inference of persistence).  The pass walks
// the alloca's use graph (getelementptr -> load/store, plus any other
// users) and collects every enclosing scf.for of every related op:
//
//   - loops directly wrapping an access inside a SIMT section;
//   - kernel-level loops wrapping whole pto.section.simt regions;
//   - every layer of multiply nested loops.
//
// Promotion rules for each collected loop:
//
//   | original state          | result                                |
//   |-------------------------|---------------------------------------|
//   | no unroll attribute     | pto.unroll = "full"                   |
//   | pto.unroll = "enable"   | replaced with pto.unroll = "full"     |
//   | pto.unroll = "full"     | unchanged                             |
//   | pto.unroll_factor = N   | hard error                            |
//
// Every promoted loop additionally gets the internal marker
// {pto.persistent_unroll}: pto-unroll-loops turns its usual
// drop-the-hint-with-a-remark fallback into a hard error for marked loops,
// because silently keeping a persistent loop as an ordinary loop would
// break the materialization precondition downstream.  The marker
// disappears together with the loop when it is unrolled.
//
// Fail-fast cases (hard errors, never a silent fallback):
//   - a fixed unroll factor on a persistent loop;
//   - a persistent access nested under scf.while (unsupported control flow);
//   - a statically known trip count above max-persistent-unroll-trip-count.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPROMOTEPERSISTENTFRAGMENTLOOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-promote-persistent-loops"

namespace {

/// Compute the constant trip count of *forOp*, or std::nullopt when any of
/// the bounds/step is not a compile-time constant.
static std::optional<int64_t> getStaticTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb) {
    return std::nullopt;
  }
  int64_t tripCount = (*ub - *lb + *step - 1) / *step;
  if (tripCount <= 0) {
    return std::nullopt;
  }
  return tripCount;
}

struct PTOPromotePersistentFragmentLoops
    : public pto::impl::PTOPromotePersistentFragmentLoopsBase<
          PTOPromotePersistentFragmentLoops> {
  using pto::impl::PTOPromotePersistentFragmentLoopsBase<
      PTOPromotePersistentFragmentLoops>::PTOPromotePersistentFragmentLoopsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: collect every op related to a persistent fragment buffer by
    // walking each persistent alloca's use graph.  The alloca itself is
    // included so kernel-level loops wrapping both the allocation and the
    // sections are discovered too.
    llvm::SmallPtrSet<Operation *, 32> relatedOps;
    SmallVector<LLVM::AllocaOp, 4> persistentAllocas;
    bool foundPersistent = false;
    func.walk([&](LLVM::AllocaOp allocaOp) {
      bool isPersistentAlloca = allocaOp->hasAttr(pto::kPersistentAttrName);
      if (!isPersistentAlloca) {
        return;
      }
      foundPersistent = true;
      persistentAllocas.push_back(allocaOp);
      SmallVector<Operation *> worklist{allocaOp.getOperation()};
      while (!worklist.empty()) {
        Operation *cur = worklist.pop_back_val();
        bool newlySeen = relatedOps.insert(cur).second;
        if (!newlySeen) {
          continue;
        }
        for (Operation *user : cur->getUsers()) {
          worklist.push_back(user);
        }
      }
    });
    if (!foundPersistent) {
      return;
    }

    // Step 2: collect every enclosing scf.for of every related op (all
    // nesting layers, including kernel-level loops wrapping
    // pto.section.simt).  A persistent access nested under scf.while is
    // unsupported control flow: the loop structure cannot carry a full-unroll
    // hint, and silently skipping it would break materialization downstream.
    // Diagnostics are collected across the whole function before the pass
    // fails once: the function pass adaptor may stop scheduling functions
    // after the first failure, so per-function completeness keeps the
    // emitted set deterministic under parallel scheduling.
    bool failed = false;
    // Discovery order (function order) is kept so that the emitted
    // diagnostics have a stable order.
    llvm::SmallPtrSet<Operation *, 8> seenLoopOps;
    SmallVector<Operation *, 8> loopOps;
    for (Operation *op : relatedOps) {
      if (auto whileOp = op->getParentOfType<scf::WhileOp>()) {
        whileOp.emitError()
            << "persistent fragment buffer is accessed inside scf.while, "
               "which cannot be fully unrolled; persistent fragment loops "
               "must use scf.for";
        failed = true;
      }
      Operation *cur = op;
      while (auto forOp = cur->getParentOfType<scf::ForOp>()) {
        if (seenLoopOps.insert(forOp.getOperation()).second) {
          loopOps.push_back(forOp.getOperation());
        }
        cur = forOp.getOperation();
      }
    }

    // Step 3: promote every collected loop to forced full unrolling.
    for (Operation *loopOp : loopOps) {
      auto forOp = cast<scf::ForOp>(loopOp);
      if (forOp->hasAttr(pto::kUnrollFactorAttrName)) {
        forOp.emitError()
            << "persistent fragment loop requires full unroll; a fixed '"
            << pto::kUnrollFactorAttrName << "' is not supported";
        failed = true;
        continue;
      }

      // Guardrail: bound the code expansion of the forced unroll.  A dynamic
      // trip count cannot be checked here; pto-unroll-loops reports a
      // persistent-specific error for it via the marker below.
      int64_t tripCountCap = maxPersistentUnrollTripCount.getValue();
      if (tripCountCap >= 0) {
        if (std::optional<int64_t> tripCount = getStaticTripCount(forOp);
            tripCount && *tripCount > tripCountCap) {
          auto diag = forOp.emitError()
                      << "persistent fragment loop in '" << func.getSymName()
                      << "' has trip count " << *tripCount
                      << ", which exceeds max-persistent-unroll-trip-count="
                      << tripCountCap;
          for (LLVM::AllocaOp allocaOp : persistentAllocas) {
            diag.attachNote(allocaOp.getLoc())
                << "persistent fragment allocation";
          }
          failed = true;
          continue;
        }
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "PTOPromotePersistentLoops: promoting scf.for at "
                 << forOp.getLoc() << " to full unroll\n");

      // "full" is idempotent; "enable" and no-hint are both overridden.
      forOp->setAttr(pto::kUnrollAttrName,
                     StringAttr::get(ctx, pto::kUnrollFullValue));
      // The marker makes pto-unroll-loops fail (instead of dropping the
      // hint) when this loop cannot be unrolled natively.  It is added on
      // already-"full" loops too so their failure path is likewise a hard
      // error, per the fail-fast contract.
      forOp->setAttr(pto::kPersistentUnrollMarkerAttrName,
                     UnitAttr::get(ctx));
    }
    if (failed) {
      signalPassFailure();
    }
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOPromotePersistentFragmentLoopsPass() {
  return std::make_unique<PTOPromotePersistentFragmentLoops>();
}
