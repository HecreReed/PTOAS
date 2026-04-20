// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static Attribute getResourcePipe(Operation *op) {
  MLIRContext *ctx = op->getContext();
  if (isa<pto::TLoadOp>(op))
    return pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE2);
  if (isa<pto::TStoreOp>(op))
    return pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE3);
  if (isa<pto::TAddOp>(op))
    return pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
  return {};
}

static bool isResourceOp(Operation *op, Attribute targetPipe) {
  Attribute pipe = getResourcePipe(op);
  return pipe && pipe == targetPipe;
}

static bool isPipeUsedInRegion(Region &region, Attribute targetPipe) {
  for (Block &block : region) {
    for (Operation &op : block) {
      if (isResourceOp(&op, targetPipe))
        return true;
      for (Region &nestedRegion : op.getRegions()) {
        if (isPipeUsedInRegion(nestedRegion, targetPipe))
          return true;
      }
    }
  }
  return false;
}

static bool hasPipelineActivityAfterOp(Operation *parentOp, Attribute targetPipe) {
  Block *parentBlock = parentOp ? parentOp->getBlock() : nullptr;
  if (!parentBlock)
    return false;
  for (auto it = std::next(parentOp->getIterator()); it != parentBlock->end();
       ++it) {
    if (isResourceOp(&*it, targetPipe))
      return true;
    if (it->getNumRegions() > 0)
      return true;
    if (isa<func::ReturnOp>(&*it))
      return false;
  }
  return false;
}

static bool isPipelineActiveFuture(Block *block, Block::iterator startIt,
                                   Attribute targetPipe) {
  for (auto it = startIt; it != block->end(); ++it) {
    Operation *op = &*it;
    if (isResourceOp(op, targetPipe))
      return true;
    for (Region &region : op->getRegions()) {
      if (isPipeUsedInRegion(region, targetPipe))
        return true;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      if (isa<func::ReturnOp>(op))
        return false;
      return hasPipelineActivityAfterOp(block->getParentOp(), targetPipe);
    }
  }
  return false;
}

static bool getSetSyncPipes(Operation *op, Attribute &src, Attribute &dst) {
  if (auto setOp = dyn_cast<pto::SetFlagOp>(op)) {
    src = setOp.getSrcPipe();
    dst = setOp.getDstPipe();
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  if (opName != "pto.set_flag_dyn" && opName != "pto.set_flag_d")
    return false;
  auto srcAttr = op->getAttrOfType<pto::PipeAttr>("src_pipe");
  auto dstAttr = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
  if (!srcAttr || !dstAttr)
    return false;
  src = srcAttr;
  dst = dstAttr;
  return true;
}

static bool getWaitSyncDst(Operation *op, Attribute &dst) {
  if (auto waitOp = dyn_cast<pto::WaitFlagOp>(op)) {
    dst = waitOp.getDstPipe();
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  if (opName != "pto.wait_flag_dyn" && opName != "pto.wait_flag_d")
    return false;
  auto dstAttr = op->getAttrOfType<pto::PipeAttr>("dst_pipe");
  if (!dstAttr)
    return false;
  dst = dstAttr;
  return true;
}

static bool shouldEraseBarrier(Block *block, Block::iterator it,
                               const llvm::DenseSet<Attribute> &dirtyPipes) {
  auto barrierOp = cast<pto::BarrierOp>(&*it);
  Attribute barrierPipe = barrierOp.getPipe();
  if (!isPipelineActiveFuture(block, std::next(it), barrierPipe))
    return true;
  if (!dirtyPipes.count(barrierPipe))
    return true;

  auto nextIt = std::next(it);
  if (nextIt == block->end())
    return false;

  Attribute nextSrc;
  Attribute nextDst;
  return getSetSyncPipes(&*nextIt, nextSrc, nextDst) && nextSrc == barrierPipe;
}

static bool shouldEraseWait(Block *block, Block::iterator it) {
  Attribute waitDst;
  return getWaitSyncDst(&*it, waitDst) &&
         !isPipelineActiveFuture(block, std::next(it), waitDst);
}

static bool shouldEraseSet(Block *block, Block::iterator it,
                           const llvm::DenseSet<Attribute> &dirtyPipes) {
  Attribute setSrc;
  Attribute setDst;
  if (!getSetSyncPipes(&*it, setSrc, setDst))
    return false;
  if (!isPipelineActiveFuture(block, std::next(it), setDst))
    return true;
  return !dirtyPipes.count(setSrc);
}

static void collectRedundantSyncOps(func::FuncOp func,
                                    SmallVectorImpl<Operation *> &opsToErase) {
  func.walk([&](Block *block) {
    llvm::DenseSet<Attribute> dirtyPipes;
    for (auto it = block->begin(); it != block->end(); ++it) {
      Operation *op = &*it;
      if (Attribute pipe = getResourcePipe(op)) {
        dirtyPipes.insert(pipe);
        continue;
      }

      if (isa<pto::BarrierOp>(op)) {
        if (shouldEraseBarrier(block, it, dirtyPipes)) {
          opsToErase.push_back(op);
          continue;
        }
        dirtyPipes.erase(cast<pto::BarrierOp>(op).getPipe());
        continue;
      }

      if (shouldEraseWait(block, it) || shouldEraseSet(block, it, dirtyPipes))
        opsToErase.push_back(op);
    }
  });
}

struct PTORemoveRedundantBarrierPass
    : public PassWrapper<PTORemoveRedundantBarrierPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORemoveRedundantBarrierPass)

  void runOnOperation() override {
    SmallVector<Operation *> opsToErase;
    collectRedundantSyncOps(getOperation(), opsToErase);
    for (Operation *op : opsToErase) {
      if (op && op->getBlock())
        op->erase();
    }
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass() {
  return std::make_unique<PTORemoveRedundantBarrierPass>();
}
} // namespace pto
} // namespace mlir
