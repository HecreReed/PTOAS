#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include <functional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCREATEFUSIONGROUPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static bool isFusibleBinaryOp(Operation *op) {
  return isa<pto::TMulOp, pto::TDivOp, pto::TAddOp, pto::TSubOp,
             pto::TMaxOp, pto::TMinOp>(op);
}

static Value getBinaryDst(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return mul.getDst();
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return div.getDst();
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return add.getDst();
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return sub.getDst();
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return max.getDst();
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return min.getDst();
  return {};
}

static SmallVector<Value, 2> getBinarySrcs(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return {mul.getSrc0(), mul.getSrc1()};
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return {div.getSrc0(), div.getSrc1()};
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return {add.getSrc0(), add.getSrc1()};
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return {sub.getSrc0(), sub.getSrc1()};
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return {max.getSrc0(), max.getSrc1()};
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return {min.getSrc0(), min.getSrc1()};
  return {};
}

struct PTOCreateFusionGroupsPass
    : public pto::impl::PTOCreateFusionGroupsBase<PTOCreateFusionGroupsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    int64_t nextGroupId = 0;

    std::function<void(Region &)> processRegion = [&](Region &region) {
      for (Block &block : region.getBlocks()) {
        SmallVector<Operation *, 8> chain;

        auto flushChain = [&]() {
          if (chain.size() < 2) {
            chain.clear();
            return;
          }
          int64_t gid = nextGroupId++;
          for (auto it : llvm::enumerate(chain)) {
            int64_t idx = static_cast<int64_t>(it.index());
            Operation *op = it.value();
            op->setAttr("pto.fusion.group_id",
                        IntegerAttr::get(IntegerType::get(ctx, 64), gid));
            op->setAttr("pto.fusion.order",
                        IntegerAttr::get(IntegerType::get(ctx, 64), idx));
          }
          chain.clear();
        };

        for (Operation &op : block.getOperations()) {
          Operation *cur = &op;

          if (!isFusibleBinaryOp(cur)) {
            flushChain();
            for (Region &nested : cur->getRegions())
              processRegion(nested);
            continue;
          }

          if (chain.empty()) {
            chain.push_back(cur);
            continue;
          }

          Value prevDst = getBinaryDst(chain.back());
          SmallVector<Value, 2> curSrcs = getBinarySrcs(cur);
          bool dependsOnPrev = llvm::is_contained(curSrcs, prevDst);
          if (!dependsOnPrev) {
            flushChain();
            chain.push_back(cur);
            continue;
          }
          chain.push_back(cur);
        }
        flushChain();
      }
    };

    processRegion(func.getRegion());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCreateFusionGroupsPass() {
  return std::make_unique<PTOCreateFusionGroupsPass>();
}
