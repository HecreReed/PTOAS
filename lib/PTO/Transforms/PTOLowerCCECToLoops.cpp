#include "PTO/IR/CCEC.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWERCCECTOLOOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static FailureOr<Value> buildVBinScalar(OpBuilder &builder, Location loc,
                                        StringRef kind, Value lhs, Value rhs) {
  if (kind == "add")
    return builder.create<arith::AddFOp>(loc, lhs, rhs).getResult();
  if (kind == "sub")
    return builder.create<arith::SubFOp>(loc, lhs, rhs).getResult();
  if (kind == "mul")
    return builder.create<arith::MulFOp>(loc, lhs, rhs).getResult();
  if (kind == "div")
    return builder.create<arith::DivFOp>(loc, lhs, rhs).getResult();
  if (kind == "max")
    return builder.create<arith::MaximumFOp>(loc, lhs, rhs).getResult();
  if (kind == "min")
    return builder.create<arith::MinimumFOp>(loc, lhs, rhs).getResult();
  return failure();
}

struct PTOLowerCCECToLoopsPass
    : public pto::impl::PTOLowerCCECToLoopsBase<PTOLowerCCECToLoopsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<ccec::VBinOp, 8> ops;
    module.walk([&](ccec::VBinOp op) { ops.push_back(op); });

    for (ccec::VBinOp op : ops) {
      auto src0Ty = dyn_cast<MemRefType>(op.getSrc0().getType());
      auto src1Ty = dyn_cast<MemRefType>(op.getSrc1().getType());
      auto dstTy = dyn_cast<MemRefType>(op.getDst().getType());
      if (!src0Ty || !src1Ty || !dstTy) {
        op.emitOpError("expects memref operands");
        signalPassFailure();
        return;
      }

      if (dstTy.getRank() != 2) {
        op.emitOpError("currently only supports rank-2 lowering");
        signalPassFailure();
        return;
      }

      Location loc = op.getLoc();
      OpBuilder builder(op);
      Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

      Value rows = dstTy.isDynamicDim(0)
                       ? builder.create<memref::DimOp>(loc, op.getDst(), c0).getResult()
                       : builder.create<arith::ConstantIndexOp>(loc, dstTy.getDimSize(0))
                             .getResult();
      Value cols = dstTy.isDynamicDim(1)
                       ? builder.create<memref::DimOp>(loc, op.getDst(), c1).getResult()
                       : builder.create<arith::ConstantIndexOp>(loc, dstTy.getDimSize(1))
                             .getResult();

      auto outer = builder.create<scf::ForOp>(loc, c0, rows, c1);
      OpBuilder outerBuilder = OpBuilder::atBlockBegin(outer.getBody());
      auto inner = outerBuilder.create<scf::ForOp>(loc, c0, cols, c1);
      OpBuilder innerBuilder = OpBuilder::atBlockBegin(inner.getBody());

      Value i = outer.getInductionVar();
      Value j = inner.getInductionVar();
      Value lhs =
          innerBuilder.create<memref::LoadOp>(loc, op.getSrc0(), ValueRange{i, j});
      Value rhs =
          innerBuilder.create<memref::LoadOp>(loc, op.getSrc1(), ValueRange{i, j});
      auto scalarOr = buildVBinScalar(innerBuilder, loc, op.getKind(), lhs, rhs);
      if (failed(scalarOr)) {
        op.emitOpError() << "unsupported ccec.vbin kind '" << op.getKind() << "'";
        signalPassFailure();
        return;
      }

      innerBuilder.create<memref::StoreOp>(loc, *scalarOr, op.getDst(),
                                           ValueRange{i, j});
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerCCECToLoopsPass() {
  return std::make_unique<PTOLowerCCECToLoopsPass>();
}
