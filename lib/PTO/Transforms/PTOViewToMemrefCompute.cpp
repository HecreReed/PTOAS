// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOViewToMemrefCompute.cpp ----------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOViewToMemrefInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::pto {

namespace {

template <typename T>
using DefaultInlineVector = SmallVector<T, 8>;

constexpr unsigned kThirdOperandIndex = 2;
constexpr unsigned kFourthOperandIndex = 3;
constexpr unsigned kFifthOperandIndex = 4;
constexpr unsigned kSixthOperandIndex = 5;

} // namespace

LogicalResult lowerViewToMemrefComputeOps(func::FuncOp func, MLIRContext *ctx) {
// ------------------------------------------------------------------
// Stage 3: Rewrite Compute Ops
// [关键] 全面使用 op->getOperand(i) 避免 Typed Accessor Crash
// ------------------------------------------------------------------

// --- TLoadOp [Src, Dst] ---
DefaultInlineVector<mlir::pto::TLoadOp> loads;
func.walk([&](mlir::pto::TLoadOp op) { loads.push_back(op); });
for (auto op : loads) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    
    Value src = op->getOperand(0); 
    Value dst = op->getOperand(1);

    auto newOp =
        rewriter.create<pto::TLoadOp>(op.getLoc(), TypeRange{}, src, dst);
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, newOp->getResults());
}

// --- TStoreOp [Src, Dst] ---
DefaultInlineVector<mlir::pto::TStoreOp> storeops;
func.walk([&](mlir::pto::TStoreOp op) { storeops.push_back(op); });
for (auto op : storeops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op->getOperand(0); 
  Value dst = op->getOperand(1);
  Value preQuant = op.getPreQuantScalar();

  pto::TStoreOp newOp;
  if (preQuant) {
    newOp = rewriter.create<pto::TStoreOp>(op.getLoc(), TypeRange{},
                                           src, dst, preQuant);
  } else {
    newOp = rewriter.create<pto::TStoreOp>(op.getLoc(), TypeRange{},
                                           src, dst, Value{});
  }
  newOp->setAttrs(op->getAttrs());
  rewriter.replaceOp(op, newOp->getResults());
}

 // --- TTransOp [Src, Tmp, Dst] ---
DefaultInlineVector<mlir::pto::TTransOp> trans;
func.walk([&](mlir::pto::TTransOp op) { trans.push_back(op); });
for (auto op : trans) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TTransOp>(
      op, TypeRange{}, op->getOperand(0), op->getOperand(1),
      op->getOperand(kThirdOperandIndex));
}

// --- TExpOp [Src, Dst] ---
DefaultInlineVector<mlir::pto::TExpOp> exp;
func.walk([&](mlir::pto::TExpOp op) { exp.push_back(op); });
for (auto op : exp) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TExpOp>(
      op, TypeRange{}, op->getOperand(0), op->getOperand(1));
}

// --- TMulOp [Src, Scalar, Dst] ---
DefaultInlineVector<mlir::pto::TMulOp> mul;
func.walk([&](mlir::pto::TMulOp op) { mul.push_back(op); });
for (auto op : mul) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMulOp>(
      op, op->getOperand(0), op.getOperand(1),
      op->getOperand(kThirdOperandIndex));
}

// --- TMulSOp [Src, Scalar, Dst] ---
DefaultInlineVector<mlir::pto::TMulSOp> muls;
func.walk([&](mlir::pto::TMulSOp op) { muls.push_back(op); });
for (auto op : muls) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMulSOp>(
      op, op->getOperand(0), op.getScalar(),
      op->getOperand(kThirdOperandIndex));
}

// --- TAddOp [Src0, Src1, Dst] ---
DefaultInlineVector<mlir::pto::TAddOp> addops;
func.walk([&](mlir::pto::TAddOp op) { addops.push_back(op); });
for (auto op : addops) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    
    rewriter.replaceOpWithNewOp<pto::TAddOp>(
        op, TypeRange{}, 
        op->getOperand(0), op->getOperand(1),
        op->getOperand(kThirdOperandIndex));
}

// --- TMatmulOp [Lhs, Rhs, Dst] (no optional bias in ODS) ---
DefaultInlineVector<mlir::pto::TMatmulOp > matmuls;
func.walk([&](mlir::pto::TMatmulOp  op) { matmuls.push_back(op); });
for (auto op : matmuls) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Value dst = op->getOperand(kThirdOperandIndex);

  rewriter.replaceOpWithNewOp<pto::TMatmulOp>(
      op, TypeRange{}, lhs, rhs, dst, op.getAccPhaseAttr());
}

// --- TMatmulAccOp [Acc, Lhs, Rhs, Dst] ---
DefaultInlineVector<mlir::pto::TMatmulAccOp > matmulAccs;
func.walk([&](mlir::pto::TMatmulAccOp  op) { matmulAccs.push_back(op); });
for (auto op : matmulAccs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMatmulAccOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex), op.getAccPhaseAttr());
}

// --- TMatmulBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
DefaultInlineVector<mlir::pto::TMatmulBiasOp > matmulBiass;
func.walk([&](mlir::pto::TMatmulBiasOp  op) { matmulBiass.push_back(op); });
for (auto op : matmulBiass) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMatmulBiasOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex));
}

// --- TMatmulMxOp---
DefaultInlineVector<mlir::pto::TMatmulMxOp > matmulMxs;
func.walk([&](mlir::pto::TMatmulMxOp  op) { matmulMxs.push_back(op); });
for (auto op : matmulMxs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMatmulMxOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex),
    op->getOperand(kFifthOperandIndex));
}

// --- TMatmulMxAccOp  ---
DefaultInlineVector<mlir::pto::TMatmulMxAccOp > matmulMxAccs;
func.walk([&](mlir::pto::TMatmulMxAccOp  op) { matmulMxAccs.push_back(op); });
for (auto op : matmulMxAccs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMatmulMxAccOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex),
    op->getOperand(kFifthOperandIndex),
    op->getOperand(kSixthOperandIndex));
}

// --- TMatmulMxBiasOp ---
DefaultInlineVector<mlir::pto::TMatmulMxBiasOp > matmulMxBiass;
func.walk([&](mlir::pto::TMatmulMxBiasOp  op) { matmulMxBiass.push_back(op); });
for (auto op : matmulMxBiass) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMatmulMxBiasOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex),
    op->getOperand(kFifthOperandIndex),
    op->getOperand(kSixthOperandIndex));
}

// --- TGemvOp [Lhs, Rhs, Dst] ---
DefaultInlineVector<mlir::pto::TGemvOp > gemvs;
func.walk([&](mlir::pto::TGemvOp  op) { gemvs.push_back(op); });
for (auto op : gemvs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Value dst = op->getOperand(kThirdOperandIndex);

  rewriter.replaceOpWithNewOp<pto::TGemvOp>(
    op, TypeRange{}, lhs, rhs, dst);
}

// --- TGemvAccOp [Acc, Lhs, Rhs, Dst] ---
DefaultInlineVector<mlir::pto::TGemvAccOp > gemvAccs;
func.walk([&](mlir::pto::TGemvAccOp  op) { gemvAccs.push_back(op); });
for (auto op : gemvAccs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TGemvAccOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex));
}

// --- TGemvBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
DefaultInlineVector<mlir::pto::TGemvBiasOp > gemvBiass;
func.walk([&](mlir::pto::TGemvBiasOp  op) { gemvBiass.push_back(op); });
for (auto op : gemvBiass) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TGemvBiasOp>(
    op, TypeRange{}, 
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex));
}

// --- TGemvMxOp [A, AScale, B, BScale, Dst] ---
DefaultInlineVector<mlir::pto::TGemvMxOp > gemvMxs;
func.walk([&](mlir::pto::TGemvMxOp  op) { gemvMxs.push_back(op); });
for (auto op : gemvMxs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TGemvMxOp>(
    op, TypeRange{},
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex),
    op->getOperand(kFifthOperandIndex));
}

// --- TGemvMxAccOp [CIn, A, AScale, B, BScale, Dst] ---
DefaultInlineVector<mlir::pto::TGemvMxAccOp > gemvMxAccs;
func.walk([&](mlir::pto::TGemvMxAccOp  op) { gemvMxAccs.push_back(op); });
for (auto op : gemvMxAccs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TGemvMxAccOp>(
    op, TypeRange{},
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex),
    op->getOperand(kFifthOperandIndex),
    op->getOperand(kSixthOperandIndex));
}

// --- TGemvMxBiasOp [A, AScale, B, BScale, Bias, Dst] ---
DefaultInlineVector<mlir::pto::TGemvMxBiasOp > gemvMxBiass;
func.walk([&](mlir::pto::TGemvMxBiasOp  op) { gemvMxBiass.push_back(op); });
for (auto op : gemvMxBiass) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TGemvMxBiasOp>(
    op, TypeRange{},
    op->getOperand(0), op->getOperand(1),
    op->getOperand(kThirdOperandIndex),
    op->getOperand(kFourthOperandIndex),
    op->getOperand(kFifthOperandIndex),
    op->getOperand(kSixthOperandIndex));
}

// --- TMovOp [Src, Dst] ---
DefaultInlineVector<mlir::pto::TMovOp > movs;
func.walk([&](mlir::pto::TMovOp  op) { movs.push_back(op); });
for (auto op : movs) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<pto::TMovOp>(
      op, TypeRange{}, op.getSrc(), op.getDst(), op.getFp(),
      op.getPreQuantScalar(), op.getAccToVecModeAttr(),
      op.getReluPreModeAttr());
}

DefaultInlineVector<mlir::pto::TAbsOp> abseops;
func.walk([&](mlir::pto::TAbsOp op) { abseops.push_back(op); });

for (auto op : abseops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TAbsOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TAddCOp> addcops;
func.walk([&](mlir::pto::TAddCOp op) { addcops.push_back(op); });

for (auto op : addcops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value src2 = op.getSrc2();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto src2Ty = dyn_cast<MemRefType>(src2.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !src2Ty ||!dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TAddCOp>(
      op,
      TypeRange{},
      src0,
      src1,
      src2,
      dst);
}

DefaultInlineVector<mlir::pto::TAddSOp> addsops;
func.walk([&](mlir::pto::TAddSOp op) { addsops.push_back(op); });

for (auto op : addsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TAddSOp>(
      op,
      TypeRange{},
      src,
      scalar,
      dst);
}

DefaultInlineVector<mlir::pto::TAddSCOp> addscops;
func.walk([&](mlir::pto::TAddSCOp op) { addscops.push_back(op); });

for (auto op : addscops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value scalar = op.getScalar();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TAddSCOp>(
      op,
      TypeRange{},
      src0,
      scalar,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TAndOp> andops;
func.walk([&](mlir::pto::TAndOp op) { andops.push_back(op); });

for (auto op : andops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TAndOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TConcatOp> concats;
func.walk([&](mlir::pto::TConcatOp op) { concats.push_back(op); });

for (auto op : concats) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TConcatOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TConcatidxOp> concatIdxs;
func.walk([&](mlir::pto::TConcatidxOp op) { concatIdxs.push_back(op); });

IRRewriter rewriter(ctx);
for (auto op : concatIdxs) {
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value src0Idx = op.getSrc0Idx();
  Value src1Idx = op.getSrc1Idx();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto src0IdxTy = dyn_cast<MemRefType>(src0Idx.getType());
  auto src1IdxTy = dyn_cast<MemRefType>(src1Idx.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !src0IdxTy || !src1IdxTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TConcatidxOp>(
      op,
      TypeRange{},
      src0,
      src1,
      src0Idx,
      src1Idx,
      dst);
}

DefaultInlineVector<mlir::pto::TAndSOp> andsops;
func.walk([&](mlir::pto::TAndSOp op) { andsops.push_back(op); });

for (auto op : andsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TAndSOp>(
      op,
      TypeRange{},
      src,
      scalar,
      dst);
}

DefaultInlineVector<mlir::pto::TCIOp> ciops;
func.walk([&](mlir::pto::TCIOp op) { ciops.push_back(op); });

for (auto op : ciops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value s = op->getOperand(0);
  Value dst = op.getDst();
  bool descending = op.getDescending();

  auto sTy = dyn_cast<IntegerType>(s.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!sTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TCIOp>(
      op,
      TypeRange{},
      s,
      dst,
      descending);
}

DefaultInlineVector<mlir::pto::TCmpOp> cmpops;
func.walk([&](mlir::pto::TCmpOp op) { cmpops.push_back(op); });

for (auto op : cmpops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

   auto newOp = rewriter.create<pto::TCmpOp>(
      op.getLoc(),
      TypeRange{},
      src0,
      src1,
      dst);
   
    if (auto a = op.getCmpModeAttr())
      newOp->setAttr("cmpMode", a);

  rewriter.replaceOp(op, newOp->getResults()); // 0 results -> OK
}

DefaultInlineVector<mlir::pto::TCmpSOp> cmpsops;
func.walk([&](mlir::pto::TCmpSOp op) { cmpsops.push_back(op); });

for (auto op : cmpsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  auto scalarTy = scalar.getType();
  bool scalarOk =
      isa<IntegerType, FloatType>(scalarTy); // ScalarType in ODS: int/float
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }
  if (!scalarOk) {
    op.emitError("expects scalar to be an integer or float type");
          return failure();
  }

  auto cmpMode = op.getCmpModeAttr();
  auto newOp = rewriter.create<pto::TCmpSOp>(
      op.getLoc(),
      TypeRange{},
      src,
      scalar,
      cmpMode,
      dst);

  rewriter.replaceOp(op, newOp->getResults()); // 0 results -> OK
}

DefaultInlineVector<mlir::pto::TColExpandOp> colexpand;
func.walk([&](mlir::pto::TColExpandOp op) { colexpand.push_back(op); });

for (auto op : colexpand) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if ( !srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TColExpandOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TColMaxOp> colmaxops;
func.walk([&](mlir::pto::TColMaxOp op) { colmaxops.push_back(op); });

for (auto op : colmaxops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if ( !srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TColMaxOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TColMinOp> colminops;
func.walk([&](mlir::pto::TColMinOp op) { colminops.push_back(op); });

for (auto op : colminops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if ( !srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TColMinOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TColExpandMulOp> colexpandmulops;
func.walk([&](mlir::pto::TColExpandMulOp op) {
  colexpandmulops.push_back(op);
});

for (auto op : colexpandmulops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TColExpandMulOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TColExpandMaxOp> colexpandmaxops;
func.walk([&](mlir::pto::TColExpandMaxOp op) {
  colexpandmaxops.push_back(op);
});

for (auto op : colexpandmaxops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TColExpandMaxOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TColExpandMinOp> colexpandminops;
func.walk([&](mlir::pto::TColExpandMinOp op) {
  colexpandminops.push_back(op);
});

for (auto op : colexpandminops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TColExpandMinOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TColSumOp> colsumops;
func.walk([&](mlir::pto::TColSumOp op) { colsumops.push_back(op); });

for (auto op : colsumops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();
  Value tmp = op.getTmp();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("src/dst are not memref yet");
          return failure();
  }

  // If tmp exists, it must have isBinary attribute
  if (tmp) {
    auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
    if (!tmpTy) {
      op.emitError("tmp is not memref yet");
          return failure();
    }

    // Get isBinary attribute (should exist if tmp exists)
    BoolAttr isBinaryAttr = op.getIsBinaryAttr();
    if (!isBinaryAttr) {
      isBinaryAttr = BoolAttr::get(ctx, false);
    }

    rewriter.replaceOpWithNewOp<pto::TColSumOp>(
        op,
        TypeRange{},
        src,
        tmp,
        dst,
        isBinaryAttr);
  } else {
    // Format 1: no tmp, no isBinary
    // Use generic builder to avoid adding default isBinary attribute
    SmallVector<Value> operands = {src, dst};
    SmallVector<NamedAttribute> attrs;
    // Copy all attributes except isBinary
    for (auto attr : op->getAttrs()) {
      if (attr.getName() != "isBinary") {
        attrs.push_back(attr);
      }
    }
    rewriter.replaceOpWithNewOp<pto::TColSumOp>(
        op,
        TypeRange{},
        operands,
        attrs);
  }
}

DefaultInlineVector<mlir::pto::TCvtOp> cvtops;
func.walk([&](mlir::pto::TCvtOp op) { cvtops.push_back(op); });

for (auto op : cvtops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  auto rmodeAttr = op.getRmodeAttr(); // PTO_RoundModeAttr
  auto satModeAttr = op.getSatModeAttr();

  auto newOp = rewriter.create<pto::TCvtOp>(
      op.getLoc(),
      TypeRange{},
      src,
      dst,
      rmodeAttr,
      satModeAttr);

  rewriter.replaceOp(op, newOp->getResults());
}

DefaultInlineVector<mlir::pto::TDivOp> divops;
func.walk([&](mlir::pto::TDivOp op) { divops.push_back(op); });

for (auto op : divops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TDivOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TDivSOp> divsops;
func.walk([&](mlir::pto::TDivSOp op) { divsops.push_back(op); });

for (auto op : divsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scale = op.getScalar();
  Value dst = op.getDst();

  // Check types - they might still be TileBufType or already converted to MemRefType
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto srcTileTy = dyn_cast<mlir::pto::TileBufType>(src.getType());
  auto scaleTileTy = dyn_cast<mlir::pto::TileBufType>(scale.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  auto dstTileTy = dyn_cast<mlir::pto::TileBufType>(dst.getType());
  
  // Determine which operand is tile-like and which is scalar-like.
  // Keep the original operand order (set by parser textual form).
  // Check if src is memref/tensor/tile (not scalar)
  bool srcIsMemref = (srcTy != nullptr || srcTileTy != nullptr || 
                      isa<RankedTensorType>(src.getType()) ||
                      isa<mlir::pto::PartitionTensorViewType>(src.getType()));
  // Check if scale is memref/tensor/tile (not scalar)
  bool scaleIsMemref = (isa<MemRefType>(scale.getType()) || 
                        scaleTileTy != nullptr ||
                        isa<RankedTensorType>(scale.getType()) ||
                        isa<mlir::pto::PartitionTensorViewType>(scale.getType()));

  // Type validation - ensure we have the right types
  if (!srcIsMemref && !scaleIsMemref) {
    op.emitError("at least one operand (src or scale) must be tile_buf or memref");
          return failure();
  }
  if (srcIsMemref && scaleIsMemref) {
    op.emitError("exactly one operand (src or scale) must be tile_buf or memref, the other must be scalar");
          return failure();
  }

  if (!dstTy && !dstTileTy) {
    op.emitError("dst operand must be tile_buf or memref");
          return failure();
  }
  rewriter.replaceOpWithNewOp<pto::TDivSOp>(
      op,
      TypeRange{},
      src,
      scale,
      dst);
}

DefaultInlineVector<mlir::pto::TExpandsOp> expandsops;
func.walk([&](mlir::pto::TExpandsOp op) { expandsops.push_back(op); });

for (auto op : expandsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TExpandsOp>(
      op,
      TypeRange{},
      scalar,
      dst);
}

DefaultInlineVector<mlir::pto::TExtractOp> extractops;
func.walk([&](mlir::pto::TExtractOp op) { extractops.push_back(op); });

for (auto op : extractops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value indexRow = op.getIndexRow();
  Value indexCol = op.getIndexCol();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto indexRowTy = dyn_cast<IndexType>(indexRow.getType());
  auto indexColTy = dyn_cast<IndexType>(indexCol.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !indexRowTy || !indexColTy || !dstTy) {
    op.emitError("ins/outs are not correct yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TExtractOp>(
      op,
      TypeRange{},
      src,
      indexRow,
      indexCol,
      dst);
}

DefaultInlineVector<mlir::pto::TFillPadOp> fillpadops;
func.walk([&](mlir::pto::TFillPadOp op) { fillpadops.push_back(op); });

for (auto op : fillpadops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TFillPadOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TFillPadInplaceOp> fillpadInplaceOps;
func.walk(
    [&](mlir::pto::TFillPadInplaceOp op) { fillpadInplaceOps.push_back(op); });

for (auto op : fillpadInplaceOps) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TFillPadInplaceOp>(
      op,
      TypeRange{},
      src,
      dst);
}

// --- TSetValOp [Dst, Offset, Val] ---
// Lower tile-world scalar write to memref-world SETVAL DPS op.
DefaultInlineVector<mlir::pto::TSetValOp> tsetvalops;
func.walk([&](mlir::pto::TSetValOp op) { tsetvalops.push_back(op); });

for (auto op : tsetvalops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value dst = op.getDst();
  Value offset = op.getOffset();
  Value val = op.getVal();

  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!dstTy) {
    op.emitError("dst is not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TSetValOp>(
      op,
      TypeRange{},
      dst,
      offset,
      val);
}

// --- TGetValOp [Src, Offset] -> Scalar ---
// Lower tile-world scalar read to memref-world GETVAL DPS op.
DefaultInlineVector<mlir::pto::TGetValOp> tgetvalops;
func.walk([&](mlir::pto::TGetValOp op) { tgetvalops.push_back(op); });

for (auto op : tgetvalops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value offset = op.getOffset();
  Type dstType = op.getDst().getType();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    op.emitError("src is not memref yet");
          return failure();
  }

  auto newOp = rewriter.create<pto::TGetValOp>(
      op.getLoc(),
      dstType,
      src,
      offset);
  rewriter.replaceOp(op, newOp.getDst());
}

DefaultInlineVector<mlir::pto::TGatherOp> gatherops;
func.walk([&](mlir::pto::TGatherOp op) { gatherops.push_back(op); });

for (auto op : gatherops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();
  Value cdst = op.getCdst();
  Value indices = op.getIndices();
  Value tmp = op.getTmp();
  Value kValue = op.getKValue();
  auto maskPattern = op.getMaskPatternAttr();
  auto cmpMode = op.getCmpModeAttr();
  auto offset = op.getOffsetAttr();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());

  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  if (maskPattern) {
    rewriter.replaceOpWithNewOp<pto::TGatherOp>(
        op,
        TypeRange{},
        src,
        dst,
        /*cdst=*/Value(),
        /*indices=*/Value(),
        /*tmp=*/Value(),
        /*kValue=*/Value(),
        /*maskPattern=*/maskPattern,
        /*cmpMode=*/pto::CmpModeAttr(),
        /*offset=*/IntegerAttr());
    continue;
  }

  if (cdst || kValue) {
    auto cdstTy = dyn_cast<MemRefType>(cdst.getType());
    auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
    if (!cdstTy || !tmpTy) {
      op.emitError("compare-form tgather expects cdst/tmp to be memref yet");
          return failure();
    }

    rewriter.replaceOpWithNewOp<pto::TGatherOp>(
        op,
        TypeRange{},
        src,
        dst,
        cdst,
        /*indices=*/Value(),
        tmp,
        kValue,
        /*maskPattern=*/pto::MaskPatternAttr(),
        cmpMode,
        offset);
    continue;
  }

  if (indices || tmp) {
    auto indicesTy = dyn_cast<MemRefType>(indices.getType());
    auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
    if (!indicesTy || !tmpTy) {
      op.emitError("index-form tgather expects indices/tmp to be memref yet");
          return failure();
    }

    rewriter.replaceOpWithNewOp<pto::TGatherOp>(
        op,
        TypeRange{},
        src,
        dst,
        /*cdst=*/Value(),
        indices,
        tmp,
        /*kValue=*/Value(),
        /*maskPattern=*/pto::MaskPatternAttr(),
        /*cmpMode=*/pto::CmpModeAttr(),
        /*offset=*/IntegerAttr());
    continue;
  }

  op.emitError("expects tgather to be in mask, index+tmp, or compare+tmp form");
          return failure();
}

DefaultInlineVector<mlir::pto::TGatherBOp> gatherbops;
func.walk([&](mlir::pto::TGatherBOp op) { gatherbops.push_back(op); });

for (auto op : gatherbops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value offsets = op.getOffsets();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto offsetsTy = dyn_cast<MemRefType>(offsets.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !offsetsTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TGatherBOp>(
      op,
      TypeRange{},
      src,
      offsets,
      dst);
}

DefaultInlineVector<mlir::pto::TLogOp> logops;
func.walk([&](mlir::pto::TLogOp op) { logops.push_back(op); });

for (auto op : logops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TLogOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TLReluOp> lreluops;
func.walk([&](mlir::pto::TLReluOp op) { lreluops.push_back(op); });

for (auto op : lreluops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value slope = op.getSlope();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto slopeTy = dyn_cast<FloatType>(slope.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !slopeTy || !dstTy) {
    op.emitError("ins/outs are not correct type yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TLReluOp>(
      op,
      TypeRange{},
      src,
      slope,
      dst);
}

DefaultInlineVector<mlir::pto::TMaxOp> maxops;
func.walk([&](mlir::pto::TMaxOp op) { maxops.push_back(op); });

for (auto op : maxops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TMaxOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TMaxSOp> maxsops;
func.walk([&](mlir::pto::TMaxSOp op) { maxsops.push_back(op); });

for (auto op : maxsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  bool scalarIsScalar = isa<IntegerType, FloatType>(scalar.getType());
  if (!srcTy || !scalarIsScalar || !dstTy) {
    op.emitError("expects src/dst to be memref and scalar to be integer/float");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TMaxSOp>(
      op,
      TypeRange{},
      src,
      scalar,
      dst);
}

DefaultInlineVector<mlir::pto::TMinOp> minops;
func.walk([&](mlir::pto::TMinOp op) { minops.push_back(op); });

for (auto op : minops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TMinOp>(
      op,
      TypeRange{},
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TMinSOp> minsops;
func.walk([&](mlir::pto::TMinSOp op) { minsops.push_back(op); });

for (auto op : minsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  bool scalarIsScalar = isa<IntegerType, FloatType>(scalar.getType());
  if (!srcTy || !scalarIsScalar || !dstTy) {
    op.emitError("expects src/dst to be memref and scalar to be integer/float");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TMinSOp>(
      op,
      TypeRange{},
      src,
      scalar,
      dst);
}

DefaultInlineVector<mlir::pto::TMovFPOp> movfpops;
func.walk([&](mlir::pto::TMovFPOp op) { movfpops.push_back(op); });

for (auto op : movfpops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value fp = op.getFp();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto fpTy = dyn_cast<MemRefType>(fp.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !fpTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TMovFPOp>(
      op,
      TypeRange{},
      src,
      fp,
      dst);
}

DefaultInlineVector<mlir::pto::TQuantOp> quantops;
func.walk([&](mlir::pto::TQuantOp op) { quantops.push_back(op); });

for (auto op : quantops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value fp = op.getFp();
  Value offset = op.getOffset();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto fpTy = dyn_cast<MemRefType>(fp.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !fpTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }
  if (offset && !dyn_cast<MemRefType>(offset.getType())) {
    op.emitError("offset is not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TQuantOp>(
      op,
      TypeRange{},
      src,
      fp,
      offset,
      dst,
      op.getQuantTypeAttr());
}

DefaultInlineVector<mlir::pto::TMrgSortOp> mrgsortops;
func.walk([&](mlir::pto::TMrgSortOp op) { mrgsortops.push_back(op); });

for (auto op : mrgsortops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  if (op.isFormat1()) {
    Value src = op.getSrc();
    Value dst = op.getDst();
    Value blockLenVal = op.getBlockLen();

    auto srcTy = dyn_cast<MemRefType>(src.getType());
    auto dstTy = dyn_cast<MemRefType>(dst.getType());
    if (!srcTy || !dstTy) {
      op.emitError("ins/outs are not memref yet");
          return failure();
    }

    rewriter.replaceOpWithNewOp<pto::TMrgSortOp>(
        op,
        TypeRange{},
        ValueRange{src},
        blockLenVal,
        ValueRange{dst},
        Value() /*tmp*/,
        Value() /*excuted*/,
        op.getExhaustedAttr());
  } else if (op.isFormat2()) {
    bool allMemRef = true;
    for (Value v : op.getSrcs())
      if (!dyn_cast<MemRefType>(v.getType())) { allMemRef = false; break; }
    if (!allMemRef) {
      op.emitError("format2 ins/outs are not memref yet");
          return failure();
    }
    if (op.getDsts().size() != 1u || !op.getTmp()) {
      op.emitError("format2 expects outs(dst) and ins(tmp)");
          return failure();
    }

    Value dst = op.getDst();
    Value tmp = op.getTmp();
    Value excuted = op.getExcuted();
    if (!dyn_cast<MemRefType>(dst.getType()) || !dyn_cast<MemRefType>(tmp.getType())) {
      op.emitError("format2 dst/tmp must be memref");
          return failure();
    }
    if (!dyn_cast<VectorType>(excuted.getType())) {
      op.emitError("format2 outs(excuted) must be vector");
          return failure();
    }

    rewriter.replaceOpWithNewOp<pto::TMrgSortOp>(
        op,
        TypeRange{},
        op.getSrcs(),
        Value() /*blockLen*/,
        ValueRange{dst},
        tmp,
        excuted,
        op.getExhaustedAttr());
  } else {
    op.emitError("tmrgsort must be format1 or format2");
          return failure();
  }
}

DefaultInlineVector<mlir::pto::TNegOp> negops;
func.walk([&](mlir::pto::TNegOp op) { negops.push_back(op); });

for (auto op : negops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TNegOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TNotOp> notops;
func.walk([&](mlir::pto::TNotOp op) { notops.push_back(op); });

for (auto op : notops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TNotOp>(
      op,
      TypeRange{},
      src,
      dst);
}

DefaultInlineVector<mlir::pto::TOrOp> orops;
func.walk([&](mlir::pto::TOrOp op) { orops.push_back(op); });

for (auto op : orops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TOrOp>(
      op,
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TOrSOp> orsops;
func.walk([&](mlir::pto::TOrSOp op) { orsops.push_back(op); });

for (auto op : orsops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value dst = op.getDst();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto scalarTy = dyn_cast<IntegerType>(scalar.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!srcTy || !scalarTy || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TOrSOp>(
      op,
      TypeRange{},
      src,
      scalar,
      dst);
}

DefaultInlineVector<mlir::pto::TPartAddOp> partaddops;
func.walk([&](mlir::pto::TPartAddOp op) { partaddops.push_back(op); });

for (auto op : partaddops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TPartAddOp>(
      op,
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::TPartMulOp> partmulops;
func.walk([&](mlir::pto::TPartMulOp op) { partmulops.push_back(op); });

for (auto op : partmulops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src0 = op.getSrc0();
  Value src1 = op.getSrc1();
  Value dst = op.getDst();

  auto src0Ty = dyn_cast<MemRefType>(src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(src1.getType());
  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!src0Ty || !src1Ty || !dstTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TPartMulOp>(
      op,
      src0,
      src1,
      dst);
}

DefaultInlineVector<mlir::pto::MGatherOp> mgatherops;
func.walk([&](mlir::pto::MGatherOp op) { mgatherops.push_back(op); });

for (auto op : mgatherops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value dst = op.getDst();
  Value idx = op.getIdx();
  Value mem = op.getMem();

  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  auto idxTy = dyn_cast<MemRefType>(idx.getType());
  auto memTy = dyn_cast<MemRefType>(mem.getType());
  if (!dstTy || !idxTy || !memTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::MGatherOp>(
      op,
      TypeRange{},
      mem,
      idx,
      dst,
      op.getGatherOobAttr());
}

DefaultInlineVector<mlir::pto::MScatterOp> mascatterops;
func.walk([&](mlir::pto::MScatterOp op) { mascatterops.push_back(op); });

for (auto op : mascatterops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();
  Value idx = op.getIdx();
  Value mem = op.getMem();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto idxTy = dyn_cast<MemRefType>(idx.getType());
  auto memTy = dyn_cast<MemRefType>(mem.getType());
  if (!srcTy || !idxTy || !memTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::MScatterOp>(
      op,
      TypeRange{},
      src,
      idx,
      mem,
      op.getScatterAtomicOpAttr(),
      op.getScatterOobAttr());
}
DefaultInlineVector<mlir::pto::TPrintOp> printops;
func.walk([&](mlir::pto::TPrintOp op) { printops.push_back(op); });

for (auto op : printops) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);

  Value src = op.getSrc();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    op.emitError("ins/outs are not memref yet");
          return failure();
  }

  rewriter.replaceOpWithNewOp<pto::TPrintOp>(
      op,
      TypeRange{},
      src);
}


  return success();
}

} // namespace mlir::pto
