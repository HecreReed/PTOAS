// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOViewToMemref.cpp ------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Lower PTO tile/view operations to memref-based IR while preserving tile
// metadata through binding ops and SSA backtracking.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "Utils.h" // 假设包含一些通用的工具函数

#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOVIEWTOMEMREF

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

namespace {

static void markForceDynamicValidShape(Operation *op, bool force,
                                       MLIRContext *ctx);

static Type convertPTOTypeToMemRef(Type t);
static LogicalResult lowerPartitionViewOps(func::FuncOp func,
                                           MLIRContext *ctx);
static LogicalResult lowerSubsetOps(func::FuncOp func, MLIRContext *ctx);
static LogicalResult lowerTileBufViewLikeOps(func::FuncOp func,
                                             MLIRContext *ctx);

// =============================================================================
// Helper: Metadata Backtracking (核心机制)
// =============================================================================
// 从一个 MemRef Value 向上回溯，找到它绑定的 TileBufConfig。
// 这解决了 "Type Erasure" 问题：memref 类型本身不包含 config，但 SSA 定义链包含。
static mlir::pto::TileBufConfigAttr lookupConfig(Value v) {
  // 1. 最直接的情况：它就是 bind_tile 的结果
  if (auto bind = v.getDefiningOp<mlir::pto::BindTileOp>()) {
    return bind.getConfig();
  }
  // PointerCastOp can also carry tile metadata (used when alloc_tile specifies
  // an explicit address).
  if (auto pc = v.getDefiningOp<mlir::pto::PointerCastOp>()) {
    if (auto cfg = pc.getConfig())
      return *cfg;
    return {};
  }
  
  // 2. 穿透 View 操作 (SubView, Cast 等) 向上查找
  if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
    return lookupConfig(subview.getSource());
  }
  if (auto cast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    return lookupConfig(cast.getSource());
  }
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    return lookupConfig(cast.getSource());
  }
  
  // 如果追溯到 BlockArgument (函数参数) 或其他无法穿透的 Op，则返回空
  return {}; 
}

// =============================================================================
// Helper: Valid dims backtracking (v_row / v_col)
// =============================================================================
static void lookupValidDims(Value v, Value &vRow, Value &vCol) {
  if (auto bind = v.getDefiningOp<mlir::pto::BindTileOp>()) {
    vRow = bind.getValidRow();
    vCol = bind.getValidCol();
    return;
  }
  if (auto pc = v.getDefiningOp<mlir::pto::PointerCastOp>()) {
    vRow = pc.getValidRow();
    vCol = pc.getValidCol();
    return;
  }
  if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
    lookupValidDims(subview.getSource(), vRow, vCol);
    return;
  }
  if (auto cast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    lookupValidDims(cast.getSource(), vRow, vCol);
    return;
  }
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    lookupValidDims(cast.getSource(), vRow, vCol);
    return;
  }
  vRow = Value();
  vCol = Value();
}

// =============================================================================
// Helper Functions for Layout Normalization
// =============================================================================

struct TileLayoutInfo {
  int64_t rowStride = 1;
  int64_t colStride = 1;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  bool boxed = false; // slayout != NoneBox
};

struct TileLayoutConfig {
  int32_t bLayout = 0;
  int32_t sLayout = 0;
  int32_t fractalSize = 512;
  int32_t compactMode = 0;
};

static int64_t getElemBytes(Type elemTy) {
  if (auto ft = elemTy.dyn_cast<FloatType>()) {
    if (ft.isF16() || ft.isBF16()) return 2;
    if (ft.isF32()) return 4;
    if (ft.isF64()) return 8;
  }
  if (auto it = elemTy.dyn_cast<IntegerType>()) {
    int64_t bytes = it.getWidth() / 8;
    return bytes > 0 ? bytes : 1;
  }
  return -1;
}

template <typename EnumAttrTy>
static bool readEnumAttrOrIntegerI32(Attribute attr, int32_t &out) {
  if (auto enumAttr = dyn_cast<EnumAttrTy>(attr)) {
    out = static_cast<int32_t>(enumAttr.getValue());
    return true;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(intAttr.getInt());
    return true;
  }
  return false;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  return readEnumAttrOrIntegerI32<BLayoutAttr>(attr, out);
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  return readEnumAttrOrIntegerI32<SLayoutAttr>(attr, out);
}

static bool readCompactModeI32(Attribute attr, int32_t &out) {
  return readEnumAttrOrIntegerI32<CompactModeAttr>(attr, out);
}

static Value peelIndexLikeCast(Value value) {
  while (true) {
    if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
      value = castOp.getIn();
      continue;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtSIOp>()) {
      value = extOp.getIn();
      continue;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
      value = extOp.getIn();
      continue;
    }
    if (auto truncOp = value.getDefiningOp<arith::TruncIOp>()) {
      value = truncOp.getIn();
      continue;
    }
    return value;
  }
}

static bool getConstIndexValue(Value value, int64_t &out) {
  value = peelIndexLikeCast(value);
  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = constIndex.value();
    return true;
  }
  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = constInt.value();
    return true;
  }
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  auto intAttr =
      constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr();
  if (!intAttr)
    return false;
  out = intAttr.getInt();
  return true;
}

static TileLayoutConfig getTileLayoutConfig(mlir::pto::TileBufConfigAttr cfg) {
  TileLayoutConfig config;
  (void)readBLayoutI32(cfg.getBLayout(), config.bLayout);
  (void)readSLayoutI32(cfg.getSLayout(), config.sLayout);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    config.fractalSize = static_cast<int32_t>(attr.getInt());
  (void)readCompactModeI32(cfg.getCompactMode(), config.compactMode);
  return config;
}

static bool getFractal512InnerExtent(int64_t elemBytes, int64_t &extent) {
  switch (elemBytes) {
  case 1:
    extent = 32;
    return true;
  case 2:
    extent = 16;
    return true;
  case 4:
    extent = 8;
    return true;
  case 8:
    extent = 4;
    return true;
  case 16:
    extent = 2;
    return true;
  case 32:
    extent = 1;
    return true;
  default:
    return false;
  }
}

static bool computeBoxInnerShape(const TileLayoutConfig &config, Type elemTy,
                                 TileLayoutInfo &info) {
  info.boxed = config.sLayout != 0;
  if (!info.boxed) {
    info.innerRows = 1;
    info.innerCols = 1;
    return true;
  }

  int64_t elemBytes = getElemBytes(elemTy);
  if (elemBytes <= 0)
    return false;

  switch (config.fractalSize) {
  case 1024:
    info.innerRows = 16;
    info.innerCols = 16;
    return true;
  case 32:
    info.innerRows = 16;
    info.innerCols = 2;
    return true;
  case 512:
    if (config.sLayout == 1) {
      info.innerRows = 16;
      return getFractal512InnerExtent(elemBytes, info.innerCols);
    }
    if (config.sLayout == 2) {
      if (!getFractal512InnerExtent(elemBytes, info.innerRows))
        return false;
      info.innerCols = 16;
      return true;
    }
    return false;
  default:
    return false;
  }
}

static bool computeTilePointerStrides(const TileLayoutConfig &config,
                                      ArrayRef<int64_t> shape,
                                      TileLayoutInfo &info) {
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  auto applyCompactToMajorStride = [&](int64_t majorStride) -> int64_t {
    if (config.compactMode == 2)
      return majorStride + 1;
    return majorStride;
  };
  if (!info.boxed) {
    if (config.bLayout == 1) {
      info.rowStride = 1;
      info.colStride = applyCompactToMajorStride(rows);
      return true;
    }
    info.rowStride = applyCompactToMajorStride(cols);
    info.colStride = 1;
    return true;
  }

  if (config.bLayout == 1) {
    if (config.sLayout != 1)
      return false;
    info.rowStride = info.innerCols;
    info.colStride = applyCompactToMajorStride(rows);
    return true;
  }

  info.rowStride = applyCompactToMajorStride(cols);
  info.colStride = info.innerRows;
  return true;
}

static bool computeTileLayoutInfo(mlir::pto::TileBufConfigAttr cfg, Type elemTy,
                                  ArrayRef<int64_t> shape,
                                  TileLayoutInfo &info) {
  if (shape.size() != 2 || llvm::is_contained(shape, ShapedType::kDynamic))
    return false;

  TileLayoutConfig config = getTileLayoutConfig(cfg);
  return computeBoxInnerShape(config, elemTy, info) &&
         computeTilePointerStrides(config, shape, info);
}

static void collectAffineAddTerms(AffineExpr root,
                                  SmallVectorImpl<AffineExpr> &terms) {
  SmallVector<AffineExpr, 4> pending{root};
  while (!pending.empty()) {
    AffineExpr current = pending.pop_back_val();
    auto addExpr = current.dyn_cast<AffineBinaryOpExpr>();
    if (!addExpr || addExpr.getKind() != AffineExprKind::Add) {
      terms.push_back(current);
      continue;
    }
    pending.push_back(addExpr.getRHS());
    pending.push_back(addExpr.getLHS());
  }
}

static bool tryAssignAffineStride(AffineExpr expr,
                                  MutableArrayRef<int64_t> strides) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
    strides[dim.getPosition()] = 1;
    return true;
  }

  auto mulExpr = expr.dyn_cast<AffineBinaryOpExpr>();
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;

  auto assignStride = [&](AffineExpr dimExpr,
                          AffineExpr constantExpr) -> bool {
    auto dim = dimExpr.dyn_cast<AffineDimExpr>();
    auto constant = constantExpr.dyn_cast<AffineConstantExpr>();
    if (!dim || !constant)
      return false;
    strides[dim.getPosition()] = constant.getValue();
    return true;
  };
  return assignStride(mulExpr.getLHS(), mulExpr.getRHS()) ||
         assignStride(mulExpr.getRHS(), mulExpr.getLHS());
}

static void decomposeStridedLayout(AffineMap map,
                                   SmallVectorImpl<int64_t> &strides) {
  strides.assign(map.getNumDims(), 0);
  if (map.getNumResults() != 1)
    return;

  SmallVector<AffineExpr, 4> terms;
  collectAffineAddTerms(map.getResult(0), terms);
  for (AffineExpr term : terms)
    (void)tryAssignAffineStride(term, strides);
}

static Value makeIndexConstant(IRRewriter &rewriter, Location loc,
                               int64_t value) {
  return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                            rewriter.getIndexAttr(value));
}

static SmallVector<int64_t> computeCompactStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    if (shape[i] != ShapedType::kDynamic)
      stride *= shape[i];
  }
  return strides;
}

static void materializeStaticValidDims(IRRewriter &rewriter, Location loc,
                                       mlir::pto::TileBufType tbTy, Value &vRow,
                                       Value &vCol) {
  ArrayRef<int64_t> validShape = tbTy.getValidShape();
  if (tbTy.hasDynamicValid())
    return;
  if (validShape.size() >= 1 && validShape[0] >= 0)
    vRow = makeIndexConstant(rewriter, loc, validShape[0]);
  if (validShape.size() >= 2 && validShape[1] >= 0)
    vCol = makeIndexConstant(rewriter, loc, validShape[1]);
}

static bool checkMultipleOf(Operation *op, int64_t value, int64_t divisor,
                            StringRef label) {
  if (divisor <= 0) {
    op->emitError("boxed layout requires positive divisor for ") << label;
    return false;
  }
  if (value % divisor == 0)
    return true;
  op->emitError("boxed layout requires ")
      << label << " multiple of " << divisor << ", got " << value;
  return false;
}

// 确保 Value 是 Index 类型
static Value ensureIndex(IRRewriter &rewriter, Location loc, Value v,
                         Operation *anchorOp) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), v);
  if (anchorOp)
    anchorOp->emitError() << "expected index or integer, but got " << v.getType();
  return Value();
}

static Value computeSubsetValidDim(IRRewriter &rewriter, Location loc,
                                   Value parentValid, Value offset,
                                   int64_t size, Operation *anchorOp) {
  Value sizeVal = rewriter.create<arith::ConstantIndexOp>(loc, size);
  if (!parentValid)
    return sizeVal;

  int64_t pvConst = 0, offConst = 0;
  if (getConstIndexValue(parentValid, pvConst) &&
      getConstIndexValue(offset, offConst)) {
    int64_t diff = 0;
    if (pvConst > 0) {
      int64_t offMod = offConst % pvConst;
      if (offMod < 0)
        offMod += pvConst;
      diff = pvConst - offMod; // in [1, pvConst] when pvConst>0
    }
    if (diff < 0)
      diff = 0;
    int64_t clipped = std::min<int64_t>(size, diff);
    return rewriter.create<arith::ConstantIndexOp>(loc, clipped);
  }

  Value pv = ensureIndex(rewriter, loc, parentValid, anchorOp);
  Value off = ensureIndex(rewriter, loc, offset, anchorOp);

  // Use the same "periodic valid dims" rule as SubsetOp::inferReturnTypes:
  // diff = pv - (off % pv), so offsets that land on the next tile (off == pv)
  // still produce a full valid dim (diff == pv), instead of 0.
  Type i64Ty = rewriter.getI64Type();
  Value pvI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pv);
  Value offI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, off);
  Value remI64 = rewriter.create<arith::RemUIOp>(loc, offI64, pvI64);
  Value diffI64 = rewriter.create<arith::SubIOp>(loc, pvI64, remI64);
  Value diff = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   diffI64);

  Value lt = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, diff,
                                            sizeVal);
  return rewriter.create<arith::SelectOp>(loc, lt, diff, sizeVal);
}

static void dumpPretty(Operation *op, llvm::raw_ostream &os) {
  OpPrintingFlags flags;
  flags.useLocalScope();            
  AsmState state(op, flags);
  op->print(os, state);
  os << "\n";
  os.flush();
}

// =============================================================================
// Type Converter Logic
// =============================================================================

static SmallVector<int64_t> buildTileMemRefStrides(mlir::pto::TileBufType tbTy) {
  SmallVector<int64_t> strides;
  auto shape = tbTy.getShape();
  TileLayoutInfo info;
  if (computeTileLayoutInfo(tbTy.getConfigAttr(), tbTy.getElementType(), shape,
                            info)) {
    return {info.rowStride, info.colStride};
  }
  return computeCompactStrides(shape);
}

static Type convertTileBufTypeToMemRef(mlir::pto::TileBufType tbTy) {
  auto layoutAttr = StridedLayoutAttr::get(tbTy.getContext(),
                                           ShapedType::kDynamic,
                                           buildTileMemRefStrides(tbTy));
  return MemRefType::get(tbTy.getShape(), tbTy.getElementType(), layoutAttr,
                         tbTy.getMemorySpace());
}

static Type convertPTOTypeToMemRef(Type t) {
  // 1. 处理 !pto.ptr<T>
  if (auto pty = dyn_cast<mlir::pto::PtrType>(t)) {
    return MemRefType::get({ShapedType::kDynamic}, pty.getElementType(),
                           MemRefLayoutAttrInterface(), Attribute());
  }
  
  // 2. 处理 !pto.tile_buf<...>
  if (auto tbTy = dyn_cast<mlir::pto::TileBufType>(t))
    return convertTileBufTypeToMemRef(tbTy);
  // 其他类型透传
  return t;
}

// Ensure scf.if result types follow the rewritten yield operand types.
// PTOViewToMemref rewrites tile values to memref in branch bodies, but scf.if
// result types are not auto-updated by those op-local rewrites.
static LogicalResult reconcileSCFIfResultTypes(func::FuncOp func) {
  SmallVector<scf::IfOp, 8> ifOps;
  func.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  for (scf::IfOp ifOp : ifOps) {
    if (ifOp.getNumResults() == 0)
      continue;

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (!thenYield || !elseYield) {
      ifOp.emitError("result-bearing scf.if must end with scf.yield in both "
                     "then/else regions");
      return failure();
    }

    if (thenYield.getNumOperands() != ifOp.getNumResults() ||
        elseYield.getNumOperands() != ifOp.getNumResults()) {
      ifOp.emitError("scf.if result count does not match yielded values");
      return failure();
    }

    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Type thenTy = thenYield.getOperand(i).getType();
      Type elseTy = elseYield.getOperand(i).getType();
      if (thenTy != elseTy) {
        ifOp.emitError() << "scf.if branch yield type mismatch at result #" << i
                         << ": then=" << thenTy << ", else=" << elseTy;
        return failure();
      }

      if (ifOp.getResult(i).getType() != thenTy)
        ifOp.getResult(i).setType(thenTy);
    }
  }

  return success();
}

static LogicalResult markLoweredSetValidShapeOps(func::FuncOp func,
                                                 MLIRContext *ctx) {
  WalkResult result = func.walk([&](mlir::pto::SetValidShapeOp op) {
    if (isa<MemRefType>(op.getSource().getType())) {
      if (!lookupConfig(op.getSource())) {
        op.emitError(
            "set_validshape requires a locally bound tile source; function "
            "arguments/results are unsupported");
        return WalkResult::interrupt();
      }
      op->setAttr(kLoweredSetValidShapeAttrName, UnitAttr::get(ctx));
      return WalkResult::advance();
    }
    op->removeAttr(kLoweredSetValidShapeAttrName);
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static void markForceDynamicValidShape(Operation *op, bool force,
                                       MLIRContext *ctx) {
  if (force) {
    op->setAttr(kForceDynamicValidShapeAttrName, UnitAttr::get(ctx));
    return;
  }
  op->removeAttr(kForceDynamicValidShapeAttrName);
}

static void rewriteFunctionSignature(func::FuncOp func, MLIRContext *ctx) {
  Block &entry = func.front();
  auto fnTy = func.getFunctionType();

  SmallVector<Type> newInputs;
  for (Type type : fnTy.getInputs())
    newInputs.push_back(convertPTOTypeToMemRef(type));

  SmallVector<Type> newResults;
  for (Type type : fnTy.getResults())
    newResults.push_back(convertPTOTypeToMemRef(type));

  for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
    if (entry.getArgument(i).getType() != newInputs[i])
      entry.getArgument(i).setType(newInputs[i]);
  }
  func.setFunctionType(FunctionType::get(ctx, newInputs, newResults));
}

static LogicalResult lowerSingleAllocTileOp(mlir::pto::AllocTileOp op,
                                            MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
  if (!tbTy)
    return success();

  SmallVector<int64_t, 4> shape(tbTy.getShape().begin(), tbTy.getShape().end());
  Type elemTy = tbTy.getElementType();
  SmallVector<int64_t> strides = buildTileMemRefStrides(tbTy);
  auto targetLayout =
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides);
  auto targetType =
      MemRefType::get(shape, elemTy, targetLayout, tbTy.getMemorySpace());

  Value vRow = op.getValidRow();
  Value vCol = op.getValidCol();
  materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

  auto configAttr = tbTy.getConfigAttr();
  if (!configAttr)
    configAttr = pto::TileBufConfigAttr::getDefault(ctx);

  if (Value addr = op.getAddr()) {
    auto pc = rewriter.create<pto::PointerCastOp>(
        loc, targetType, ValueRange{addr}, vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(pc, tbTy.hasDynamicValid(), ctx);
    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, targetType, pc.getResult(), vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
    rewriter.replaceOp(op, bindOp.getResult());
    return success();
  }

  auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides);
  auto allocType = MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());
  Value alloc = rewriter.create<memref::AllocOp>(loc, allocType);
  auto bindOp = rewriter.create<pto::BindTileOp>(
      loc, targetType, alloc, vRow ? vRow : Value(), vCol ? vCol : Value(),
      configAttr);
  markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
  rewriter.replaceOp(op, bindOp.getResult());
  return success();
}

static LogicalResult lowerAllocTileOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::AllocTileOp, 8> allocTiles;
  func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

  for (auto op : allocTiles)
    if (failed(lowerSingleAllocTileOp(op, ctx)))
      return failure();
  return success();
}

static LogicalResult lowerDeclareTileOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::DeclareTileOp, 8> declaredTiles;
  func.walk([&](mlir::pto::DeclareTileOp op) { declaredTiles.push_back(op); });

  for (auto op : declaredTiles) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getTile().getType());
    if (!tbTy) {
      op.emitError("declare_tile result must be tile_buf type");
      return failure();
    }

    auto targetType = dyn_cast<MemRefType>(convertPTOTypeToMemRef(tbTy));
    if (!targetType) {
      op.emitError("failed to convert declare_tile result to memref type");
      return failure();
    }

    auto configAttr = tbTy.getConfigAttr();
    if (!configAttr)
      configAttr = pto::TileBufConfigAttr::getDefault(ctx);

    Value vRow;
    Value vCol;
    materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

    auto declaredMemRef =
        rewriter.create<pto::DeclareTileMemRefOp>(loc, targetType);
    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, targetType, declaredMemRef.getResult(), vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
    rewriter.replaceOp(op, bindOp.getResult());
  }
  return success();
}

static void foldAddPtrIntoViewBase(IRRewriter &rewriter, Location loc,
                                   Value &baseBuf, OpFoldResult &off0,
                                   bool &foldedAddPtr) {
  Value cur = baseBuf;
  Value totalOffset;
  while (auto add = cur.getDefiningOp<mlir::pto::AddPtrOp>()) {
    foldedAddPtr = true;
    Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
    totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                              : off;
    cur = add.getOperand(0);
  }
  if (cur == baseBuf)
    return;
  baseBuf = cur;
  off0 = totalOffset ? OpFoldResult(totalOffset) : off0;
}

static LogicalResult lowerSingleMakeTensorViewOp(mlir::pto::MakeTensorViewOp op,
                                                 MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  Value baseBuf = op.getOperand(0);
  OpFoldResult off0 = rewriter.getIndexAttr(0);
  bool foldedAddPtr = false;
  foldAddPtrIntoViewBase(rewriter, loc, baseBuf, off0, foldedAddPtr);

  auto baseMr = dyn_cast<BaseMemRefType>(baseBuf.getType());
  if (!baseMr) {
    op.emitError("make_tensor_view base must be memref");
    return failure();
  }

  size_t rank = op.getShape().size();
  int64_t dyn = ShapedType::kDynamic;
  SmallVector<int64_t> dynStrides(rank, dyn);
  auto layout = StridedLayoutAttr::get(ctx, dyn, dynStrides);
  SmallVector<int64_t> dynShape(rank, dyn);
  auto mrTy = MemRefType::get(dynShape, baseMr.getElementType(), layout,
                              baseMr.getMemorySpace());

  SmallVector<OpFoldResult, 4> sizes;
  for (Value value : op.getShape())
    sizes.push_back(ensureIndex(rewriter, loc, value, op));
  SmallVector<OpFoldResult, 4> strides;
  for (Value value : op.getStrides())
    strides.push_back(ensureIndex(rewriter, loc, value, op));

  auto rc = rewriter.create<memref::ReinterpretCastOp>(loc, mrTy, baseBuf, off0,
                                                       sizes, strides);
  if (foldedAddPtr)
    rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
  if (auto layoutAttr = op.getLayoutAttr())
    rc->setAttr("layout", layoutAttr);
  rewriter.replaceOp(op, rc.getResult());
  return success();
}

static LogicalResult lowerMakeTensorViewOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::MakeTensorViewOp, 8> makeViews;
  func.walk([&](mlir::pto::MakeTensorViewOp op) { makeViews.push_back(op); });

  for (auto op : makeViews)
    if (failed(lowerSingleMakeTensorViewOp(op, ctx)))
      return failure();
  return success();
}

static LogicalResult lowerTensorViewDimOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::GetTensorViewDimOp, 8> tvDims;
  func.walk([&](mlir::pto::GetTensorViewDimOp op) { tvDims.push_back(op); });

  for (auto op : tvDims) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Value view = op.getTensorView();
    auto mrTy = dyn_cast<BaseMemRefType>(view.getType());
    if (!mrTy)
      continue;
    Value dim = rewriter.create<memref::DimOp>(op.getLoc(), view, op.getDimIndex());
    rewriter.replaceOp(op, dim);
  }
  return success();
}

static LogicalResult foldAddPtrIntoLoadScalarOp(mlir::pto::LoadScalarOp op,
                                                MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  Value base = op.getPtr();
  Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);
  bool foldedAddPtr = false;
  while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
    foldedAddPtr = true;
    Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
    totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                              : off;
    base = add.getOperand(0);
  }
  if (!foldedAddPtr)
    return success();

  auto newOp = rewriter.create<pto::LoadScalarOp>(loc, op.getValue().getType(),
                                                  base, totalOffset);
  rewriter.replaceOp(op, newOp.getValue());
  return success();
}

static LogicalResult foldAddPtrIntoStoreScalarOp(mlir::pto::StoreScalarOp op,
                                                 MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();

  Value base = op.getPtr();
  Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);
  bool foldedAddPtr = false;
  while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
    foldedAddPtr = true;
    Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
    totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                              : off;
    base = add.getOperand(0);
  }
  if (!foldedAddPtr)
    return success();

  rewriter.create<pto::StoreScalarOp>(loc, base, totalOffset, op.getValue());
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult eraseOrRejectRemainingAddPtrOps(func::FuncOp func) {
  SmallVector<Operation *, 8> addPtrs;
  func.walk([&](mlir::pto::AddPtrOp op) { addPtrs.push_back(op.getOperation()); });
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &op : addPtrs) {
      if (!op)
        continue;
      if (op->use_empty()) {
        op->erase();
        op = nullptr;
        changed = true;
      }
    }
  }
  for (Operation *op : addPtrs) {
    if (!op)
      continue;
    op->emitError(
        "addptr must feed make_tensor_view, initialize_l2g2l_pipe(gm_addr), "
        "or load/store_scalar for lowering");
    return failure();
  }
  return success();
}

static LogicalResult foldAddPtrIntoScalarOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::LoadScalarOp, 8> loadScalars;
  func.walk([&](mlir::pto::LoadScalarOp op) { loadScalars.push_back(op); });
  for (auto op : loadScalars)
    if (failed(foldAddPtrIntoLoadScalarOp(op, ctx)))
      return failure();

  SmallVector<mlir::pto::StoreScalarOp, 8> storeScalars;
  func.walk([&](mlir::pto::StoreScalarOp op) { storeScalars.push_back(op); });
  for (auto op : storeScalars)
    if (failed(foldAddPtrIntoStoreScalarOp(op, ctx)))
      return failure();

  return success();
}

static LogicalResult normalizeTAssignOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::TAssignOp, 8> tassignOps;
  func.walk([&](mlir::pto::TAssignOp op) { tassignOps.push_back(op); });
  for (auto op : tassignOps) {
    Type targetTy = op.getTile().getType();
    if (op.getResult().getType() == targetTy)
      continue;
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    auto normalized =
        rewriter.create<pto::TAssignOp>(op.getLoc(), targetTy, op.getTile(),
                                        op.getAddr());
    rewriter.replaceOp(op, normalized.getResult());
  }
  return success();
}

static LogicalResult foldAddPtrIntoPipeInitOps(func::FuncOp func,
                                               MLIRContext *ctx) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<mlir::pto::AddPtrOp, 8> candidates;
    func.walk([&](mlir::pto::AddPtrOp op) {
      bool eligible = !op->use_empty();
      for (Operation *user : op->getUsers()) {
        auto init = dyn_cast<mlir::pto::InitializeL2G2LPipeOp>(user);
        if (!init || init.getGmAddr() != op.getResult()) {
          eligible = false;
          break;
        }
      }
      if (eligible)
        candidates.push_back(op);
    });

    for (auto op : candidates) {
      IRRewriter rewriter(ctx);
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      Value base = op->getOperand(0);
      Value totalOffset = ensureIndex(rewriter, loc, op->getOperand(1), op);
      while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
        Value off = ensureIndex(rewriter, loc, add->getOperand(1), add);
        totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
        base = add->getOperand(0);
      }

      auto baseMrTy = dyn_cast<MemRefType>(base.getType());
      if (!baseMrTy || baseMrTy.getRank() != 1)
        continue;

      int64_t dyn = ShapedType::kDynamic;
      auto layout = StridedLayoutAttr::get(ctx, dyn, {dyn});
      auto targetTy = MemRefType::get({dyn}, baseMrTy.getElementType(), layout,
                                      baseMrTy.getMemorySpace());
      SmallVector<OpFoldResult, 1> sizes{rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult, 1> strides{rewriter.getIndexAttr(1)};
      auto rc = rewriter.create<memref::ReinterpretCastOp>(
          loc, targetTy, base, OpFoldResult(totalOffset), sizes, strides);
      rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
      rewriter.replaceOp(op, rc.getResult());
      changed = true;
    }
  }
  return success();
}

template <typename OpTy>
static LogicalResult rebuildCollectedOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<OpTy, 8> ops;
  func.walk([&](OpTy op) { ops.push_back(op); });
  for (OpTy op : ops) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Operation *cloned = rewriter.clone(*op.getOperation());
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
      continue;
    }
    rewriter.replaceOp(op, cloned->getResults());
  }
  return success();
}

#define PTO_REBUILD_OP(OP)                                                      \
  if (failed(rebuildCollectedOps<mlir::pto::OP>(func, ctx)))                    \
    return failure()

static LogicalResult rewriteLinearAlgebraOps(func::FuncOp func,
                                             MLIRContext *ctx) {
  PTO_REBUILD_OP(TLoadOp);
  PTO_REBUILD_OP(TStoreOp);
  PTO_REBUILD_OP(TTransOp);
  PTO_REBUILD_OP(TExpOp);
  PTO_REBUILD_OP(TMulOp);
  PTO_REBUILD_OP(TMulSOp);
  PTO_REBUILD_OP(TAddOp);
  PTO_REBUILD_OP(TMatmulOp);
  PTO_REBUILD_OP(TMatmulAccOp);
  PTO_REBUILD_OP(TMatmulBiasOp);
  PTO_REBUILD_OP(TMatmulMxOp);
  PTO_REBUILD_OP(TMatmulMxAccOp);
  PTO_REBUILD_OP(TMatmulMxBiasOp);
  PTO_REBUILD_OP(TGemvOp);
  PTO_REBUILD_OP(TGemvAccOp);
  PTO_REBUILD_OP(TGemvBiasOp);
  PTO_REBUILD_OP(TGemvMxOp);
  PTO_REBUILD_OP(TGemvMxAccOp);
  PTO_REBUILD_OP(TGemvMxBiasOp);
  PTO_REBUILD_OP(TMovOp);
  return success();
}

static LogicalResult rewriteVectorComputeOps(func::FuncOp func,
                                             MLIRContext *ctx) {
  PTO_REBUILD_OP(TAbsOp);
  PTO_REBUILD_OP(TAddCOp);
  PTO_REBUILD_OP(TAddSOp);
  PTO_REBUILD_OP(TAddSCOp);
  PTO_REBUILD_OP(TAndOp);
  PTO_REBUILD_OP(TConcatOp);
  PTO_REBUILD_OP(TAndSOp);
  PTO_REBUILD_OP(TCIOp);
  PTO_REBUILD_OP(TCmpOp);
  PTO_REBUILD_OP(TCmpSOp);
  PTO_REBUILD_OP(TColExpandOp);
  PTO_REBUILD_OP(TColMaxOp);
  PTO_REBUILD_OP(TColMinOp);
  PTO_REBUILD_OP(TColExpandMulOp);
  PTO_REBUILD_OP(TColExpandMaxOp);
  PTO_REBUILD_OP(TColExpandMinOp);
  PTO_REBUILD_OP(TColSumOp);
  PTO_REBUILD_OP(TCvtOp);
  PTO_REBUILD_OP(TDivOp);
  PTO_REBUILD_OP(TDivSOp);
  PTO_REBUILD_OP(TExpandsOp);
  PTO_REBUILD_OP(TExtractOp);
  PTO_REBUILD_OP(TFillPadOp);
  PTO_REBUILD_OP(TFillPadInplaceOp);
  PTO_REBUILD_OP(TSetValOp);
  PTO_REBUILD_OP(TGetValOp);
  PTO_REBUILD_OP(TGatherOp);
  PTO_REBUILD_OP(TGatherBOp);
  PTO_REBUILD_OP(TLogOp);
  PTO_REBUILD_OP(TLReluOp);
  PTO_REBUILD_OP(TMaxOp);
  PTO_REBUILD_OP(TMaxSOp);
  PTO_REBUILD_OP(TMinOp);
  PTO_REBUILD_OP(TMinSOp);
  PTO_REBUILD_OP(TMovFPOp);
  PTO_REBUILD_OP(TQuantOp);
  PTO_REBUILD_OP(TMrgSortOp);
  PTO_REBUILD_OP(TNegOp);
  PTO_REBUILD_OP(TNotOp);
  PTO_REBUILD_OP(TOrOp);
  PTO_REBUILD_OP(TOrSOp);
  PTO_REBUILD_OP(TPartAddOp);
  PTO_REBUILD_OP(TPartMulOp);
  PTO_REBUILD_OP(MGatherOp);
  PTO_REBUILD_OP(MScatterOp);
  PTO_REBUILD_OP(TPrintOp);
  return success();
}

static LogicalResult rewriteComputeOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteLinearAlgebraOps(func, ctx)))
    return failure();
  if (failed(rewriteVectorComputeOps(func, ctx)))
    return failure();
  return success();
}

static LogicalResult finalizeFunctionLowering(func::FuncOp func,
                                              MLIRContext *ctx) {
  if (failed(reconcileSCFIfResultTypes(func)))
    return failure();
  if (failed(markLoweredSetValidShapeOps(func, ctx)))
    return failure();
  return success();
}

static LogicalResult lowerSingleFunction(func::FuncOp func, MLIRContext *ctx) {
  rewriteFunctionSignature(func, ctx);
  if (failed(lowerAllocTileOps(func, ctx)) ||
      failed(lowerDeclareTileOps(func, ctx)) ||
      failed(normalizeTAssignOps(func, ctx)) ||
      failed(lowerMakeTensorViewOps(func, ctx)) ||
      failed(lowerTensorViewDimOps(func, ctx)) ||
      failed(lowerPartitionViewOps(func, ctx)) ||
      failed(lowerSubsetOps(func, ctx)) ||
      failed(lowerTileBufViewLikeOps(func, ctx)) ||
      failed(foldAddPtrIntoScalarOps(func, ctx)) ||
      failed(foldAddPtrIntoPipeInitOps(func, ctx)) ||
      failed(eraseOrRejectRemainingAddPtrOps(func)) ||
      failed(rewriteComputeOps(func, ctx)) ||
      failed(finalizeFunctionLowering(func, ctx)))
    return failure();
  return success();
}

#undef PTO_REBUILD_OP

static void buildPartitionViewSizes(IRRewriter &rewriter, Location loc,
                                    mlir::pto::PartitionViewOp op,
                                    SmallVector<int64_t> &staticSizes,
                                    SmallVector<OpFoldResult> &mixedSizes) {
  for (Value size : op.getSizes()) {
    IntegerAttr constAttr;
    bool isStatic = false;
    if (auto cOp = size.getDefiningOp<arith::ConstantIndexOp>()) {
      constAttr = rewriter.getIndexAttr(cOp.value());
      isStatic = true;
    } else if (auto cInt = size.getDefiningOp<arith::ConstantIntOp>()) {
      constAttr = rewriter.getIndexAttr(cInt.value());
      isStatic = true;
    }

    if (isStatic) {
      mixedSizes.push_back(constAttr);
      staticSizes.push_back(constAttr.getInt());
      continue;
    }
    mixedSizes.push_back(ensureIndex(rewriter, loc, size, op));
    staticSizes.push_back(ShapedType::kDynamic);
  }
}

static SmallVector<OpFoldResult>
buildPartitionViewOffsets(IRRewriter &rewriter, Location loc,
                          mlir::pto::PartitionViewOp op) {
  SmallVector<OpFoldResult> mixedOffsets;
  for (Value offset : op.getOffsets()) {
    IntegerAttr constAttr;
    bool isStatic = false;
    if (auto cOp = offset.getDefiningOp<arith::ConstantIndexOp>()) {
      constAttr = rewriter.getIndexAttr(cOp.value());
      isStatic = true;
    } else if (auto cInt = offset.getDefiningOp<arith::ConstantIntOp>()) {
      constAttr = rewriter.getIndexAttr(cInt.value());
      isStatic = true;
    }
    mixedOffsets.push_back(isStatic ? OpFoldResult(constAttr)
                                    : OpFoldResult(ensureIndex(rewriter, loc,
                                                               offset, op)));
  }
  return mixedOffsets;
}

static LogicalResult lowerSinglePartitionViewOp(mlir::pto::PartitionViewOp op,
                                                MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value src = op.getOperand(0);
  auto srcMrTy = dyn_cast<MemRefType>(src.getType());
  int64_t rank = srcMrTy.getRank();

  SmallVector<int64_t> staticSizes;
  SmallVector<OpFoldResult> mixedSizes;
  buildPartitionViewSizes(rewriter, loc, op, staticSizes, mixedSizes);
  SmallVector<OpFoldResult> mixedOffsets =
      buildPartitionViewOffsets(rewriter, loc, op);

  int64_t dyn = ShapedType::kDynamic;
  SmallVector<int64_t> dynStrides(rank, dyn);
  auto layout = StridedLayoutAttr::get(ctx, dyn, dynStrides);
  auto resTy = MemRefType::get(staticSizes, srcMrTy.getElementType(), layout,
                               srcMrTy.getMemorySpace());

  SmallVector<OpFoldResult> mixedStrides(rank, rewriter.getIndexAttr(1));
  auto sv = rewriter.create<memref::SubViewOp>(loc, resTy, src, mixedOffsets,
                                               mixedSizes, mixedStrides);
  if (Operation *srcDef = src.getDefiningOp()) {
    if (auto layoutAttr = srcDef->getAttrOfType<pto::LayoutAttr>("layout"))
      sv->setAttr("layout", layoutAttr);
  }
  rewriter.replaceOp(op, sv.getResult());
  return success();
}

static LogicalResult lowerPartitionViewOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::PartitionViewOp, 8> partitionViews;
  func.walk([&](mlir::pto::PartitionViewOp op) { partitionViews.push_back(op); });

  for (auto op : partitionViews)
    if (failed(lowerSinglePartitionViewOp(op, ctx)))
      return failure();
  return success();
}

static SmallVector<int64_t> getSubsetStaticSizes(mlir::pto::SubsetOp op) {
  SmallVector<int64_t> staticSizes;
  for (Attribute attr : op.getSizes())
    staticSizes.push_back(cast<IntegerAttr>(attr).getInt());
  return staticSizes;
}

static SmallVector<OpFoldResult> getSubsetMixedSizes(IRRewriter &rewriter,
                                                     ArrayRef<int64_t> staticSizes) {
  SmallVector<OpFoldResult> mixedSizes;
  for (int64_t size : staticSizes)
    mixedSizes.push_back(rewriter.getIndexAttr(size));
  return mixedSizes;
}

static SmallVector<OpFoldResult>
buildSubsetMixedOffsets(IRRewriter &rewriter, Location loc,
                        mlir::pto::SubsetOp op) {
  SmallVector<OpFoldResult> mixedOffsets;
  for (Value offset : op.getOffsets()) {
    IntegerAttr constAttr;
    bool isStatic = false;
    if (auto cOp = offset.getDefiningOp<arith::ConstantIndexOp>()) {
      constAttr = rewriter.getIndexAttr(cOp.value());
      isStatic = true;
    } else if (auto cInt = offset.getDefiningOp<arith::ConstantIntOp>()) {
      constAttr = rewriter.getIndexAttr(cInt.value());
      isStatic = true;
    }
    mixedOffsets.push_back(isStatic ? OpFoldResult(constAttr)
                                    : OpFoldResult(ensureIndex(rewriter, loc,
                                                               offset, op)));
  }
  return mixedOffsets;
}

static LogicalResult validateBoxedSubsetAlignment(mlir::pto::SubsetOp op,
                                                  ArrayRef<int64_t> staticSizes,
                                                  const TileLayoutInfo &layoutInfo,
                                                  int64_t &off0, int64_t &off1,
                                                  bool &off0Const,
                                                  bool &off1Const) {
  if (staticSizes.size() != 2 || op.getOffsets().size() != 2) {
    op.emitError("boxed layout subset expects 2D sizes/offsets");
    return failure();
  }
  if (!checkMultipleOf(op, staticSizes[0], layoutInfo.innerRows, "row size") ||
      !checkMultipleOf(op, staticSizes[1], layoutInfo.innerCols, "col size")) {
    return failure();
  }

  off0Const = getConstIndexValue(op.getOffsets()[0], off0);
  off1Const = getConstIndexValue(op.getOffsets()[1], off1);
  if (off0Const &&
      !checkMultipleOf(op, off0, layoutInfo.innerRows, "row offset")) {
    return failure();
  }
  if (off1Const &&
      !checkMultipleOf(op, off1, layoutInfo.innerCols, "col offset")) {
    return failure();
  }
  return success();
}

static LogicalResult validateBoxedSubsetFullAxis(mlir::pto::SubsetOp op,
                                                 MemRefType srcMrTy,
                                                 pto::TileBufConfigAttr configAttr,
                                                 ArrayRef<int64_t> staticSizes,
                                                 int64_t off0, int64_t off1,
                                                 bool off0Const,
                                                 bool off1Const) {
  int32_t bl = 0;
  (void)readBLayoutI32(configAttr.getBLayout(), bl);
  auto srcShape = srcMrTy.getShape();
  if (srcShape.size() != 2)
    return success();
  if (bl == 0) {
    if (staticSizes[1] != srcShape[1]) {
      op.emitError("boxed RowMajor subset must keep full cols");
      return failure();
    }
    if (!off1Const || off1 != 0) {
      op.emitError("boxed RowMajor subset requires static col offset = 0");
      return failure();
    }
    return success();
  }

  if (staticSizes[0] != srcShape[0]) {
    op.emitError("boxed ColMajor subset must keep full rows");
    return failure();
  }
  if (!off0Const || off0 != 0) {
    op.emitError("boxed ColMajor subset requires static row offset = 0");
    return failure();
  }
  return success();
}

static LogicalResult validateBoxedSubsetLayout(mlir::pto::SubsetOp op,
                                               MemRefType srcMrTy,
                                               pto::TileBufConfigAttr configAttr,
                                               ArrayRef<int64_t> staticSizes,
                                               const TileLayoutInfo &layoutInfo) {
  if (!layoutInfo.boxed)
    return success();

  int64_t off0 = 0;
  int64_t off1 = 0;
  bool off0Const = false;
  bool off1Const = false;
  if (failed(validateBoxedSubsetAlignment(op, staticSizes, layoutInfo, off0,
                                          off1, off0Const, off1Const))) {
    return failure();
  }
  return validateBoxedSubsetFullAxis(op, srcMrTy, configAttr, staticSizes, off0,
                                     off1, off0Const, off1Const);
}

static MemRefType buildSubsetResultMemRefType(MLIRContext *ctx,
                                              MemRefType srcMrTy,
                                              ArrayRef<int64_t> staticSizes) {
  SmallVector<int64_t> srcStrides;
  int64_t srcOffset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(srcMrTy, srcStrides, srcOffset)))
    srcStrides = computeCompactStrides(srcMrTy.getShape());

  auto resultLayout =
      StridedLayoutAttr::get(ctx, ShapedType::kDynamic, srcStrides);
  return MemRefType::get(staticSizes, srcMrTy.getElementType(), resultLayout,
                         srcMrTy.getMemorySpace());
}

static void computeSubsetBoundValidDims(IRRewriter &rewriter, Location loc,
                                        mlir::pto::SubsetOp op,
                                        ArrayRef<int64_t> staticSizes, Value src,
                                        Value &vRow, Value &vCol) {
  Value parentVRow;
  Value parentVCol;
  lookupValidDims(src, parentVRow, parentVCol);
  if (!staticSizes.empty()) {
    vRow = computeSubsetValidDim(rewriter, loc, parentVRow, op.getOffsets()[0],
                                 staticSizes[0], op);
  }
  if (staticSizes.size() > 1) {
    vCol = computeSubsetValidDim(rewriter, loc, parentVCol, op.getOffsets()[1],
                                 staticSizes[1], op);
  }
}

static Value bindSubsetSubviewResult(IRRewriter &rewriter, Location loc,
                                     mlir::pto::SubsetOp op,
                                     MemRefType resultMemRefType, Value subView,
                                     pto::TileBufConfigAttr configAttr,
                                     ArrayRef<int64_t> staticSizes, Value src,
                                     mlir::pto::TileBufType resultTileTy,
                                     MLIRContext *ctx) {
  Value vRow;
  Value vCol;
  computeSubsetBoundValidDims(rewriter, loc, op, staticSizes, src, vRow, vCol);
  auto bindOp = rewriter.create<pto::BindTileOp>(
      loc, resultMemRefType, subView, vRow ? vRow : Value(),
      vCol ? vCol : Value(), configAttr);
  markForceDynamicValidShape(bindOp,
                             resultTileTy && resultTileTy.hasDynamicValid(),
                             ctx);
  return bindOp.getResult();
}

static LogicalResult lowerSingleSubsetOp(mlir::pto::SubsetOp op,
                                         MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  auto resultTileTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
  Value src = op->getOperand(0);
  auto srcMrTy = dyn_cast<MemRefType>(src.getType());
  if (!srcMrTy) {
    op.emitError("pto.subset source must be lowered to memref first");
    return failure();
  }

  SmallVector<int64_t> staticSizes = getSubsetStaticSizes(op);
  SmallVector<OpFoldResult> mixedSizes = getSubsetMixedSizes(rewriter, staticSizes);
  SmallVector<OpFoldResult> mixedOffsets =
      buildSubsetMixedOffsets(rewriter, loc, op);

  auto configAttr = lookupConfig(src);
  if (!configAttr)
    configAttr = pto::TileBufConfigAttr::getDefault(ctx);

  TileLayoutInfo layoutInfo;
  if (!computeTileLayoutInfo(configAttr, srcMrTy.getElementType(),
                             srcMrTy.getShape(), layoutInfo)) {
    op.emitError("unsupported tile layout for pto.subset");
    return failure();
  }
  if (failed(validateBoxedSubsetLayout(op, srcMrTy, configAttr, staticSizes,
                                       layoutInfo))) {
    return failure();
  }

  auto resultMemRefType = buildSubsetResultMemRefType(ctx, srcMrTy, staticSizes);
  SmallVector<OpFoldResult> mixedStrides(staticSizes.size(),
                                         rewriter.getIndexAttr(1));
  auto sv = rewriter.create<memref::SubViewOp>(loc, resultMemRefType, src,
                                               mixedOffsets, mixedSizes,
                                               mixedStrides);
  rewriter.replaceOp(op, bindSubsetSubviewResult(
                             rewriter, loc, op, resultMemRefType, sv.getResult(),
                             configAttr, staticSizes, src, resultTileTy, ctx));
  return success();
}

static LogicalResult lowerSubsetOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::SubsetOp, 8> subsets;
  func.walk([&](mlir::pto::SubsetOp op) { subsets.push_back(op); });

  for (auto op : subsets)
    if (failed(lowerSingleSubsetOp(op, ctx)))
      return failure();
  return success();
}

static Value buildTileBufViewLikeValue(Operation *anchorOp, Value src,
                                       mlir::pto::TileBufType tbTy,
                                       StringRef viewSemantics,
                                       MLIRContext *ctx) {
  Location loc = anchorOp->getLoc();
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(anchorOp);

  auto srcMrTy = dyn_cast<MemRefType>(src.getType());
  if (!srcMrTy) {
    anchorOp->emitError("tile_buf view op src must be lowered to memref first");
    return Value();
  }

  auto targetType = dyn_cast<MemRefType>(convertPTOTypeToMemRef(tbTy));
  if (!targetType) {
    anchorOp->emitError("failed to convert tile_buf type to memref type");
    return Value();
  }
  for (int64_t dim : targetType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      anchorOp->emitError("dynamic shapes are not supported for tile_buf view ops");
      return Value();
    }
  }

  Value parentVRow;
  Value parentVCol;
  lookupValidDims(src, parentVRow, parentVCol);
  Value vRow = parentVRow;
  Value vCol = parentVCol;
  materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

  auto configAttr = tbTy.getConfigAttr();
  if (!configAttr)
    configAttr = pto::TileBufConfigAttr::getDefault(ctx);

  auto bindOp = rewriter.create<pto::BindTileOp>(
      loc, targetType, src, vRow ? vRow : Value(), vCol ? vCol : Value(),
      configAttr);
  markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
  if (!viewSemantics.empty())
    bindOp->setAttr("pto.view_semantics", rewriter.getStringAttr(viewSemantics));
  return bindOp.getResult();
}

static LogicalResult lowerTileBufViewLikeOps(func::FuncOp func, MLIRContext *ctx) {
  SmallVector<mlir::pto::TReshapeOp, 8> reshapes;
  func.walk([&](mlir::pto::TReshapeOp op) { reshapes.push_back(op); });
  for (auto op : reshapes) {
    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy) {
      op.emitError("treshape result must be tile_buf type");
      return failure();
    }
    Value lowered = buildTileBufViewLikeValue(op, op->getOperand(0), tbTy,
                                              "treshape", ctx);
    if (!lowered)
      return failure();
    IRRewriter rewriter(ctx);
    rewriter.replaceOp(op, lowered);
  }

  SmallVector<mlir::pto::BitcastOp, 8> bitcasts;
  func.walk([&](mlir::pto::BitcastOp op) { bitcasts.push_back(op); });
  for (auto op : bitcasts) {
    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy) {
      op.emitError("bitcast result must be tile_buf type");
      return failure();
    }
    Value lowered = buildTileBufViewLikeValue(op, op->getOperand(0), tbTy,
                                              "bitcast", ctx);
    if (!lowered)
      return failure();
    IRRewriter rewriter(ctx);
    rewriter.replaceOp(op, lowered);
  }
  return success();
}

// =============================================================================
// The Pass Implementation
// =============================================================================

struct PTOViewToMemrefPass
    : public PassWrapper<PTOViewToMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOViewToMemrefPass)

  StringRef getArgument() const final { return "pto-view-to-memref"; }
  StringRef getDescription() const final {
    return "Lower PTO views to memref with Metadata Binding";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::pto::PTODialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (failed(lowerSingleFunction(func, ctx))) {
        signalPassFailure();
        return;
      }
    }

    dumpPretty(mod.getOperation(), llvm::errs());
  }
};

} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<PTOViewToMemrefPass>();
}

} // namespace pto
} // namespace mlir
