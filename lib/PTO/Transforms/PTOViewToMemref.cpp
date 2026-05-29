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
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTOViewToMemrefInternal.h"

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

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVIEWTOMEMREF
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "Utils.h" // 假设包含一些通用的工具函数

#include <algorithm>
#include <functional>
#include <limits>

#define DEBUG_TYPE "pto-view-to-memref"

using namespace mlir;

namespace mlir {
namespace pto {

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

namespace {

static void markForceDynamicValidShape(Operation *op, bool force,
                                       MLIRContext *ctx);

static Type convertPTOTypeToMemRef(Type t);

constexpr size_t kTileRank2D = 2;
constexpr size_t kRowDimensionIndex = 0;
constexpr size_t kColumnDimensionIndex = 1;
constexpr unsigned kShapeVectorInlineCapacity = 4;
constexpr unsigned kOperationVectorInlineCapacity = 8;

constexpr int64_t kElementBytes1 = 1;
constexpr int64_t kElementBytes2 = 2;
constexpr int64_t kElementBytes4 = 4;
constexpr int64_t kElementBytes8 = 8;
constexpr int64_t kElementBytes16 = 16;
constexpr int64_t kElementBytes32 = 32;

constexpr int64_t kInnerExtent1 = 1;
constexpr int64_t kInnerExtent2 = 2;
constexpr int64_t kInnerExtent4 = 4;
constexpr int64_t kInnerExtent8 = 8;
constexpr int64_t kInnerExtent16 = 16;
constexpr int64_t kInnerExtent32 = 32;

constexpr int32_t kFractalSize32 = 32;
constexpr int32_t kFractalSize512 = 512;
constexpr int32_t kFractalSize1024 = 1024;

constexpr int32_t kBLayoutColMajor =
    static_cast<int32_t>(BLayout::ColMajor);
constexpr int32_t kSLayoutNoneBox =
    static_cast<int32_t>(SLayout::NoneBox);
constexpr int32_t kSLayoutRowMajor =
    static_cast<int32_t>(SLayout::RowMajor);
constexpr int32_t kSLayoutColMajor =
    static_cast<int32_t>(SLayout::ColMajor);
constexpr int32_t kCompactModeRowPlusOne =
    static_cast<int32_t>(CompactMode::RowPlusOne);

template <typename T>
using SmallInlineVector = SmallVector<T, kShapeVectorInlineCapacity>;

template <typename T>
using DefaultInlineVector = SmallVector<T, kOperationVectorInlineCapacity>;

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
  int32_t fractalSize = kFractalSize512;
  int32_t compactMode = 0;
};

static int64_t getElemBytes(Type elemTy) {
  unsigned bytes = getPTOStorageElemByteSize(elemTy);
  return bytes == 0 ? -1 : static_cast<int64_t>(bytes);
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
  case kElementBytes1:
    extent = kInnerExtent32;
    return true;
  case kElementBytes2:
    extent = kInnerExtent16;
    return true;
  case kElementBytes4:
    extent = kInnerExtent8;
    return true;
  case kElementBytes8:
    extent = kInnerExtent4;
    return true;
  case kElementBytes16:
    extent = kInnerExtent2;
    return true;
  case kElementBytes32:
    extent = kInnerExtent1;
    return true;
  default:
    return false;
  }
}

static bool computeBoxInnerShape(const TileLayoutConfig &config, Type elemTy,
                                 TileLayoutInfo &info) {
  info.boxed = config.sLayout != kSLayoutNoneBox;
  if (!info.boxed) {
    info.innerRows = kInnerExtent1;
    info.innerCols = kInnerExtent1;
    return true;
  }

  int64_t elemBytes = getElemBytes(elemTy);
  if (elemBytes <= 0)
    return false;

  switch (config.fractalSize) {
  case kFractalSize1024:
    info.innerRows = kInnerExtent16;
    info.innerCols = kInnerExtent16;
    return true;
  case kFractalSize32:
    info.innerRows = kInnerExtent16;
    info.innerCols = kInnerExtent2;
    return true;
  case kFractalSize512:
    if (config.sLayout == kSLayoutRowMajor) {
      info.innerRows = kInnerExtent16;
      return getFractal512InnerExtent(elemBytes, info.innerCols);
    }
    if (config.sLayout == kSLayoutColMajor) {
      if (!getFractal512InnerExtent(elemBytes, info.innerRows))
        return false;
      info.innerCols = kInnerExtent16;
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
    if (config.compactMode == kCompactModeRowPlusOne)
      return majorStride + kInnerExtent1;
    return majorStride;
  };
  if (!info.boxed) {
    if (config.bLayout == kBLayoutColMajor) {
      info.rowStride = kInnerExtent1;
      info.colStride = applyCompactToMajorStride(rows);
      return true;
    }
    info.rowStride = applyCompactToMajorStride(cols);
    info.colStride = kInnerExtent1;
    return true;
  }

  if (config.bLayout == kBLayoutColMajor) {
    if (config.sLayout != kSLayoutRowMajor)
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
  if (shape.size() != kTileRank2D ||
      llvm::is_contained(shape, ShapedType::kDynamic))
    return false;

  TileLayoutConfig config = getTileLayoutConfig(cfg);
  return computeBoxInnerShape(config, elemTy, info) &&
         computeTilePointerStrides(config, shape, info);
}

static void collectAffineAddTerms(AffineExpr root,
                                  SmallVectorImpl<AffineExpr> &terms) {
  SmallInlineVector<AffineExpr> pending{root};
  while (!pending.empty()) {
    AffineExpr current = pending.pop_back_val();
    auto addExpr = llvm::dyn_cast<AffineBinaryOpExpr>(current);
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
  if (auto dim = llvm::dyn_cast<AffineDimExpr>(expr)) {
    strides[dim.getPosition()] = 1;
    return true;
  }

  auto mulExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr);
  if (!mulExpr || mulExpr.getKind() != AffineExprKind::Mul)
    return false;

  auto assignStride = [&](AffineExpr dimExpr,
                          AffineExpr constantExpr) -> bool {
    auto dim = llvm::dyn_cast<AffineDimExpr>(dimExpr);
    auto constant = llvm::dyn_cast<AffineConstantExpr>(constantExpr);
    if (!dim || !constant)
      return false;
    strides[dim.getPosition()] = constant.getValue();
    return true;
  };
  return assignStride(mulExpr.getLHS(), mulExpr.getRHS()) ||
         assignStride(mulExpr.getRHS(), mulExpr.getLHS());
}

[[maybe_unused]] static void decomposeStridedLayout(AffineMap map,
                                   SmallVectorImpl<int64_t> &strides) {
  strides.assign(map.getNumDims(), 0);
  if (map.getNumResults() != 1)
    return;

  SmallInlineVector<AffineExpr> terms;
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
  if (!validShape.empty() && validShape[kRowDimensionIndex] >= 0)
    vRow = makeIndexConstant(rewriter, loc, validShape[kRowDimensionIndex]);
  if (validShape.size() >= kTileRank2D &&
      validShape[kColumnDimensionIndex] >= 0)
    vCol = makeIndexConstant(rewriter, loc, validShape[kColumnDimensionIndex]);
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

static bool tryGetIndexAttrFromValue(IRRewriter &rewriter, Value v,
                                     IntegerAttr &constAttr) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    constAttr = rewriter.getIndexAttr(cOp.value());
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    constAttr = rewriter.getIndexAttr(cInt.value());
    return true;
  }
  return false;
}

static void appendMixedIndex(IRRewriter &rewriter, Location loc, Value v,
                             Operation *anchorOp,
                             SmallVectorImpl<OpFoldResult> &mixedVals) {
  IntegerAttr constAttr;
  if (tryGetIndexAttrFromValue(rewriter, v, constAttr)) {
    mixedVals.push_back(constAttr);
    return;
  }
  mixedVals.push_back(ensureIndex(rewriter, loc, v, anchorOp));
}

static bool foldAddPtrChainIntoOffset(IRRewriter &rewriter, Location loc,
                                      Value &base, Value &totalOffset) {
  bool folded = false;
  while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
    folded = true;
    Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
    totalOffset =
        totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off) : off;
    base = add.getOperand(0);
  }
  return folded;
}

static Value clampSubViewValidDim(IRRewriter &rewriter, Location loc,
                                  Value explicitValid, int64_t size,
                                  Operation *anchorOp) {
  Value sizeVal = rewriter.create<arith::ConstantIndexOp>(loc, size);
  if (!explicitValid)
    return sizeVal;

  int64_t cst = 0;
  if (getConstIndexValue(explicitValid, cst))
    return rewriter.create<arith::ConstantIndexOp>(loc, std::min(cst, size));

  Value v = ensureIndex(rewriter, loc, explicitValid, anchorOp);
  Value lt = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, v,
                                            sizeVal);
  return rewriter.create<arith::SelectOp>(loc, lt, v, sizeVal);
}

[[maybe_unused]] static void dumpPretty(Operation *op, llvm::raw_ostream &os) {
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
  TileLayoutInfo info;
  if (computeTileLayoutInfo(tbTy.getConfigAttr(), tbTy.getElementType(),
                            tbTy.getShape(), info)) {
    return {info.rowStride, info.colStride};
  }
  return computeCompactStrides(tbTy.getShape());
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
  if (auto tvTy = dyn_cast<mlir::pto::TensorViewType>(t))
    return MemRefType::get(tvTy.getShape(), tvTy.getElementType(),
                           MemRefLayoutAttrInterface(), Attribute());
  if (auto partTy = dyn_cast<mlir::pto::PartitionTensorViewType>(t))
    return MemRefType::get(partTy.getShape(), partTy.getElementType(),
                           MemRefLayoutAttrInterface(), Attribute());
  // 其他类型透传
  return t;
}

// Ensure scf.if result types follow the rewritten yield operand types.
// PTOViewToMemref rewrites tile values to memref in branch bodies, but scf.if
// result types are not auto-updated by those op-local rewrites.
static LogicalResult reconcileSCFIfResultTypes(func::FuncOp func) {
  DefaultInlineVector<scf::IfOp> ifOps;
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

static LogicalResult reconcileSCFForResultTypes(func::FuncOp func) {
  DefaultInlineVector<scf::ForOp> forOps;
  func.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  for (scf::ForOp forOp : forOps) {
    if (forOp.getNumResults() == 0)
      continue;

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield) {
      forOp.emitError("result-bearing scf.for must end with scf.yield");
      return failure();
    }

    if (yield.getNumOperands() != forOp.getNumResults() ||
        forOp.getInitArgs().size() != forOp.getNumResults()) {
      forOp.emitError("scf.for result count does not match iter/yield values");
      return failure();
    }

    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      Type initTy = forOp.getInitArgs()[i].getType();
      Type yieldTy = yield.getOperand(i).getType();
      if (initTy != yieldTy) {
        forOp.emitError() << "scf.for init/yield type mismatch at result #" << i
                          << ": init=" << initTy << ", yield=" << yieldTy;
        return failure();
      }

      BlockArgument iterArg = forOp.getRegionIterArg(i);
      if (iterArg.getType() != initTy)
        iterArg.setType(initTy);
      if (forOp.getResult(i).getType() != initTy)
        forOp.getResult(i).setType(initTy);
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

[[maybe_unused]] static void rewriteFunctionSignature(func::FuncOp func, MLIRContext *ctx) {
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

[[maybe_unused]] static LogicalResult lowerAllocTileOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::AllocTileOp> allocTiles;
  func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

  for (auto op : allocTiles) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    if (!tbTy)
      continue;

    SmallInlineVector<int64_t> shape(tbTy.getShape().begin(),
                                  tbTy.getShape().end());
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
      continue;
    }

    auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides);
    auto allocType =
        MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());
    Value alloc = rewriter.create<memref::AllocOp>(loc, allocType);
    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, targetType, alloc, vRow ? vRow : Value(), vCol ? vCol : Value(),
        configAttr);
    markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);
    rewriter.replaceOp(op, bindOp.getResult());
  }
  return success();
}

[[maybe_unused]] static LogicalResult lowerDeclareTileOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::DeclareTileOp> declaredTiles;
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

static Value castIndexToI64(IRRewriter &rewriter, Location loc, Value value) {
  Type i64Ty = rewriter.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  return rewriter.create<arith::IndexCastOp>(loc, i64Ty, value).getResult();
}

static FailureOr<Value>
materializePtrToIntAddPtrAddress(IRRewriter &rewriter, Location loc,
                                 mlir::pto::PtrToIntOp anchor, Value source) {
  SmallVector<mlir::pto::AddPtrOp, 4> addPtrChain;
  Value base = source;
  while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
    addPtrChain.push_back(add);
    base = add.getOperand(0);
  }

  if (addPtrChain.empty())
    return failure();

  auto baseMemTy = dyn_cast<MemRefType>(base.getType());
  if (!baseMemTy) {
    anchor.emitOpError(
        "pto.addptr source base could not be lowered to a GM memref");
    return failure();
  }

  Value byteAddress = rewriter.create<mlir::pto::PtrToIntOp>(
      loc, rewriter.getI64Type(), base);
  for (auto add : addPtrChain) {
    auto addPtrTy = dyn_cast<mlir::pto::PtrType>(add.getResult().getType());
    if (!addPtrTy) {
      anchor.emitOpError("requires pto.addptr source to have !pto.ptr result "
                         "type before byte-address lowering");
      return failure();
    }

    unsigned elemBytes =
        mlir::pto::getPTOStorageElemByteSize(addPtrTy.getElementType());
    if (elemBytes == 0) {
      anchor.emitOpError("cannot lower pto.addptr source with unknown element "
                         "byte size to a byte address");
      return failure();
    }

    Value byteOffset = castIndexToI64(rewriter, loc, add.getOffset());
    if (elemBytes != 1) {
      Value elemBytesValue =
          rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
      byteOffset =
          rewriter.create<arith::MulIOp>(loc, byteOffset, elemBytesValue)
              .getResult();
    }
    byteAddress =
        rewriter.create<arith::AddIOp>(loc, byteAddress, byteOffset).getResult();
  }

  return byteAddress;
}

static LogicalResult lowerIntToPtrOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::IntToPtrOp> intToPtrs;
  func.walk([&](mlir::pto::IntToPtrOp op) { intToPtrs.push_back(op); });

  for (auto op : intToPtrs) {
    if (!isa<mlir::pto::PtrType>(op.getResult().getType()))
      continue;

    auto targetTy =
        dyn_cast<MemRefType>(convertPTOTypeToMemRef(op.getResult().getType()));
    if (!targetTy) {
      op.emitError("failed to convert inttoptr result to memref type");
      return failure();
    }

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    auto lowered =
        rewriter.create<mlir::pto::IntToPtrOp>(op.getLoc(), targetTy,
                                               op.getAddr());
    lowered->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, lowered.getResult());
  }

  return success();
}

static LogicalResult lowerPtrToIntOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::PtrToIntOp> ptrToInts;
  func.walk([&](mlir::pto::PtrToIntOp op) { ptrToInts.push_back(op); });

  for (auto op : ptrToInts) {
    Value source = op.getPtr();
    if (source.getDefiningOp<mlir::pto::AddPtrOp>()) {
      IRRewriter rewriter(ctx);
      rewriter.setInsertionPoint(op);
      FailureOr<Value> byteAddress =
          materializePtrToIntAddPtrAddress(rewriter, op.getLoc(), op, source);
      if (failed(byteAddress))
        return failure();
      rewriter.replaceOp(op, *byteAddress);
      continue;
    }

    if (isa<mlir::pto::PtrType>(source.getType()))
      continue;
  }

  DefaultInlineVector<mlir::pto::PtrToIntOp> remaining;
  func.walk([&](mlir::pto::PtrToIntOp op) {
    if (isa<mlir::pto::PtrType>(op.getPtr().getType()))
      remaining.push_back(op);
  });
  for (auto op : remaining) {
    op.emitError("ptrtoint source could not be lowered to a GM memref");
    return failure();
  }

  return success();
}

[[maybe_unused]] static LogicalResult lowerMakeTensorViewOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::MakeTensorViewOp> makeViews;
  func.walk([&](mlir::pto::MakeTensorViewOp op) { makeViews.push_back(op); });

  for (auto op : makeViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value baseBuf = op.getOperand(0);
    OpFoldResult off0 = rewriter.getIndexAttr(0);
    bool foldedAddPtr = false;
    {
      Value cur = baseBuf;
      Value totalOffset;
      while (auto add = cur.getDefiningOp<mlir::pto::AddPtrOp>()) {
        foldedAddPtr = true;
        Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
        totalOffset = totalOffset ? rewriter.create<arith::AddIOp>(loc, totalOffset, off)
                                  : off;
        cur = add.getOperand(0);
      }
      if (cur != baseBuf) {
        baseBuf = cur;
        off0 = totalOffset ? OpFoldResult(totalOffset) : off0;
      }
    }

    auto baseMr = dyn_cast<BaseMemRefType>(baseBuf.getType());
    if (!baseMr) {
      op.emitError("make_tensor_view base must be memref");
      return failure();
    }

    size_t rank = op.getShape().size();
    int64_t dyn = ShapedType::kDynamic;
    SmallVector<int64_t> dynStrides(rank, dyn);
    auto layout =
        StridedLayoutAttr::get(ctx, /*offset=*/dyn, /*strides=*/dynStrides);
    SmallVector<int64_t> dynShape(rank, dyn);
    auto mrTy = MemRefType::get(dynShape, baseMr.getElementType(), layout,
                                baseMr.getMemorySpace());

    SmallInlineVector<OpFoldResult> sizes;
    for (Value value : op.getShape())
      sizes.push_back(ensureIndex(rewriter, loc, value, op));
    SmallInlineVector<OpFoldResult> strides;
    for (Value value : op.getStrides())
      strides.push_back(ensureIndex(rewriter, loc, value, op));

    auto rc = rewriter.create<memref::ReinterpretCastOp>(loc, mrTy, baseBuf, off0,
                                                         sizes, strides);
    if (foldedAddPtr)
      rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
    if (auto layoutAttr = op.getLayoutAttr())
      rc->setAttr("layout", layoutAttr);
    rewriter.replaceOp(op, rc.getResult());
  }
  return success();
}

[[maybe_unused]] static LogicalResult lowerTensorViewDimOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::GetTensorViewDimOp> tvDims;
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

[[maybe_unused]] static LogicalResult foldAddPtrIntoScalarOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::LoadScalarOp> loadScalars;
  func.walk([&](mlir::pto::LoadScalarOp op) { loadScalars.push_back(op); });
  for (auto op : loadScalars) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value base = op.getPtr();
    Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);
    bool foldedAddPtr = foldAddPtrChainIntoOffset(rewriter, loc, base, totalOffset);
    if (foldedAddPtr) {
      auto newOp =
          rewriter.create<pto::LoadScalarOp>(loc, op.getValue().getType(), base,
                                             totalOffset);
      rewriter.replaceOp(op, newOp.getValue());
    }
  }

  DefaultInlineVector<mlir::pto::StoreScalarOp> storeScalars;
  func.walk([&](mlir::pto::StoreScalarOp op) { storeScalars.push_back(op); });
  for (auto op : storeScalars) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();

    Value base = op.getPtr();
    Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);
    bool foldedAddPtr = foldAddPtrChainIntoOffset(rewriter, loc, base, totalOffset);
    if (foldedAddPtr) {
      rewriter.create<pto::StoreScalarOp>(loc, base, totalOffset, op.getValue());
      rewriter.eraseOp(op);
    }
  }

  DefaultInlineVector<Operation *> addPtrs;
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
        "addptr must feed make_tensor_view or load/store_scalar for lowering");
    return failure();
  }
  return success();
}

static LogicalResult lowerPartitionViewOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::PartitionViewOp> partitionViews;
  func.walk([&](mlir::pto::PartitionViewOp op) { partitionViews.push_back(op); });

  for (auto op : partitionViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    Value src = op.getOperand(0);
    auto srcMrTy = dyn_cast<MemRefType>(src.getType());
    if (!srcMrTy)
      continue;
    int64_t rank = srcMrTy.getRank();

    SmallVector<int64_t> staticSizes;
    SmallVector<OpFoldResult> mixedSizes;
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
      } else {
        mixedSizes.push_back(ensureIndex(rewriter, loc, size, op));
        staticSizes.push_back(ShapedType::kDynamic);
      }
    }

    SmallVector<OpFoldResult> mixedOffsets;
    for (Value offset : op.getOffsets()) {
      appendMixedIndex(rewriter, loc, offset, op, mixedOffsets);
    }

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
  }
  return success();
}

static LogicalResult lowerSubViewOps(func::FuncOp func, MLIRContext *ctx) {
  DefaultInlineVector<mlir::pto::SubViewOp> subViews;
  func.walk([&](mlir::pto::SubViewOp op) { subViews.push_back(op); });

  for (auto op : subViews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto resultTileTy =
        dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
    Value src = op->getOperand(0);
    auto srcMrTy = dyn_cast<MemRefType>(src.getType());
    if (!srcMrTy) {
      op.emitError("pto.subview source must be lowered to memref first");
      return failure();
    }

    ArrayAttr sizeAttr = op.getSizes();
    SmallVector<int64_t> staticSizes;
    SmallVector<OpFoldResult> mixedSizes;
    for (Attribute attr : sizeAttr) {
      int64_t size = cast<IntegerAttr>(attr).getInt();
      staticSizes.push_back(size);
      mixedSizes.push_back(rewriter.getIndexAttr(size));
    }

    SmallVector<OpFoldResult> mixedOffsets;
    for (Value offset : op.getOffsets()) {
      appendMixedIndex(rewriter, loc, offset, op, mixedOffsets);
    }

    auto configAttr = lookupConfig(src);
    if (!configAttr)
      configAttr = pto::TileBufConfigAttr::getDefault(ctx);

    TileLayoutInfo layoutInfo;
    if (!computeTileLayoutInfo(configAttr, srcMrTy.getElementType(),
                               srcMrTy.getShape(), layoutInfo)) {
      op.emitError("unsupported tile layout for pto.subview");
      return failure();
    }

    if (layoutInfo.boxed) {
      if (staticSizes.size() != kTileRank2D ||
          op.getOffsets().size() != kTileRank2D) {
        op.emitError("boxed layout subview expects 2D sizes/offsets");
        return failure();
      }
      if (!checkMultipleOf(op, staticSizes[0], layoutInfo.innerRows, "row size") ||
          !checkMultipleOf(op, staticSizes[1], layoutInfo.innerCols, "col size")) {
        return failure();
      }

      int64_t off0 = 0;
      int64_t off1 = 0;
      bool off0Const = getConstIndexValue(op.getOffsets()[0], off0);
      bool off1Const = getConstIndexValue(op.getOffsets()[1], off1);
      if (off0Const &&
          !checkMultipleOf(op, off0, layoutInfo.innerRows, "row offset")) {
        return failure();
      }
      if (off1Const &&
          !checkMultipleOf(op, off1, layoutInfo.innerCols, "col offset")) {
        return failure();
      }

      int32_t bl = 0;
      (void)readBLayoutI32(configAttr.getBLayout(), bl);
      auto srcShape = srcMrTy.getShape();
      if (srcShape.size() == 2) {
        if (bl == 0) {
          if (staticSizes[1] != srcShape[1]) {
            op.emitError("boxed RowMajor subview must keep full cols");
            return failure();
          }
          if (!off1Const || off1 != 0) {
            op.emitError("boxed RowMajor subview requires static col offset = 0");
            return failure();
          }
        } else {
          if (staticSizes[0] != srcShape[0]) {
            op.emitError("boxed ColMajor subview must keep full rows");
            return failure();
          }
          if (!off0Const || off0 != 0) {
            op.emitError("boxed ColMajor subview requires static row offset = 0");
            return failure();
          }
        }
      }
    }

    SmallVector<int64_t> srcStrides;
    int64_t srcOffset = ShapedType::kDynamic;
    if (failed(getStridesAndOffset(srcMrTy, srcStrides, srcOffset)))
      srcStrides = computeCompactStrides(srcMrTy.getShape());

    // Keep parent physical shape + strides for bound tile semantics.
    auto resultLayout =
        StridedLayoutAttr::get(ctx, ShapedType::kDynamic, srcStrides);
    auto parentShape = srcMrTy.getShape();
    auto resultMemRefType =
        MemRefType::get(parentShape, srcMrTy.getElementType(), resultLayout,
                        srcMrTy.getMemorySpace());

    // Intermediate memref.subview keeps logical subview size.
    auto subViewMemRefType =
        MemRefType::get(staticSizes, srcMrTy.getElementType(), resultLayout,
                        srcMrTy.getMemorySpace());

    SmallVector<OpFoldResult> mixedStrides(staticSizes.size(),
                                           rewriter.getIndexAttr(1));
    auto sv = rewriter.create<memref::SubViewOp>(loc, subViewMemRefType, src,
                                                 mixedOffsets, mixedSizes,
                                                 mixedStrides);

    Value vRow;
    Value vCol;
    if (!staticSizes.empty())
      vRow = clampSubViewValidDim(rewriter, loc, op.getValidRow(),
                                  staticSizes[0], op);
    if (staticSizes.size() > 1)
      vCol = clampSubViewValidDim(rewriter, loc, op.getValidCol(),
                                  staticSizes[1], op);

    auto bindOp = rewriter.create<pto::BindTileOp>(
        loc, resultMemRefType, sv.getResult(), vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    markForceDynamicValidShape(bindOp,
                               resultTileTy && resultTileTy.hasDynamicValid(),
                               ctx);
    bindOp->setAttr("pto.view_semantics", rewriter.getStringAttr("subview"));
    rewriter.replaceOp(op, bindOp.getResult());
  }
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
  DefaultInlineVector<mlir::pto::TReshapeOp> reshapes;
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

  DefaultInlineVector<mlir::pto::BitcastOp> bitcasts;
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
    : public mlir::pto::impl::PTOViewToMemrefBase<PTOViewToMemrefPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOViewToMemrefPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;

      // ------------------------------------------------------------------
      // Stage 0: ensure inttoptr values remain scalar-load/store only.
      // ------------------------------------------------------------------
      if (failed(validateIntToPtrUses(func))) {
        signalPassFailure();
        return;
      }

      Block &entry = func.front();
      auto fnTy = func.getFunctionType();

      // ------------------------------------------------------------------
      // Stage 0.10: Rewrite Function Signature
      // ------------------------------------------------------------------
      SmallVector<Type> newInputs;
      for (Type t : fnTy.getInputs()) newInputs.push_back(convertPTOTypeToMemRef(t));

      SmallVector<Type> newResults;
      for (Type t : fnTy.getResults()) newResults.push_back(convertPTOTypeToMemRef(t));

      // Update entry block arguments
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        if (entry.getArgument(i).getType() != newInputs[i]) {
            entry.getArgument(i).setType(newInputs[i]);
        }
      }

      // Update function type
      func.setFunctionType(FunctionType::get(ctx, newInputs, newResults));

      // ------------------------------------------------------------------
      // Stage 0.20: lower pto.inttoptr result types to GM memrefs.
      // ------------------------------------------------------------------
      if (failed(lowerIntToPtrOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 0.30: materialize pto.ptrtoint(addptr ...) byte offsets.
      // ------------------------------------------------------------------
      if (failed(lowerPtrToIntOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 0.5: lower pto.alloc_tile -> memref.alloc + pto.bind_tile
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::AllocTileOp> allocTiles;
      func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

      for (auto op : allocTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
        if (!tbTy) continue;

        // 1. 获取 Shape 和 ElementType
        SmallInlineVector<int64_t> shape(tbTy.getShape().begin(), tbTy.getShape().end());
        Type elemTy = tbTy.getElementType();

        // 2. 计算 Strides (layout-aware when possible)
        SmallVector<int64_t> strides;
        TileLayoutInfo info;
        if (computeTileLayoutInfo(tbTy.getConfigAttr(), elemTy, shape, info)) {
          strides = {info.rowStride, info.colStride};
        } else {
          strides.resize(shape.size());
          int64_t s = 1;
          for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            if (shape[i] != ShapedType::kDynamic) s *= shape[i];
          }
        }

        // 3. 构造 [BindTile 输出] 的动态类型 (Offset: ?)
        // 这必须与 convertPTOTypeToMemRef 返回的类型一致，以便与 Subview 兼容
        auto targetLayout =
            StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides); // offset = ?
        auto targetType =
            MemRefType::get(shape, elemTy, targetLayout, tbTy.getMemorySpace());

        // 4. Preserve tile valid dims (v_row / v_col).
        //
        // `pto.alloc_tile` encodes the valid shape in the result TileBufType
        // (e.g. acc tile may be rows=16 but v_row=1). The alloc op itself does
        // not necessarily carry explicit operands for static valid dims, so we
        // must materialize them from the type to keep them through
        // tile_buf -> memref lowering.
        //
        // For dynamically valid tiles (validShape == [-1, -1]), preserve the
        // runtime operands if present.
        Value vRow = op.getValidRow();
        Value vCol = op.getValidCol();
        // TileBuf valid dims use a negative sentinel (e.g. '?' / -1), which is
        // distinct from MLIR's ShapedType::kDynamic (INT64_MIN). Treat any
        // negative value as dynamic here.
        materializeStaticValidDims(rewriter, loc, tbTy, vRow, vCol);

        // 5. 获取 Config (保持不变)
        auto configAttr = tbTy.getConfigAttr();
        if (!configAttr) configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        // 6. If alloc_tile provides an explicit address, keep the original
        // pointer_cast lowering intact and additionally rebind through
        // pto.bind_tile. PointerCastOp continues to carry the tile metadata
        // used by existing lowering paths, while BindTileOp provides the
        // unified anchor EmitC uses to recover tile_buf information.
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
          continue;
        }

        // 7. Otherwise allocate a concrete memref buffer and bind tile.
        // memref.alloc 要求明确的 layout，不能是动态 offset。
        auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides); // offset = 0
        auto allocType = MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());
        Value alloc = rewriter.create<memref::AllocOp>(loc, allocType);

        // BindTileOp 的 Builder 会自动处理空的 Value，将其视为静态维度
        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, targetType, alloc, vRow ? vRow : Value(), vCol ? vCol : Value(),
            configAttr);
        markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 0.75: lower pto.declare_tile -> pto.declare_tile_memref +
      //             pto.bind_tile
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::DeclareTileOp> declaredTiles;
      func.walk([&](mlir::pto::DeclareTileOp op) { declaredTiles.push_back(op); });

      for (auto op : declaredTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getTile().getType());
        if (!tbTy) {
          op.emitError("declare_tile result must be tile_buf type");
          signalPassFailure();
          return;
        }

        auto targetType = dyn_cast<MemRefType>(convertPTOTypeToMemRef(tbTy));
        if (!targetType) {
          op.emitError("failed to convert declare_tile result to memref type");
          signalPassFailure();
          return;
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
            loc, targetType, declaredMemRef.getResult(),
            vRow ? vRow : Value(), vCol ? vCol : Value(), configAttr);
        markForceDynamicValidShape(bindOp, tbTy.hasDynamicValid(), ctx);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 0.8: normalize pto.tassign result type to match tile operand
      // after tile_buf -> memref lowering (required for verifier consistency).
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::TAssignOp> tassignOps;
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

      // ------------------------------------------------------------------
      // Stage 1: Lower pto.make_tensor_view -> memref.reinterpret_cast
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::MakeTensorViewOp> makeViews;
      func.walk([&](mlir::pto::MakeTensorViewOp op) { makeViews.push_back(op); });

      for (auto op : makeViews) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value baseBuf = op.getOperand(0);
        OpFoldResult off0 = rewriter.getIndexAttr(0);

        // Fold pto.addptr chains into the view base to avoid nested reinterpret_cast.
        bool foldedAddPtr = false;
        {
          Value cur = baseBuf;
          Value totalOffset;
          while (auto add = cur.getDefiningOp<mlir::pto::AddPtrOp>()) {
            foldedAddPtr = true;
            Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
            if (totalOffset)
              totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
            else
              totalOffset = off;
            cur = add.getOperand(0);
          }
          if (cur != baseBuf) {
            baseBuf = cur;
            off0 = totalOffset ? OpFoldResult(totalOffset) : off0;
          }
        }

        auto baseMr = dyn_cast<BaseMemRefType>(baseBuf.getType());
        if (!baseMr) {
             op.emitError("make_tensor_view base must be memref"); signalPassFailure(); return;
        }

        // [修复] 获取动态 Rank (根据 shape 输入的数量)
        size_t rank = op.getShape().size(); 

        // Construct target type with dynamic offset/strides
        Type elemTy = baseMr.getElementType();
        int64_t dyn = ShapedType::kDynamic;
        
        // [修复] 构建 N 维 Strided Layout
        // strides 数组长度必须等于 rank
        SmallVector<int64_t> dynStrides(rank, dyn);
        auto layout = StridedLayoutAttr::get(ctx, /*offset=*/dyn, /*strides=*/dynStrides);
        
        // [修复] 构建 N 维 Shape
        SmallVector<int64_t> dynShape(rank, dyn);
        auto mrTy = MemRefType::get(dynShape, elemTy, layout, baseMr.getMemorySpace());

        SmallInlineVector<OpFoldResult> sizes;
        for (Value v : op.getShape()) sizes.push_back(ensureIndex(rewriter, loc, v, op));

        SmallInlineVector<OpFoldResult> strides;
        for (Value v : op.getStrides()) strides.push_back(ensureIndex(rewriter, loc, v, op));

        auto rc = rewriter.create<memref::ReinterpretCastOp>(
            loc, mrTy, baseBuf, off0, sizes, strides);
        if (foldedAddPtr) {
          rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
        }
        if (auto layoutAttr = op.getLayoutAttr()) {
          rc->setAttr("layout", layoutAttr);
        }

        rewriter.replaceOp(op, rc.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 1.25: Lower pto.get_tensor_view_dim -> memref.dim
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::GetTensorViewDimOp> tvDims;
      func.walk([&](mlir::pto::GetTensorViewDimOp op) { tvDims.push_back(op); });

      for (auto op : tvDims) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value view = op.getTensorView();
        auto mrTy = dyn_cast<BaseMemRefType>(view.getType());
        if (!mrTy)
          continue; // leave it to later passes if it hasn't been lowered yet

        Value dimIdx = op.getDimIndex();
        Value dim = rewriter.create<memref::DimOp>(loc, view, dimIdx);
        rewriter.replaceOp(op, dim);
      }

      // ------------------------------------------------------------------
      // Stage 1.3: Lower pto.partition_view -> memref.subview
      // ------------------------------------------------------------------
      if (failed(lowerPartitionViewOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1.35: Lower pto.subview -> memref.subview + pto.bind_tile
      // ------------------------------------------------------------------
      if (failed(lowerSubViewOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1.4: Lower tile_buf view-like ops (treshape/bitcast)
      // ------------------------------------------------------------------
      if (failed(lowerTileBufViewLikeOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 1.5: Fold pto.addptr chains into load/store_scalar.
      // ------------------------------------------------------------------
      DefaultInlineVector<mlir::pto::LoadScalarOp> loadScalars;
      func.walk([&](mlir::pto::LoadScalarOp op) { loadScalars.push_back(op); });

      for (auto op : loadScalars) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value base = op.getPtr();
        Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);

        bool foldedAddPtr = false;
        while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
          foldedAddPtr = true;
          Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
          if (totalOffset)
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
          else
            totalOffset = off;
          base = add.getOperand(0);
        }

        if (foldedAddPtr) {
          auto newOp = rewriter.create<pto::LoadScalarOp>(
              loc, op.getValue().getType(), base, totalOffset);
          rewriter.replaceOp(op, newOp.getValue());
        }
      }

      DefaultInlineVector<mlir::pto::StoreScalarOp> storeScalars;
      func.walk([&](mlir::pto::StoreScalarOp op) { storeScalars.push_back(op); });

      for (auto op : storeScalars) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value base = op.getPtr();
        Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);

        bool foldedAddPtr = false;
        while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
          foldedAddPtr = true;
          Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
          if (totalOffset)
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
          else
            totalOffset = off;
          base = add.getOperand(0);
        }

        if (foldedAddPtr) {
          rewriter.create<pto::StoreScalarOp>(
              loc, base, totalOffset, op.getValue());
          rewriter.eraseOp(op);
        }
      }

      // ------------------------------------------------------------------
      // Stage 1.75: Fold addptr used by initialize_l2g2l_pipe(gm_addr).
      // This keeps IR well-typed after function arguments are rewritten from
      // !pto.ptr<T> to memref<?xT>.
      // ------------------------------------------------------------------
      bool foldedPipeInitAddPtr = true;
      while (foldedPipeInitAddPtr) {
        foldedPipeInitAddPtr = false;
        DefaultInlineVector<mlir::pto::AddPtrOp> addPtrsForPipeInit;
        func.walk([&](mlir::pto::AddPtrOp op) {
          bool eligible = !op->use_empty();
          for (Operation *user : op->getUsers()) {
            auto init = dyn_cast<mlir::pto::InitializeL2G2LPipeOp>(user);
            if (!init || init.getGmAddr() != op->getResult(0)) {
              eligible = false;
              break;
            }
          }
          if (eligible)
            addPtrsForPipeInit.push_back(op);
        });

        for (auto op : addPtrsForPipeInit) {
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
          foldedPipeInitAddPtr = true;
        }
      }

      // Clean up: addptr should be folded into make_tensor_view.
      DefaultInlineVector<Operation *> addPtrs;
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
      for (auto *op : addPtrs) {
        if (!op)
          continue;
        op->emitError("addptr must feed make_tensor_view,  initialize_l2g2l_pipe(gm_addr) or load/store_scalar for lowering");
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 3: Rewrite Compute Ops
      // ------------------------------------------------------------------
      if (failed(lowerViewToMemrefComputeOps(func, ctx))) {
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 4: Reconcile control-flow result types
      // ------------------------------------------------------------------
      if (failed(reconcileSCFIfResultTypes(func))) {
        signalPassFailure();
        return;
      }
      if (failed(reconcileSCFForResultTypes(func))) {
        signalPassFailure();
        return;
      }

      // Mark memref-form set_validshape only after control-flow result-type
      // reconciliation. Values such as scf.if results can stay tile_buf until
      // this late stage.
      if (failed(markLoweredSetValidShapeOps(func, ctx))) {
        signalPassFailure();
        return;
      }
    }
    
    // Debug Output
    LLVM_DEBUG(llvm::dbgs() << mod.getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<PTOViewToMemrefPass>();
}

} // namespace pto
} // namespace mlir
