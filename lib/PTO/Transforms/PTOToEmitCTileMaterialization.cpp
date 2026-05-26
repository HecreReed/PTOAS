// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCTileMaterialization.cpp ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOToEmitCInternal.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

// =============================================================================
// 2. BindTileOp Lowering (FIX: Trace back to physical address)
// =============================================================================
struct PTOBindTileToEmitC : public OpConversionPattern<pto::BindTileOp> {
  using OpConversionPattern::OpConversionPattern;

  struct TileBuildSpec {
    std::string tileTypeStr;
    bool useConstructor = false;
    SmallVector<Value> constructorArgs;
  };

  static bool getIndexConst(Value v, int64_t &out) {
    if (!v)
      return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  static bool getTilePointerStrides(pto::TileBufConfigAttr configAttr,
                                    Type elemTy, int64_t rows, int64_t cols,
                                    int64_t &rowStride,
                                    int64_t &colStride) {
    if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
      return false;

    int32_t blVal = 0;
    if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
      blVal = static_cast<int32_t>(blAttr.getValue());
    else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getBLayout()))
      blVal = static_cast<int32_t>(intAttr.getInt());

    int32_t slVal = 0;
    if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout()))
      slVal = static_cast<int32_t>(slAttr.getValue());
    else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getSLayout()))
      slVal = static_cast<int32_t>(intAttr.getInt());

    bool boxed = slVal != 0;
    int64_t innerRows = 1;
    int64_t innerCols = 1;
    if (boxed) {
      int32_t fractal = 512;
      if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
        fractal = static_cast<int32_t>(frAttr.getInt());

      unsigned elemBytes = pto::getPTOStorageElemByteSize(elemTy);
      if (elemBytes == 0)
        return false;

      switch (fractal) {
      case 1024:
        innerRows = 16;
        innerCols = 16;
        break;
      case 32:
        innerRows = 16;
        innerCols = 2;
        break;
      case 512:
        if (slVal == 1) {
          innerRows = 16;
          innerCols = 32 / elemBytes;
        } else if (slVal == 2) {
          innerRows = 32 / elemBytes;
          innerCols = 16;
        } else {
          return false;
        }
        break;
      default:
        return false;
      }
      if (innerRows <= 0 || innerCols <= 0)
        return false;
    }

    if (!boxed) {
      if (blVal == 1) {
        rowStride = 1;
        colStride = rows;
      } else {
        rowStride = cols;
        colStride = 1;
      }
      return true;
    }

    if (blVal == 1) {
      if (slVal != 1)
        return false;
      rowStride = innerCols;
      colStride = rows;
      return true;
    }

    rowStride = cols;
    colStride = innerRows;
    return true;
  }

  LogicalResult matchAndRewrite(pto::BindTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto configAttr = op.getConfigAttr();
    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool isSubView = viewSemantics && viewSemantics.getValue() == "subview";

    auto peelAllCasts = [](Value v) {
      while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        v = castOp.getOperand(0);
      if (auto castOp = v.getDefiningOp<emitc::CastOp>())
        v = castOp.getOperand();
      return v;
    };
    auto isTileLike = [](Value v) -> bool {
      auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return false;
      StringRef s = ot.getValue();
      return s.contains("Tile<") || s.contains("ConvTile<");
    };
    auto buildTileSpec = [&]() -> FailureOr<TileBuildSpec> {
      auto resMrTy = dyn_cast<MemRefType>(op.getType());
      if (!resMrTy)
        return failure();

      const char *roleTok = "TileType::Vec";
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace())) {
        switch (asAttr.getAddressSpace()) {
        case pto::AddressSpace::VEC:
          roleTok = "TileType::Vec";
          break;
        case pto::AddressSpace::MAT:
          roleTok = "TileType::Mat";
          break;
        case pto::AddressSpace::LEFT:
          roleTok = "TileType::Left";
          break;
        case pto::AddressSpace::RIGHT:
          roleTok = "TileType::Right";
          break;
        case pto::AddressSpace::ACC:
          roleTok = "TileType::Acc";
          break;
        case pto::AddressSpace::BIAS:
          roleTok = "TileType::Bias";
          break;
        case pto::AddressSpace::SCALING:
          roleTok = "TileType::Scaling";
          break;
        case pto::AddressSpace::GM:
        case pto::AddressSpace::Zero:
          roleTok = "TileType::Vec";
          break;
        }
      }

      Type elemTy = resMrTy.getElementType();
      Type emitElemTy = getTypeConverter()->convertType(elemTy);
      if (!emitElemTy)
        return failure();
      auto emitElemOpaque = dyn_cast<emitc::OpaqueType>(emitElemTy);
      if (!emitElemOpaque)
        return failure();
      std::string elemTypeStr = emitElemOpaque.getValue().str();

      if (resMrTy.getRank() < 2)
        return failure();
      int64_t rows = resMrTy.getDimSize(0);
      int64_t cols = resMrTy.getDimSize(1);
      if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
        return failure();

      std::string blTok = "BLayout::RowMajor";
      if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
        if (static_cast<int32_t>(blAttr.getValue()) == 1)
          blTok = "BLayout::ColMajor";
      }
      pto::BLayout blayout = getTileBufBLayoutValue(configAttr);

      if (isSubView) {
        auto subMrTy = dyn_cast<MemRefType>(op.getSource().getType());
        auto subViewOp = op.getSource().getDefiningOp<memref::SubViewOp>();
        if (subMrTy && subMrTy.getRank() >= 2 && subViewOp) {
          int64_t subRows = subMrTy.getDimSize(0);
          int64_t subCols = subMrTy.getDimSize(1);
          SmallVector<int64_t> inheritedStrides;
          int64_t inheritedOffset = ShapedType::kDynamic;

          if (!pto::isPTOFloat4PackedType(elemTy) &&
              subRows != ShapedType::kDynamic &&
              subCols != ShapedType::kDynamic &&
              succeeded(getStridesAndOffset(subMrTy, inheritedStrides,
                                            inheritedOffset)) &&
              inheritedStrides.size() >= 2) {
            int64_t childRowStride = 0;
            int64_t childColStride = 0;
            bool sameStrides = getTilePointerStrides(
                configAttr, elemTy, subRows, subCols, childRowStride,
                childColStride);
            sameStrides = sameStrides &&
                          inheritedStrides[0] == childRowStride &&
                          inheritedStrides[1] == childColStride;
            if (sameStrides) {
              rows = subRows;
              cols = subCols;
            }
          }
        }
      }

      std::string slTok = "SLayout::NoneBox";
      if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
        int32_t slVal = static_cast<int32_t>(slAttr.getValue());
        slTok = (slVal == 1) ? "SLayout::RowMajor"
                             : (slVal == 2) ? "SLayout::ColMajor"
                                            : "SLayout::NoneBox";
      }

      int32_t fractal = 512;
      if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
        fractal = frAttr.getInt();

      std::string padTok = "PadValue::Null";
      if (auto padAttr = dyn_cast<PadValueAttr>(configAttr.getPad())) {
        switch (static_cast<int32_t>(padAttr.getValue())) {
        case 1:
          padTok = "PadValue::Zero";
          break;
        case 2:
          padTok = "PadValue::Max";
          break;
        case 3:
          padTok = "PadValue::Min";
          break;
        default:
          padTok = "PadValue::Null";
          break;
        }
      }

      std::string compactTok = "CompactMode::Null";
      if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
        switch (static_cast<int32_t>(compactAttr.getValue())) {
        case 1:
          compactTok = "CompactMode::Normal";
          break;
        case 2:
          compactTok = "CompactMode::RowPlusOne";
          break;
        default:
          compactTok = "CompactMode::Null";
          break;
        }
      }

      std::string vrowTok, vcolTok;
      bool useConstructor = false;
      bool rowIsDynamic = false;
      bool colIsDynamic = false;
      SmallVector<Value> constructorArgs;

      Value vRow = op.getValidRow();
      Value vCol = op.getValidCol();
      Value vRowEmitC = adaptor.getValidRow();
      Value vColEmitC = adaptor.getValidCol();
      bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
      int64_t cRow = 0, cCol = 0;
      bool rowIsConst = vRow && getIndexConst(vRow, cRow);
      bool colIsConst = vCol && getIndexConst(vCol, cCol);

      auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
        if (emitted)
          return emitted;
        return makeEmitCIntConstant(
            rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
      };
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
      };

      if (forceDynamicValid) {
        vrowTok = "-1";
        vcolTok = "-1";
        useConstructor = true;
        constructorArgs.push_back(
            makeCtorDimValue(maybeScaleDynamicValid(vRowEmitC, 0),
                             renderTileTemplateDim(rowIsConst ? cRow : rows,
                                                   elemTy, blayout, 0)));
        constructorArgs.push_back(
            makeCtorDimValue(maybeScaleDynamicValid(vColEmitC, 1),
                             renderTileTemplateDim(colIsConst ? cCol : cols,
                                                   elemTy, blayout, 1)));
      } else {
        if (rowIsConst) {
          vrowTok = std::to_string(
              renderTileTemplateDim(cRow, elemTy, blayout, 0));
        } else if (vRow) {
          vrowTok = "-1";
          rowIsDynamic = true;
          useConstructor = true;
        } else {
          vrowTok = std::to_string(
              renderTileTemplateDim(rows, elemTy, blayout, 0));
        }

        if (colIsConst) {
          vcolTok = std::to_string(
              renderTileTemplateDim(cCol, elemTy, blayout, 1));
        } else if (vCol) {
          vcolTok = "-1";
          colIsDynamic = true;
          useConstructor = true;
        } else {
          vcolTok = std::to_string(
              renderTileTemplateDim(cols, elemTy, blayout, 1));
        }

        if (useConstructor) {
          if (rowIsDynamic && vRowEmitC)
            constructorArgs.push_back(maybeScaleDynamicValid(vRowEmitC, 0));
          if (colIsDynamic && vColEmitC)
            constructorArgs.push_back(maybeScaleDynamicValid(vColEmitC, 1));
        }
      }

      std::string tileTypeStr = std::string("Tile<") + roleTok + ", " +
                                elemTypeStr + ", " +
                                std::to_string(renderTileTemplateDim(
                                    rows, elemTy, blayout, 0)) +
                                ", " +
                                std::to_string(renderTileTemplateDim(
                                    cols, elemTy, blayout, 1)) +
                                ", " + blTok +
                                ", " + vrowTok + ", " + vcolTok + ", " + slTok +
                                ", " + std::to_string(fractal) + ", " + padTok +
                                ", " + compactTok +
                                ">";
      return TileBuildSpec{tileTypeStr, useConstructor, constructorArgs};
    };

    auto buildTileValue = [&](const TileBuildSpec &spec,
                              bool forceDeclaration = false) -> Value {
      auto tileType = emitc::OpaqueType::get(ctx, spec.tileTypeStr);
      if (spec.useConstructor && !forceDeclaration) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, tileType, spec.tileTypeStr,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(spec.constructorArgs))
            .getResult(0);
      }

      return rewriter
          .create<emitc::VariableOp>(loc, tileType, emitc::OpaqueAttr::get(ctx, ""))
          .getResult();
    };

    auto emitElemTypeToString = [&](Type elemTy) -> std::string {
      return getEmitCScalarTypeToken(elemTy);
    };

    auto buildIntegralAddress = [&](Value sourceValue) -> FailureOr<Value> {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});

      Value rawPtr = sourceValue;
      if (auto ot = dyn_cast<emitc::OpaqueType>(sourceValue.getType())) {
        StringRef tyStr = ot.getValue();
        if (tyStr.contains("Tile<") || tyStr.contains("ConvTile<")) {
          auto srcMrTy = dyn_cast<MemRefType>(op.getSource().getType());
          if (!srcMrTy)
            return failure();
          std::string elemTok = emitElemTypeToString(srcMrTy.getElementType());
          pto::AddressSpace as = pto::AddressSpace::GM;
          if (auto asAttr =
                  dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace()))
            as = asAttr.getAddressSpace();
          rawPtr = materializeTileDataValue(rewriter, loc, sourceValue, as,
                                            elemTok);
        }
      }

      if (isSetFFTsPointerLikeType(rawPtr.getType())) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                         ArrayAttr{}, rcU64, ValueRange{rawPtr})
            .getResult(0);
      }

      if (rawPtr.getType() == u64Ty)
        return rawPtr;
      return rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    };

    if (op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>()) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      rewriter.replaceOp(op, buildTileValue(*tileSpec));
      return success();
    }

    Value tileCandidate = peelAllCasts(adaptor.getSource());
    if (viewSemantics && viewSemantics.getValue() == "bitcast" &&
        isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    if (viewSemantics && viewSemantics.getValue() == "treshape" &&
        isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec, /*forceDeclaration=*/true);

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TRESHAPE",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, tileCandidate});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    // Subview origins are kept distinct from generic tile rebinding:
    // even when source/destination C++ tile types match, subview may carry
    // shifted base address semantics and should materialize a fresh handle.
    if (isSubView) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    // Generic tile-to-tile rebind path: preserve the same backing storage and
    // rebuild a sibling tile with updated metadata/valid dims.
    if (isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();

      if (!tileSpec->useConstructor) {
        if (auto srcTy = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
          if (srcTy.getValue() == tileSpec->tileTypeStr) {
            rewriter.replaceOp(op, tileCandidate);
            return success();
          }
        }
      }

      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    SmallVector<Value> physAddrs;
    Value source = op.getSource();

    while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>())
      source = castOp.getOperand(0);

    if (auto upstreamCast = source.getDefiningOp<pto::PointerCastOp>()) {
      auto upstreamOperands = upstreamCast.getAddrs();
      physAddrs.append(upstreamOperands.begin(), upstreamOperands.end());
    } else {
      physAddrs.push_back(adaptor.getSource());
    }

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();

    auto newCast = rewriter.create<pto::PointerCastOp>(
        loc, op.getType(), physAddrs, vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    if (viewSemantics)
      newCast->setAttr("pto.view_semantics", viewSemantics);
    if (op->hasAttr(kForceDynamicValidShapeAttrName))
      newCast->setAttr(kForceDynamicValidShapeAttrName,
                       op->getAttr(kForceDynamicValidShapeAttrName));
    rewriter.replaceOp(op, newCast.getResult());

    return success();
  }
};

struct PTOAllocTileToEmitC
    : public OpConversionPattern<pto::AllocTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AllocTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto tileTy = cast<pto::TileBufType>(op.getResult().getType());
    auto tileTypeString = getEmitCTileTypeString(tileTy);
    if (!tileTypeString)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 alloc_tile handles can be converted to EmitC");

    Type convertedTy = getTypeConverter()->convertType(tileTy);
    if (!convertedTy)
      convertedTy = emitc::OpaqueType::get(ctx, *tileTypeString);

    auto validShape = tileTy.getValidShape();
    bool hasDynamicValidDim =
        llvm::any_of(validShape, [](int64_t dim) { return dim < 0; });
    bool useConstructor = hasDynamicValidDim;

    SmallVector<Value> constructorArgs;
    if (useConstructor) {
      Type elemTy = tileTy.getElementType();
      pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two)
            .getResult();
      };

      if (validShape.size() > 0 && validShape[0] < 0) {
        Value validRow = adaptor.getValidRow();
        if (!validRow)
          return rewriter.notifyMatchFailure(
              op, "dynamic alloc_tile valid row must have an operand");
        if (validRow)
          validRow = peelUnrealized(validRow);
        constructorArgs.push_back(maybeScaleDynamicValid(validRow, 0));
      }
      if (validShape.size() > 1 && validShape[1] < 0) {
        Value validCol = adaptor.getValidCol();
        if (!validCol)
          return rewriter.notifyMatchFailure(
              op, "dynamic alloc_tile valid col must have an operand");
        if (validCol)
          validCol = peelUnrealized(validCol);
        constructorArgs.push_back(maybeScaleDynamicValid(validCol, 1));
      }
    }

    Value tile;
    if (useConstructor) {
      tile = rewriter
                 .create<emitc::CallOpaqueOp>(
                     loc, convertedTy, *tileTypeString, ArrayAttr{},
                     ArrayAttr{}, ValueRange(constructorArgs))
                 .getResult(0);
    } else {
      tile =
          rewriter
              .create<emitc::VariableOp>(
                  loc, convertedTy, emitc::OpaqueAttr::get(ctx, ""))
              .getResult();
    }

    Value addr = adaptor.getAddr();
    if (addr) {
      addr = peelUnrealized(addr);
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      if (isa<emitc::PointerType>(addr.getType()) ||
          (isa<emitc::OpaqueType>(addr.getType()) &&
           cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
        auto rcU64 =
            rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
        addr = rewriter
                   .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                                ArrayAttr{}, rcU64,
                                                ValueRange{addr})
                   .getResult(0);
      } else if (addr.getType() != u64Ty) {
        addr = rewriter.create<emitc::CastOp>(loc, u64Ty, addr).getResult();
      }

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{tile, addr});
    }

    rewriter.replaceOp(op, tile);
    return success();
  }
};

static FailureOr<Value>
createEmitCTileVariable(ConversionPatternRewriter &rewriter, Location loc,
                        const TypeConverter *typeConverter,
                        pto::TileBufType tileTy) {
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();

  Type convertedTy = typeConverter->convertType(tileTy);
  if (!convertedTy)
    convertedTy = emitc::OpaqueType::get(rewriter.getContext(), *tileTypeString);

  return rewriter
      .create<emitc::VariableOp>(
          loc, convertedTy, emitc::OpaqueAttr::get(rewriter.getContext(), ""))
      .getResult();
}

struct PTOTReshapeToEmitC : public OpConversionPattern<pto::TReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tileTy = dyn_cast<pto::TileBufType>(op.getResult().getType());
    if (!tileTy)
      return failure();

    FailureOr<Value> dst =
        createEmitCTileVariable(rewriter, op.getLoc(), getTypeConverter(), tileTy);
    if (failed(dst))
      return failure();

    Value src = peelUnrealized(adaptor.getSrc());
    if (auto castOp = src.getDefiningOp<emitc::CastOp>())
      src = castOp.getOperand();

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TRESHAPE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*dst, src});
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOBitcastToEmitC : public OpConversionPattern<pto::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<pto::TileBufType>(op.getResult().getType());
    auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!dstTy || !srcTy)
      return failure();

    FailureOr<Value> dst =
        createEmitCTileVariable(rewriter, op.getLoc(), getTypeConverter(), dstTy);
    if (failed(dst))
      return failure();

    Value src = peelUnrealized(adaptor.getSrc());
    if (auto castOp = src.getDefiningOp<emitc::CastOp>())
      src = castOp.getOperand();

    pto::AddressSpace as = pto::AddressSpace::GM;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcTy.getMemorySpace()))
      as = asAttr.getAddressSpace();
    std::string elemTok = getEmitCScalarTypeToken(srcTy.getElementType());

    Value rawPtr = materializeTileDataValue(rewriter, op.getLoc(), src, as, elemTok);
    auto u64Ty = emitc::OpaqueType::get(rewriter.getContext(), "uint64_t");
    Value addr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                        "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(op.getLoc(), u64Ty,
                                              "reinterpret_cast", ArrayAttr{},
                                              rcU64, ValueRange{rawPtr})
                 .getResult(0);
    } else if (addr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(op.getLoc(), u64Ty, addr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*dst, addr});
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOMaterializeTileToEmitC
    : public OpConversionPattern<pto::MaterializeTileOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool isTileLike(Value v) {
    auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
    if (!ot)
      return false;
    StringRef s = ot.getValue();
    return s.contains("Tile<") || s.contains("ConvTile<");
  }

  LogicalResult matchAndRewrite(pto::MaterializeTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto tileTy = cast<pto::TileBufType>(op.getResult().getType());
    auto tileTypeString = getEmitCTileTypeString(tileTy);
    if (!tileTypeString)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 tile_buf handles can be materialized to EmitC");

    Type convertedTy = getTypeConverter()->convertType(tileTy);
    if (!convertedTy)
      convertedTy = emitc::OpaqueType::get(ctx, *tileTypeString);

    Value source = peelUnrealized(adaptor.getSource());
    if (auto castOp = source.getDefiningOp<emitc::CastOp>())
      source = castOp.getOperand();

    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
    bool isReshape = viewSemantics && viewSemantics.getValue() == "treshape";
    bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
    bool sourceIsDeclaredTile =
        op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>();

    auto createTileValue = [&]() -> Value {
      SmallVector<Value, 2> constructorArgs;
      bool useConstructor = false;
      pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
      Type elemTy = tileTy.getElementType();
      auto shape = tileTy.getShape();
      auto validShape = tileTy.getValidShape();

      auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
        if (emitted)
          return emitted;
        return makeEmitCIntConstant(
            rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
      };
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
      };
      auto fallbackDim = [&](int dimIdx) {
        return renderTileTemplateDim(shape[dimIdx], elemTy, blayout, dimIdx);
      };

      if (forceDynamicValid) {
        useConstructor = true;
        constructorArgs.push_back(makeCtorDimValue(
            maybeScaleDynamicValid(adaptor.getValidRow(), 0), fallbackDim(0)));
        constructorArgs.push_back(makeCtorDimValue(
            maybeScaleDynamicValid(adaptor.getValidCol(), 1), fallbackDim(1)));
      } else {
        if (validShape[0] == ShapedType::kDynamic) {
          useConstructor = true;
          constructorArgs.push_back(makeCtorDimValue(
              maybeScaleDynamicValid(adaptor.getValidRow(), 0), fallbackDim(0)));
        }
        if (validShape[1] == ShapedType::kDynamic) {
          useConstructor = true;
          constructorArgs.push_back(makeCtorDimValue(
              maybeScaleDynamicValid(adaptor.getValidCol(), 1), fallbackDim(1)));
        }
      }

      if (useConstructor) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, convertedTy, *tileTypeString,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(constructorArgs))
            .getResult(0);
      }

      return rewriter
          .create<emitc::VariableOp>(loc, convertedTy,
                                     emitc::OpaqueAttr::get(ctx, ""))
          .getResult();
    };

    if (!isSubview && !forceDynamicValid && isTileLike(source)) {
      if (auto srcTy = dyn_cast<emitc::OpaqueType>(source.getType())) {
        if (srcTy.getValue() == *tileTypeString) {
          rewriter.replaceOp(op, source);
          return success();
        }
      }
    }

    Value tile = createTileValue();
    if (sourceIsDeclaredTile) {
      rewriter.replaceOp(op, tile);
      return success();
    }

    if (isReshape && isTileLike(source)) {
      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TRESHAPE",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{tile, source});
      rewriter.replaceOp(op, tile);
      return success();
    }

    pto::AddressSpace as = pto::AddressSpace::GM;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(tileTy.getMemorySpace()))
      as = asAttr.getAddressSpace();
    std::string elemTok = getEmitCScalarTypeToken(tileTy.getElementType());

    Value rawPtr = source;
    if (isTileLike(rawPtr))
      rawPtr = materializeTileDataValue(rewriter, loc, rawPtr, as, elemTok);

    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    Value addr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                              ArrayAttr{}, rcU64,
                                              ValueRange{rawPtr})
                 .getResult(0);
    } else if (rawPtr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{tile, addr});
    rewriter.replaceOp(op, tile);
    return success();
  }
};


} // namespace

void populatePTOToEmitCTileMaterializationPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOAllocTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaterializeTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOBindTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOTReshapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOBitcastToEmitC>(typeConverter, ctx);
}

} // namespace mlir::pto
