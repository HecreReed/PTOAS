// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitCMemoryOps.cpp --------------------------------------------===//
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

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";

struct PointerCastConversion : public OpConversionPattern<pto::PointerCastOp> {
  static bool getIndexConst(Value v, int64_t &out) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  using OpConversionPattern<pto::PointerCastOp>::OpConversionPattern;

  enum class TileRole { Vec, Mat, Left, Right, Acc, Bias, Scaling };

  static void collectUserOpsThroughCasts(Value v, SmallVectorImpl<Operation *> &out) {
    for (Operation *u : v.getUsers()) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(u)) {
        for (Value r : castOp.getResults())
          collectUserOpsThroughCasts(r, out);
        continue;
      }
      out.push_back(u);
    }
  }

  static Value peelUnrealized(Value v) {
    while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      v = castOp.getOperand(0);
    }
    return v;
  }

  static TileRole inferRole(pto::PointerCastOp op) {
    // 1. 优先检查 AddressSpace
    if (auto memRefTy = dyn_cast<MemRefType>(op.getType())) {
      Attribute memorySpace = memRefTy.getMemorySpace();
      if (auto ptoAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        switch (ptoAttr.getAddressSpace()) {
          case pto::AddressSpace::LEFT:  return TileRole::Left;
          case pto::AddressSpace::RIGHT: return TileRole::Right;
          case pto::AddressSpace::ACC:   return TileRole::Acc;
          case pto::AddressSpace::BIAS:  return TileRole::Bias; 
          case pto::AddressSpace::MAT:   return TileRole::Mat;
          case pto::AddressSpace::SCALING: return TileRole::Scaling;
          default: break; 
        }
      }
    }

    // 2. 通过 Usage 推导 (Fallback)
    SmallVector<Operation *, 8> users;
    collectUserOpsThroughCasts(op.getResult(), users);

    for (Operation *user : users) {
      if (auto mm = dyn_cast<pto::TMatmulOp>(user)) {
        if (mm.getDst() && peelUnrealized(mm.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mm.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mm.getRhs()) == op.getResult()) return TileRole::Right;
      }
      if (auto mmacc = dyn_cast<pto::TMatmulAccOp>(user)) {
        if (mmacc.getDst() && peelUnrealized(mmacc.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getAccIn()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mmacc.getRhs()) == op.getResult()) return TileRole::Right;
      }
    }

    return TileRole::Vec;
  }

  // [新增] 辅助函数：判断 Value 是否源自 arith.constant
  static bool isConstant(Value v, int64_t &outVal) {
    if (!v) return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
       if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
           outVal = attr.getInt();
           return true;
       }
    }
    return false;
  }

  LogicalResult matchAndRewrite(pto::PointerCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto selfType = mlir::cast<MemRefType>(op.getType());
    ArrayRef<int64_t> shape = selfType.getShape();
    Type elemType = selfType.getElementType();
    
    // 1. 推导 Tile Role
    TileRole role = inferRole(op);

    // 2. 类型字符串生成 (elemTypeStr, dimStr)
    std::string elemTypeStr = getEmitCScalarTypeToken(elemType);

    std::string dimStr;
    pto::BLayout blayout = pto::BLayout::RowMajor;
    auto dimToString = [&](int64_t dim, const char *symbol,
                           int dimIdx) -> std::string {
        if (dim == ShapedType::kDynamic)
          return std::string(symbol);
        return std::to_string(renderTileTemplateDim(dim, elemType, blayout,
                                                    dimIdx));
    };

    // 3. Role Token
    const char *roleTok = "TileType::Vec";
    switch (role) {
      case TileRole::Left:  roleTok = "TileType::Left"; break;
      case TileRole::Right: roleTok = "TileType::Right"; break;
      case TileRole::Acc:   roleTok = "TileType::Acc"; break;
      case TileRole::Bias:  roleTok = "TileType::Bias"; break;
      case TileRole::Mat:   roleTok = "TileType::Mat"; break;
      case TileRole::Vec:   roleTok = "TileType::Vec"; break;
      case TileRole::Scaling: roleTok = "TileType::Scaling"; break;
    }

    // 4. Config & Layout (support BLayoutAttr/SLayoutAttr/PadValueAttr after namespace change)
    std::string layoutParams = "BLayout::RowMajor";
    std::string extraParams = "";
    if (auto configOpt = op.getConfig()) {
        auto config = *configOpt;
        int32_t blVal = 0;
        if (auto attr = dyn_cast<BLayoutAttr>(config.getBLayout()))
            blVal = static_cast<int32_t>(attr.getValue());
 
        if (blVal == 1) layoutParams = "BLayout::ColMajor";
        blayout = blVal == 1 ? pto::BLayout::ColMajor : pto::BLayout::RowMajor;

        int32_t slVal = 0;
        if (auto attr = dyn_cast<SLayoutAttr>(config.getSLayout()))
            slVal = static_cast<int32_t>(attr.getValue());

        std::string slStr = (slVal == 1) ? "SLayout::RowMajor" : (slVal == 2) ? "SLayout::ColMajor" : "SLayout::NoneBox";

        int32_t frVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getSFractalSize())) frVal = attr.getInt();

        int32_t padVal = 0;
        if (auto attr = dyn_cast<PadValueAttr>(config.getPad()))
            padVal = static_cast<int32_t>(attr.getValue());

        std::string padStr = "PadValue::Null";
        switch (padVal) {
            case 1: padStr = "PadValue::Zero"; break;
            case 2: padStr = "PadValue::Max";  break;
            case 3: padStr = "PadValue::Min";  break;
        }

        int32_t compactVal = 0;
        if (auto attr = dyn_cast<CompactModeAttr>(config.getCompactMode()))
            compactVal = static_cast<int32_t>(attr.getValue());

        std::string compactStr = "CompactMode::Null";
        switch (compactVal) {
            case 1: compactStr = "CompactMode::Normal"; break;
            case 2: compactStr = "CompactMode::RowPlusOne"; break;
        }

        if (!slStr.empty()) {
            extraParams += ", " + slStr + ", " + std::to_string(frVal) + ", " +
                           padStr + ", " + compactStr;
        }
    } else {
        extraParams = ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null";
    }

    if (role == TileRole::Left)
      dimStr = dimToString(shape[0], "M", 0) + ", " +
               dimToString(shape[1], "K", 1);
    else if (role == TileRole::Right)
      dimStr = dimToString(shape[0], "K", 0) + ", " +
               dimToString(shape[1], "N", 1);
    else if (role == TileRole::Bias)
      dimStr = "1, " + dimToString(shape[1], "N", 1);
    else
      dimStr = dimToString(shape[0], "M", 0) + ", " +
               dimToString(shape[1], "N", 1);

    // [核心修改] Valid Dims 处理逻辑 (支持混合静态/动态)
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
    bool rowIsConst = vRow && isConstant(vRow, cRow);
    bool colIsConst = vCol && isConstant(vCol, cCol);

    auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
      if (emitted)
        return emitted;
      return makeEmitCIntConstant(
          rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
    };
    auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
      if (!emitted || !pto::isPTOFloat4PackedType(elemType))
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
                           renderTileTemplateDim(rowIsConst ? cRow : shape[0],
                                                 elemType, blayout, 0)));
      constructorArgs.push_back(
          makeCtorDimValue(maybeScaleDynamicValid(vColEmitC, 1),
                           renderTileTemplateDim(colIsConst ? cCol : shape[1],
                                                 elemType, blayout, 1)));
    } else {
      if (rowIsConst) {
        vrowTok = std::to_string(
            renderTileTemplateDim(cRow, elemType, blayout, 0));
      } else if (vRow) {
        vrowTok = "-1";
        rowIsDynamic = true;
        useConstructor = true;
      } else {
        vrowTok = std::to_string(
            renderTileTemplateDim(shape[0], elemType, blayout, 0));
      }

      if (colIsConst) {
        vcolTok = std::to_string(
            renderTileTemplateDim(cCol, elemType, blayout, 1));
      } else if (vCol) {
        vcolTok = "-1";
        colIsDynamic = true;
        useConstructor = true;
      } else {
        vcolTok = std::to_string(
            renderTileTemplateDim(shape[1], elemType, blayout, 1));
      }

      if (useConstructor) {
        if (rowIsDynamic && vRowEmitC)
          constructorArgs.push_back(maybeScaleDynamicValid(vRowEmitC, 0));
        if (colIsDynamic && vColEmitC)
          constructorArgs.push_back(maybeScaleDynamicValid(vColEmitC, 1));
      }
    }

    // 5. 生成 Tile 类型字符串
    std::string tileTypeStr =
      std::string("Tile<") + roleTok + ", " + elemTypeStr + ", " + dimStr + ", " +
      layoutParams + ", " + vrowTok + ", " + vcolTok + extraParams + ">";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value resultValue;

    if (useConstructor) {
        // 使用 CallOpaqueOp 生成构造函数调用 (Tile v = Tile(...))
        auto ctorOp = rewriter.create<emitc::CallOpaqueOp>(
            loc, 
            tileType,        // Result Type
            tileTypeStr,     // Callee Name (类名)
            ArrayAttr{},     // args
            ArrayAttr{},     // template_args
            ValueRange(constructorArgs) // operands
        );
        resultValue = ctorOp.getResult(0);
    } else {
        // 静态情况 (Tile v;)
        auto varOp = rewriter.create<emitc::VariableOp>(
            loc, 
            tileType, 
            emitc::OpaqueAttr::get(ctx, "")
        );
        resultValue = varOp.getResult();
    }

    // TASSIGN: pto-isa expects an integral address.
    Value addr = adaptor.getAddrs()[0];
    if (isa<emitc::PointerType>(addr.getType()) ||
        (isa<emitc::OpaqueType>(addr.getType()) &&
         cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter.create<emitc::CallOpaqueOp>(
                 loc, u64Ty, "reinterpret_cast",
                 /*args=*/ArrayAttr{}, /*templateArgs=*/rcU64,
                 /*operands=*/ValueRange{addr})
                 .getResult(0);
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{resultValue, addr});

    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_dps / pto.store_dps lowering (FIX: keep optional result)
//===----------------------------------------------------------------------===

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())


static std::optional<int64_t> getStaticIndexLikeValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static FailureOr<Value> buildGlobalTensorViewFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides = {},
    StringRef layoutEnum = "pto::Layout::ND") {
  if (llvm::any_of(shape, [](int64_t dim) {
        return dim == ShapedType::kDynamic;
      }))
    return failure();

  auto *ctx = rewriter.getContext();
  SmallVector<int64_t> rowMajorStrides;
  ArrayRef<int64_t> effectiveStrides = strides;
  if (effectiveStrides.empty()) {
    rowMajorStrides = buildRowMajorStrides(shape);
    effectiveStrides = rowMajorStrides;
  }
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, effectiveStrides, shape5D, stride5D);

  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  auto shapeVal = rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, emitc::OpaqueType::get(ctx, shapeType),
                          shapeType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                      .getResult(0);
  auto strideVal = rewriter
                       .create<emitc::CallOpaqueOp>(
                           loc, emitc::OpaqueType::get(ctx, strideType),
                           strideType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                       .getResult(0);

  std::string gtTypeStr =
      getGlobalTensorTypeStringFromShapeAndStrides(elemTy, shape,
                                                   effectiveStrides,
                                                   layoutEnum);
  auto gtType = emitc::OpaqueType::get(ctx, gtTypeStr);
  auto gt = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, gtTypeStr, ArrayAttr{}, ArrayAttr{},
      ValueRange{ptr, shapeVal, strideVal});
  return gt.getResult(0);
}

static bool parseIntegerTemplateList(StringRef token, StringRef marker,
                                     SmallVectorImpl<int64_t> &values) {
  size_t pos = token.find(marker);
  if (pos == StringRef::npos)
    return false;
  pos += marker.size();
  size_t end = token.find('>', pos);
  if (end == StringRef::npos)
    return false;

  SmallVector<StringRef, 8> parts;
  token.slice(pos, end).split(parts, ',');
  values.clear();
  for (StringRef part : parts) {
    int64_t value = 0;
    if (part.trim().getAsInteger(10, value))
      return false;
    values.push_back(value);
  }
  return true;
}

static LogicalResult getStaticTensorViewStrides(
    Value source, Value convertedSource, pto::TensorViewType sourceType,
    SmallVectorImpl<int64_t> &strides) {
  int64_t rank = sourceType.getRank();
  strides.clear();

  if (auto makeView = source.getDefiningOp<pto::MakeTensorViewOp>()) {
    if ((int64_t)makeView.getStrides().size() != rank)
      return failure();
    for (Value strideValue : makeView.getStrides()) {
      auto cst = getStaticIndexLikeValue(strideValue);
      if (!cst)
        return failure();
      strides.push_back(*cst);
    }
    return success();
  }

  Value src = peelUnrealized(convertedSource);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(src.getType())) {
    SmallVector<int64_t, 5> stride5D;
    StringRef token = opaqueTy.getValue();
    if ((parseIntegerTemplateList(token, "pto::Stride<", stride5D) ||
         parseIntegerTemplateList(token, "Stride<", stride5D)) &&
        (int64_t)stride5D.size() >= rank) {
      strides.append(stride5D.end() - rank, stride5D.end());
      return success();
    }
  }

  auto fallback = buildRowMajorStrides(sourceType.getShape());
  strides.append(fallback.begin(), fallback.end());
  return success();
}

struct PTOPartitionViewToEmitC
    : public OpConversionPattern<mlir::pto::PartitionViewOp> {
  using OpConversionPattern<
      mlir::pto::PartitionViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::PartitionViewOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<pto::TensorViewType>(op.getSource().getType());
    auto resTy = dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
    if (!srcTy || !resTy)
      return rewriter.notifyMatchFailure(
          op, "expected tensor_view source and partition_tensor_view result");

    if (op.getOffsets().size() != static_cast<size_t>(srcTy.getRank()) ||
        op.getSizes().size() != static_cast<size_t>(srcTy.getRank()))
      return rewriter.notifyMatchFailure(op, "rank mismatch");

    for (auto [idx, value] : llvm::enumerate(op.getSizes())) {
      auto cst = getStaticIndexLikeValue(value);
      if (!cst)
        return rewriter.notifyMatchFailure(
            op, "globaltensor partition_view requires static sizes");
      int64_t resultDim = resTy.getShape()[idx];
      if (resultDim != ShapedType::kDynamic && resultDim != *cst)
        return rewriter.notifyMatchFailure(
            op, "partition_view static size does not match result type");
    }

    SmallVector<int64_t> srcStrides;
    if (failed(getStaticTensorViewStrides(op.getSource(), adaptor.getSource(),
                                          srcTy, srcStrides)))
      return rewriter.notifyMatchFailure(
          op, "partition_view requires static source strides");
    int64_t staticLinearOffset = 0;
    SmallVector<std::pair<Value, int64_t>> dynamicOffsetTerms;
    for (auto [idx, values] :
         llvm::enumerate(llvm::zip(op.getOffsets(), adaptor.getOffsets()))) {
      Value originalOffset = std::get<0>(values);
      Value convertedOffset = std::get<1>(values);
      int64_t stride = srcStrides[idx];
      if (stride == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic source stride is not supported");

      if (auto cst = getStaticIndexLikeValue(originalOffset)) {
        if (*cst != 0)
          staticLinearOffset += (*cst) * stride;
        continue;
      }
      dynamicOffsetTerms.push_back({convertedOffset, stride});
    }

    auto *ctx = rewriter.getContext();
    std::string elemTypeStr = getElemTypeStringForGT(srcTy.getElementType());
    auto ptrTy = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
    Value src = peelUnrealized(adaptor.getSource());
    auto data = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), ptrTy, "PTOAS__GLOBAL_TENSOR_DATA",
                        ArrayAttr{}, ArrayAttr{}, ValueRange{src})
                    .getResult(0);
    Value ptr = data;
    if (!dynamicOffsetTerms.empty()) {
      Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
      auto makeU32 = [&](int64_t value) {
        return makeEmitCIntConstant(rewriter, op.getLoc(), u32Ty, value);
      };
      auto asU32 = [&](Value value) -> Value {
        if (value.getType() == u32Ty)
          return value;
        return rewriter.create<emitc::CastOp>(op.getLoc(), u32Ty, value)
            .getResult();
      };

      Value totalOffset = makeU32(staticLinearOffset);
      for (auto [offsetValue, stride] : dynamicOffsetTerms) {
        Value term = asU32(offsetValue);
        if (stride != 1) {
          Value strideValue = makeU32(stride);
          term = rewriter
                     .create<emitc::MulOp>(op.getLoc(), u32Ty, term,
                                           strideValue)
                     .getResult();
        }
        totalOffset = rewriter
                          .create<emitc::AddOp>(op.getLoc(), u32Ty,
                                                totalOffset, term)
                          .getResult();
      }
      ptr = rewriter
                .create<emitc::AddOp>(op.getLoc(), data.getType(), data,
                                      totalOffset)
                .getResult();
    } else {
      ptr = applyStaticMemrefOffset(rewriter, op.getLoc(), data,
                                    staticLinearOffset);
    }

    auto resultOr = buildGlobalTensorViewFromPointer(
        rewriter, op.getLoc(), ptr, resTy.getElementType(), resTy.getShape(),
        srcStrides);
    if (failed(resultOr))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize partition GlobalTensor");

    rewriter.replaceOp(op, *resultOr);
    return success();
  }
};


} // namespace

void populatePTOToEmitCMemoryOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx) {
  patterns.add<PointerCastConversion>(typeConverter, ctx);
  patterns.add<PTOPartitionViewToEmitC>(typeConverter, ctx);
}

} // namespace mlir::pto
