// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitC.cpp - PTO to EmitC conversion pass ----------------------===//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic ignored "-Woverloaded-virtual"
// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8

#include <cassert>
#include <climits>

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTOToEmitCInternal.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                   
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
#define GEN_PASS_DEF_EMITPTOMANUAL
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

static bool getStaticMemrefLayout(MemRefType mrTy,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);
static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs);
static std::string getGlobalTensorTypeStringFromShape(Type elemTy,
                                                      ArrayRef<int64_t> shape,
                                                      StringRef layoutEnum =
                                                          "pto::Layout::ND");
static emitc::OpaqueType getGlobalTensorOpaqueTypeFromShape(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum = "pto::Layout::ND");

static const char *addrSpaceQualifier(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::Zero:
    return "__gm__";
  case pto::AddressSpace::VEC:
    return "__ubuf__";
  case pto::AddressSpace::GM:
    return "__gm__";
  case pto::AddressSpace::MAT:
    return "__cbuf__";
  case pto::AddressSpace::LEFT:
    return "__ca__";
  case pto::AddressSpace::RIGHT:
    return "__cb__";
  case pto::AddressSpace::ACC:
    return "__cc__";
  case pto::AddressSpace::BIAS:
    // Bias tiles are special in pto-isa; keep a safe fallback qualifier.
    return "__gm__";
  case pto::AddressSpace::SCALING:
    // pto-isa TileType::Scaling maps to __fbuf__ (see pto/common/memory.hpp).
    return "__fbuf__";
  }
  return "__gm__";
}

[[maybe_unused]] static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
[[maybe_unused]] static constexpr llvm::StringLiteral kLoweredSetValidShapeConfigAttrName =
    "__pto.lowered_set_validshape_config";
[[maybe_unused]] static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";
[[maybe_unused]] static constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";

Value mlir::pto::peelUnrealized(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return v;
}


static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor);

static bool hasCompatibleKnownExtentForMGather(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == rhs;
}

static bool isKnownUnitExtentForMGather(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

struct GatherScatterShapeLayoutInfo {
  SmallVector<int64_t, 2> shape;
  bool rowMajor = false;
  bool colMajor = false;
};

static std::optional<GatherScatterShapeLayoutInfo>
getGatherScatterShapeLayoutInfo(Type ty) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    ArrayRef<int64_t> validShape = tileTy.getValidShape();
    if (validShape.size() != 2)
      return std::nullopt;

    GatherScatterShapeLayoutInfo info;
    info.shape.assign(validShape.begin(), validShape.end());
    int32_t blayout = tileTy.getBLayoutValueI32();
    info.rowMajor = blayout == static_cast<int32_t>(pto::BLayout::RowMajor);
    info.colMajor = blayout == static_cast<int32_t>(pto::BLayout::ColMajor);
    return info;
  }

  auto memRefTy = dyn_cast<MemRefType>(ty);
  if (!memRefTy || memRefTy.getRank() != 2)
    return std::nullopt;

  SmallVector<int64_t, 4> strides;
  int64_t offset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(memRefTy, strides, offset)) ||
      strides.size() != 2)
    return std::nullopt;

  GatherScatterShapeLayoutInfo info;
  info.shape.assign(memRefTy.getShape().begin(), memRefTy.getShape().end());
  info.rowMajor = strides[1] == 1;
  info.colMajor = strides[0] == 1;
  return info;
}

static bool isRowCoalescedMGatherIndexType(Type dataTy, Type idxTy) {
  auto dataInfo = getGatherScatterShapeLayoutInfo(dataTy);
  auto idxInfo = getGatherScatterShapeLayoutInfo(idxTy);
  if (!dataInfo || !idxInfo)
    return false;

  const bool rowCoalesce1xR =
      idxInfo->rowMajor && isKnownUnitExtentForMGather(idxInfo->shape[0]) &&
      hasCompatibleKnownExtentForMGather(idxInfo->shape[1], dataInfo->shape[0]);
  const bool rowCoalesceRx1 =
      idxInfo->colMajor &&
      hasCompatibleKnownExtentForMGather(idxInfo->shape[0], dataInfo->shape[0]) &&
      isKnownUnitExtentForMGather(idxInfo->shape[1]);
  return rowCoalesce1xR || rowCoalesceRx1;
}

static std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>("layout"))
    return attr.getLayout();
  return std::nullopt;
}

static std::optional<mlir::pto::Layout> resolveLayoutFromValueChain(Value v) {
  v = peelUnrealized(v);
  while (Operation *def = v.getDefiningOp()) {
    if (auto layout = getLayoutAttrFromOp(def))
      return layout;
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      v = peelUnrealized(subview.getSource());
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = peelUnrealized(reinterpret.getSource());
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = peelUnrealized(cast.getSource());
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized->getNumOperands() == 0)
        break;
      v = peelUnrealized(unrealized.getOperand(0));
      continue;
    }
    break;
  }
  return std::nullopt;
}

static std::optional<mlir::pto::Layout>
resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) {
  if (auto layout = getLayoutAttrFromOp(anchor))
    return layout;
  return resolveLayoutFromValueChain(basePtr);
}

static std::string layoutToEmitCString(mlir::pto::Layout layout) {
  switch (layout) {
  case mlir::pto::Layout::ND:
    return "pto::Layout::ND";
  case mlir::pto::Layout::DN:
    return "pto::Layout::DN";
  case mlir::pto::Layout::NZ:
    return "pto::Layout::NZ";
  }
  return "pto::Layout::ND";
}

bool mlir::pto::isEmitCGlobalTensorLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  return opaqueTy && opaqueTy.getValue().contains("GlobalTensor<");
}

std::string mlir::pto::getEmitCScalarTypeToken(Type elemTy) {
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E4M3() || elemTy.isFloat8E4M3FN() ||
       elemTy.isFloat8E4M3FNUZ() || elemTy.isFloat8E4M3B11FNUZ()))
    return "float8_e4m3_t";
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E5M2() || elemTy.isFloat8E5M2FNUZ()))
    return "float8_e5m2_t";
  if (isa<pto::HiF8Type>(elemTy))
    return "hifloat8_t";
  if (isa<pto::F4E1M2x2Type>(elemTy))
    return "float4_e1m2x2_t";
  if (isa<pto::F4E2M1x2Type>(elemTy))
    return "float4_e2m1x2_t";
  if (elemTy.isF16())
    return "half";
  if (elemTy.isBF16())
    return "bfloat16_t";
  if (elemTy.isF32())
    return "float";
  if (elemTy.isF64())
    return "double";
  if (elemTy.isInteger(8))
    return (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8)) ? "int8_t"
                                                                       : "uint8_t";
  if (elemTy.isInteger(16))
    return (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
               ? "int16_t"
               : "uint16_t";
  if (elemTy.isInteger(32))
    return (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
               ? "int32_t"
               : "uint32_t";
  if (elemTy.isInteger(64))
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  return "float";
}

static emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                              StringRef pointeeTypeStr) {
  return emitc::PointerType::get(emitc::OpaqueType::get(ctx, pointeeTypeStr));
}

static emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                              StringRef qualifier,
                                              StringRef elemTypeStr) {
  return getEmitCPointerType(ctx, (qualifier + " " + elemTypeStr).str());
}

static bool isEmitCPointerLikeType(Type ty) {
  if (isa<emitc::PointerType>(ty))
    return true;
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty))
    return opaqueTy.getValue().ends_with("*");
  return false;
}

[[maybe_unused]] static int64_t getEmitCScalarByteWidth(Type elemTy) {
  if (pto::getPTOStorageElemByteSize(elemTy) == 1)
    return 1;
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(16))
    return 2;
  if (elemTy.isF32() || elemTy.isInteger(32))
    return 4;
  if (elemTy.isF64() || elemTy.isInteger(64))
    return 8;
  return 4;
}

static std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr);
static std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr);
static std::string tileBufPadToken(pto::TileBufConfigAttr configAttr);
pto::BLayout mlir::pto::getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr);
int64_t mlir::pto::renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx);

static const char *tileRoleToken(Attribute memorySpace) {
  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    switch (asAttr.getAddressSpace()) {
    case pto::AddressSpace::VEC:
      return "TileType::Vec";
    case pto::AddressSpace::MAT:
      return "TileType::Mat";
    case pto::AddressSpace::LEFT:
      return "TileType::Left";
    case pto::AddressSpace::RIGHT:
      return "TileType::Right";
    case pto::AddressSpace::ACC:
      return "TileType::Acc";
    case pto::AddressSpace::BIAS:
      return "TileType::Bias";
    case pto::AddressSpace::SCALING:
      return "TileType::Scaling";
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return "TileType::Vec";
    }
  }
  return "TileType::Vec";
}

static std::string tileBufCompactToken(pto::TileBufConfigAttr configAttr) {
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
  return compactTok;
}

std::optional<std::string> mlir::pto::getEmitCTileTypeString(pto::TileBufType type) {
  if (type.getRank() != 2)
    return std::nullopt;
  auto validShape = type.getValidShape();
  if (validShape.size() != 2)
    return std::nullopt;

  Type elemTy = type.getElementType();
  auto configAttr = type.getConfigAttr();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  auto render = [&](int64_t dim, int dimIdx) {
    return renderTileTemplateDim(dim, elemTy, blayout, dimIdx);
  };

  std::string vrowTok =
      validShape[0] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[0], 0));
  std::string vcolTok =
      validShape[1] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[1], 1));

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();

  return std::string("Tile<") + tileRoleToken(type.getMemorySpace()) + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " +
         std::to_string(render(rows, 0)) + ", " +
         std::to_string(render(cols, 1)) + ", " +
         tileBufBLayoutToken(configAttr) + ", " + vrowTok + ", " + vcolTok +
         ", " + tileBufSLayoutToken(configAttr) + ", " +
         std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ", " +
         tileBufCompactToken(configAttr) + ">";
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx, PTOArch targetArch) {
    // ---------------------------------------------------------
    // 1. 基本类型 (f32, i32, index)
    // ---------------------------------------------------------
    addConversion([Ctx](FloatType type) -> Type {
      if (type.isFloat8E4M3() || type.isFloat8E4M3FN() ||
          type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ())
        return emitc::OpaqueType::get(Ctx, "float8_e4m3_t");
      if (type.isFloat8E5M2() || type.isFloat8E5M2FNUZ())
        return emitc::OpaqueType::get(Ctx, "float8_e5m2_t");
      if (type.isF32()) return emitc::OpaqueType::get(Ctx, "float");
      if (type.isF16()) return emitc::OpaqueType::get(Ctx, "half");
      if (type.isBF16()) return emitc::OpaqueType::get(Ctx, "bfloat16_t");
      if (type.isF64()) return emitc::OpaqueType::get(Ctx, "double");
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](pto::HiF8Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "hifloat8_t");
    });
    addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
    });
    addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
    });

    addConversion([Ctx](IntegerType type) -> Type {
      if (type.getWidth() == 1)
        return type;

      // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
      // explicitly signed/unsigned; treat signless as signed by default.
      const bool isUnsigned = type.isUnsignedInteger();
      switch (type.getWidth()) {
      case 8:
        return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
      case 16:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint16_t" : "int16_t");
      case 32:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint32_t" : "int32_t");
      case 64:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint64_t" : "int64_t");
      default:
        llvm::errs() << "[Debug] Unsupported IntegerType width: "
                     << type.getWidth() << "\n";
        return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
      }
    });

    addConversion([Ctx](IndexType type) -> Type {
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });

    // vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
    addConversion([Ctx](VectorType type) -> Type {
      if (type.getRank() == 1 && type.getNumElements() == 4 &&
          type.getElementType().isInteger(16))
        return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
      return Type{};
    });

    // ---------------------------------------------------------
    // 2. PTO 特殊类型 (透传或转换)
    // ---------------------------------------------------------
    addConversion([](emitc::OpaqueType type) { return type; });
    addConversion([](emitc::PointerType type) { return type; });

    // ---------------------------------------------------------
    // 2.5 PtrType 转换 (指针类型)
    // ---------------------------------------------------------
    addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType);
      if (!newElemType)
        return std::nullopt;

      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
        llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }

      std::string qualifier = "__gm__";

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      return getEmitCPointerType(Ctx, finalTypeStr);
    });

    addConversion([Ctx](pto::PipeType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "auto");
    });

    addConversion([Ctx](pto::EventIdArrayType type) -> Type {
      std::string tok = "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">";
      return emitc::OpaqueType::get(Ctx, tok);
    });

    // !pto.local_array<D1 x D2 x ... x T> -> !emitc.array<D1 x D2 x ... x T>.
    // Variables of this type render as `T a[D1][D2]...;` in the emitted C++.
    addConversion([this](pto::LocalArrayType type) -> std::optional<Type> {
      Type convertedElem = convertType(type.getElementType());
      if (!convertedElem)
        return std::nullopt;
      return emitc::ArrayType::get(type.getShape(), convertedElem);
    });

    addConversion([Ctx](pto::AsyncSessionType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncSession");
    });

    addConversion([Ctx](pto::AsyncEventType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncEvent");
    });

    addConversion([Ctx](pto::PrefetchAsyncContextType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::PrefetchAsyncContext");
    });

    addConversion([Ctx](pto::TensorViewType type) -> Type {
      return getGlobalTensorOpaqueTypeFromShape(
          Ctx, type.getElementType(), type.getShape());
    });

    addConversion([Ctx](pto::PartitionTensorViewType type) -> Type {
      return getGlobalTensorOpaqueTypeFromShape(
          Ctx, type.getElementType(), type.getShape());
    });

    addConversion([Ctx](pto::TileBufType type) -> std::optional<Type> {
      auto typeString = getEmitCTileTypeString(type);
      if (!typeString)
        return std::nullopt;
      return emitc::OpaqueType::get(Ctx, *typeString);
    });

    // ---------------------------------------------------------
    // 3. MemRef 转换 (Debug 重点)
    // ---------------------------------------------------------
    addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");

      // A. 转换元素类型
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType); 
      if (!newElemType) {
        llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
        return std::nullopt;
      }
      
      // 获取元素类型的字符串
      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
         llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
         return std::nullopt;
      }

      // B. 处理 Memory Space
      std::string qualifier = "";
      Attribute memorySpace = type.getMemorySpace();
      
      if (!memorySpace) {
         qualifier = "__gm__";
      } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
         qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
      } else {
         llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
         qualifier = "__gm__"; // Fallback
      }

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
      
      return getEmitCPointerType(Ctx, finalTypeStr);
    });

    // ---------------------------------------------------------
    // 4. Function & Materialization
    // ---------------------------------------------------------
    addConversion([this](FunctionType type) -> Type {
      SmallVector<Type> inputs;
      if (failed(convertTypes(type.getInputs(), inputs))) return Type{};
      SmallVector<Type> results;
      if (failed(convertTypes(type.getResults(), results))) return Type{};
      return FunctionType::get(type.getContext(), inputs, results);
    });

    auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                              ValueRange Inputs, Location Loc) -> Value {
      if (Inputs.size() != 1) return Value();
      return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
    };

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
    // Needed for region/block signature conversions (e.g. CFG block args).
    addArgumentMaterialization(materializeCast);
  }
};

[[maybe_unused]] static constexpr unsigned kPTOIndexBitWidth =
    32; // keep consistent with IndexType conversion

// Forward declarations (definitions below).
static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth);
static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth);
static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth);
static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth);
static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr);

bool mlir::pto::isSetFFTsPointerLikeType(Type ty) {
  return isEmitCPointerLikeType(ty);
}

static bool tileDataReturnsIntegralAddress(pto::AddressSpace as) {
  return as == pto::AddressSpace::BIAS;
}

static Type getTileDataResultType(MLIRContext *ctx, pto::AddressSpace as,
                                  StringRef elemTok) {
  if (tileDataReturnsIntegralAddress(as))
    return emitc::OpaqueType::get(ctx, "uint64_t");
  return getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
}

Value mlir::pto::materializeTileDataValue(ConversionPatternRewriter &rewriter,
                                      Location loc, Value tile,
                                      pto::AddressSpace as,
                                      StringRef elemTok) {
  auto rawTy = getTileDataResultType(rewriter.getContext(), as, elemTok);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, rawTy, "PTOAS__TILE_DATA",
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{tile})
      .getResult(0);
}

Value mlir::pto::materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                         Location loc, Value addr,
                                         pto::AddressSpace as,
                                         StringRef elemTok) {
  auto *ctx = rewriter.getContext();
  std::string ptrTyStr =
      std::string(addrSpaceQualifier(as)) + " " + elemTok.str() + "*";
  auto ptrTy = getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
  if (isSetFFTsPointerLikeType(addr.getType())) {
    if (addr.getType() == ptrTy)
      return addr;
    return rewriter.create<emitc::CastOp>(loc, ptrTy, addr).getResult();
  }
  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, ptrTyStr)});
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "reinterpret_cast",
                                   ArrayAttr{}, castTyAttr,
                                   ValueRange{addr})
      .getResult(0);
}

static bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SyncSetOp, pto::SyncWaitOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasSetFFTsOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

// Arith/Affine conversion patterns live in PTOToEmitCArith.cpp.

//===----------------------------------------------------------------------===//
// Arith -> EmitC helpers
//===----------------------------------------------------------------------===//

static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "int16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "int32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "int64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "__int128");
  default:
    llvm::errs() << "[Debug] Unsupported signed integer bitwidth: " << bitWidth
                 << "\n";
    return emitc::OpaqueType::get(ctx, "int64_t");
  }
}

static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "uint16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "uint32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "uint64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "unsigned __int128");
  default:
    llvm::errs() << "[Debug] Unsupported unsigned integer bitwidth: "
                 << bitWidth << "\n";
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
}

[[maybe_unused]] static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getSignedIntOpaqueType(ctx, 16);
  case 16:
    return getSignedIntOpaqueType(ctx, 32);
  case 32:
    return getSignedIntOpaqueType(ctx, 64);
  case 64:
    return getSignedIntOpaqueType(ctx, 128);
  default:
    return getSignedIntOpaqueType(ctx, 128);
  }
}

[[maybe_unused]] static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getUnsignedIntOpaqueType(ctx, 16);
  case 16:
    return getUnsignedIntOpaqueType(ctx, 32);
  case 32:
    return getUnsignedIntOpaqueType(ctx, 64);
  case 64:
    return getUnsignedIntOpaqueType(ctx, 128);
  default:
    return getUnsignedIntOpaqueType(ctx, 128);
  }
}

Value mlir::pto::makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal) {
  auto attr = emitc::OpaqueAttr::get(rewriter.getContext(), literal);
  return rewriter.create<emitc::ConstantOp>(loc, type, attr);
}

Value mlir::pto::makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value) {
  return makeEmitCOpaqueConstant(rewriter, loc, type, std::to_string(value));
}

[[maybe_unused]] static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(targetType);
  if (!opaqueTy)
    return failure();

  if (opaqueTy.getValue() == "pto::MrgSortExecutedNumList") {
    auto dense = dyn_cast_or_null<DenseIntElementsAttr>(valueAttr);
    if (!dense)
      return failure();

    auto vecTy = dyn_cast<VectorType>(dense.getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getNumElements() != 4 ||
        !vecTy.getElementType().isInteger(16))
      return failure();

    std::string literal;
    llvm::raw_string_ostream os(literal);
    os << "pto::MrgSortExecutedNumList{";
    bool first = true;
    for (APInt elem : dense.getValues<APInt>()) {
      if (!first)
        os << ", ";
      first = false;
      os << elem.getZExtValue();
    }
    os << "}";
    os.flush();
    return literal;
  }

  return failure();
}

Value mlir::pto::emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src) {
  if (src.getType() == dstType)
    return src;
  return rewriter.createOrFold<emitc::CastOp>(loc, dstType, src);
}

// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.
Value mlir::pto::castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth) {
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return emitCCast(rewriter, loc, uTy, v);
}

//===----------------------------------------------------------------------===//
// pto.mgather lowering -> MGATHER(dst, src, indexes)  (pto-isa)
//===----------------------------------------------------------------------===//

struct PTOMGatherToMGATHER : public OpConversionPattern<pto::MGatherOp> {
  using OpConversionPattern<pto::MGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value mem = peelUnrealized(adaptor.getMem());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    Value memArg = maybeWrapGlobalMemrefAsGlobalTensor(
        rewriter, op.getLoc(), mem, op.getMem().getType(), op.getOperation());

    auto gatherOobTok = [&](pto::GatherOOB mode) -> StringRef {
      switch (mode) {
      case pto::GatherOOB::Undefined:
        return "pto::GatherOOB::Undefined";
      case pto::GatherOOB::Clamp:
        return "pto::GatherOOB::Clamp";
      case pto::GatherOOB::Wrap:
        return "pto::GatherOOB::Wrap";
      case pto::GatherOOB::Zero:
        return "pto::GatherOOB::Zero";
      }
      llvm_unreachable("unknown GatherOOB");
    };

    SmallVector<Attribute, 2> templateArgVec;
    const bool rowCoalesce =
        isRowCoalescedMGatherIndexType(op.getDst().getType(), op.getIdx().getType());
    templateArgVec.push_back(emitc::OpaqueAttr::get(
        ctx, rowCoalesce ? "pto::Coalesce::Row" : "pto::Coalesce::Elem"));
    if (op.getGatherOob() != pto::GatherOOB::Undefined) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, gatherOobTok(op.getGatherOob())));
    }
    ArrayAttr templateArgs = rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MGATHER",
        ArrayAttr{}, templateArgs,
        ValueRange{dst, memArg, idx});

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, dst);
    }
    return success();
  }
};

static std::optional<StringRef> getKernelKindMacro(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return StringRef("__DAV_CUBE__");
  case FunctionKernelKind::Vector:
    return StringRef("__DAV_VEC__");
  }

  llvm_unreachable("unexpected kernel kind");
}

struct FuncToEmitC : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature with the type converter.
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    // Create the EmitC function with the converted signature.
    auto emitcFunc =
        rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), funcType);

    for (const auto &namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName().strref();
      if (name == op.getFunctionTypeAttrName() ||
          name == SymbolTable::getSymbolAttrName() ||
          name == pto::kPTOEntryAttrName ||
          name == pto::kLegacyHACCEntryAttrName ||
          name == "pto.internal.entry")
        continue;
      emitcFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    if (op.isDeclaration()) {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"extern"}));
      rewriter.eraseOp(op);
      return success();
    }

    if (pto::isPTOEntryFunction(op)) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"__global__ AICORE"}));
    } else if (op.isPrivate()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"static", "AICORE"}));
    } else {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"AICORE"}));
    }

    std::optional<StringRef> kernelKindMacro = getKernelKindMacro(op);
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    // Inline the original body, then convert region/block argument types to
    // match the converted signature (also covers CFG blocks introduced by
    // pre-lowering, e.g. scf.while -> cf.br/cf.cond_br).
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(),
                                emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));

    if (failed(rewriter.convertRegionTypes(&emitcFunc.getBody(),
                                           *getTypeConverter(), &entryConv)))
      return failure();

    // Preserve the existing function prologue shape. `kernel_kind` functions are
    // emitted with the same macro guard/reset sequence that used to come from
    // early pto.section wrapping, but only after SCF pre-lowering has finished.
    {
      Block &entryBlock = emitcFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
      if (kernelKindMacro) {
        std::string startMacro = "\n#if defined(" + kernelKindMacro->str() + ")";
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), startMacro);
        if (*kernelKindMacro == "__DAV_VEC__") {
          rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_mask_norm();");
          rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                             "set_vector_mask(-1, -1);");
          if (needsNoSplitGuard)
            rewriter.create<emitc::VerbatimOp>(
                op.getLoc(), "if (get_subblockid() == 0) {");
        }
      }
    }

    if (kernelKindMacro) {
      Block &lastBlock = emitcFunc.getBody().back();
      rewriter.setInsertionPoint(lastBlock.getTerminator());
      if (*kernelKindMacro == "__DAV_VEC__" && needsNoSplitGuard)
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), "}");
      std::string endMacro = "#endif // " + kernelKindMacro->str() + "\n";
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), endMacro);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SubView lowering to GlobalTensor (keep your existing code)
//===----------------------------------------------------------------------===

enum class Role { A, B, C, Unknown };

template <typename MatmulLikeOp>
static std::optional<Role> inferMatmulLikeSubviewRole(MatmulLikeOp op,
                                                      Value buffer) {
  if (op.getLhs() == buffer)
    return Role::A;
  if (op.getRhs() == buffer)
    return Role::B;
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromLoadUser(mlir::pto::TLoadOp load) {
  Value buffer = load.getDst();
  if (!buffer)
    return std::nullopt;
  for (Operation *user : buffer.getUsers()) {
    if (auto matmul = dyn_cast<mlir::pto::TMatmulOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmul, buffer))
        return role;
      continue;
    }
    if (auto matmulAcc = dyn_cast<mlir::pto::TMatmulAccOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmulAcc, buffer))
        return role;
    }
  }
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromUser(Operation *user, Value result) {
  if (auto load = dyn_cast<mlir::pto::TLoadOp>(user))
    return inferSubviewRoleFromLoadUser(load);
  if (auto store = dyn_cast<mlir::pto::TStoreOp>(user)) {
    if (store.getDst() == result)
      return Role::C;
  }
  return std::nullopt;
}

[[maybe_unused]] static Role inferSubviewRole(memref::SubViewOp sv) {
  Value result = sv.getResult();
  for (Operation *user : result.getUsers()) {
    if (auto role = inferSubviewRoleFromUser(user, result))
      return *role;
  }
  return Role::Unknown;
}

// =============================================================================
// 4. MemRef SubView -> Explicit Shape/Stride Construction (Full Implementation)
// =============================================================================
struct SubviewToEmitCPattern : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  // 辅助函数：尝试从 OpFoldResult 中提取静态整数值
  std::optional<int64_t> extractStaticInt(OpFoldResult ofr) const {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getInt();
    } else {
      Value v = ofr.get<Value>();
      if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto iAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
          return iAttr.getInt();
      } else if (auto idxOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return idxOp.value();
      }
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    
    // 获取源 MemRef 类型信息
    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());
    int64_t rank = srcType.getRank();

	    auto elemTypeToString = [&](Type elemTy) -> std::string {
	      if (elemTy.isF16())
	        return "half";
	      if (elemTy.isBF16())
	        return "bfloat16_t";
	      if (elemTy.isF32())
	        return "float";
	      if (elemTy.isF64())
	        return "double";
      if (elemTy.isInteger(8)) {
        if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
          return "int8_t";
        return "uint8_t";
      }
      if (elemTy.isInteger(16)) {
        if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
          return "int16_t";
        return "uint16_t";
      }
      if (elemTy.isInteger(32)) {
        if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
          return "int32_t";
        return "uint32_t";
      }
      if (elemTy.isInteger(64)) {
        return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
      }
      return "float";
    };

    // -------------------------------------------------------------------------
    // Part 1: 指针偏移计算 (Runtime Pointer Arithmetic)
    // -------------------------------------------------------------------------
    
    // 准备类型: unsigned
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    
    // Helper: 创建 unsigned 常量
    auto mkU32 = [&](int64_t v) -> Value {
      return rewriter.create<emitc::ConstantOp>(
          loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(v)));
    };

    // Helper: 将 OpFoldResult 转为 EmitC Value (用于计算)
    auto ofrToEmitCValue = [&](OpFoldResult ofr) -> Value {
      if (auto v = ofr.dyn_cast<Value>()) {
        Value rv = rewriter.getRemappedValue(v);
        // 如果类型不匹配，插入 Cast
        if (rv.getType() != u32Ty)
             return rewriter.create<emitc::CastOp>(loc, u32Ty, rv).getResult();
        return rv;
      }
      if (auto attr = ofr.dyn_cast<Attribute>()) {
         if (auto ia = dyn_cast<IntegerAttr>(attr))
             return mkU32(ia.getValue().getSExtValue());
      }
      return mkU32(0);
    };

    // 1. 获取 Source 的 Strides (支持动态 Stride 收集)
    SmallVector<OpFoldResult> sourceStrides;

    if (auto rc = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        sourceStrides = rc.getMixedStrides();
    } else {
        SmallVector<int64_t> strideInts;
        int64_t offset = ShapedType::kDynamic;
        bool useTypeStrides = succeeded(getStridesAndOffset(srcType, strideInts, offset));
        (void)offset;
        if (useTypeStrides) {
          for (int64_t s : strideInts) {
            if (s == ShapedType::kDynamic)
              useTypeStrides = false;
          }
        }
        if (useTypeStrides) {
            for (int64_t s : strideInts) {
                sourceStrides.push_back(rewriter.getIndexAttr(s));
            }
        } else {
            // Fallback: Compact Layout
            auto shape = srcType.getShape();
            int64_t current = 1;
            sourceStrides.resize(rank);
            for (int i = rank - 1; i >= 0; --i) {
                sourceStrides[i] = rewriter.getIndexAttr(current);
                if (shape[i] != ShapedType::kDynamic) current *= shape[i];
            }
        }
    }

    // 2. 计算运行时 Offset
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = adaptor.getOffsets();
    int dynOffIdx = 0;
    Value totalOffset = mkU32(0);

    for (int i = 0; i < rank; ++i) {
        // A. 获取 Offset
        Value offVal;
        if (staticOffsets[i] == ShapedType::kDynamic) {
            Value rawDyn = dynamicOffsets[dynOffIdx++];
            offVal = rewriter.create<emitc::CastOp>(loc, u32Ty, rawDyn);
        } else {
            offVal = mkU32(staticOffsets[i]);
        }

        // B. 获取 Stride (用于指针计算)
        Value strideVal = mkU32(1);
        if (i < (int)sourceStrides.size()) {
            strideVal = ofrToEmitCValue(sourceStrides[i]);
        }

        // C. 累加
        Value term = rewriter.create<emitc::MulOp>(loc, u32Ty, offVal, strideVal);
        totalOffset = rewriter.create<emitc::AddOp>(loc, u32Ty, totalOffset, term);
    }

    // 3. 生成新指针
    //
    // NOTE: Some toolchains may materialize kernel pointer params as `void*` even
    // when the underlying element type is i16. Pointer arithmetic on `void*`
    // is ill-formed in C++, so we explicitly cast to a typed pointer for i16.
    Value sourcePtr = adaptor.getSource();
    Value tileCandidate = sourcePtr;
    if (auto castOp = sourcePtr.getDefiningOp<emitc::CastOp>()) {
      tileCandidate = castOp.getOperand();
    } else if (auto uc =
                   sourcePtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      tileCandidate = uc.getOperand(0);
    }
    if (auto ot = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
      auto tyStr = ot.getValue();
      if (tyStr.find("Tile<") != std::string::npos ||
          tyStr.find("ConvTile<") != std::string::npos) {
        std::string elemTok = elemTypeToString(srcType.getElementType());
        pto::AddressSpace as = pto::AddressSpace::GM;
        if (auto asAttr =
                dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace()))
          as = asAttr.getAddressSpace();
        sourcePtr =
            materializeTileDataValue(rewriter, loc, tileCandidate, as, elemTok);
        if (tileDataReturnsIntegralAddress(as))
          sourcePtr =
              materializeAddressAsPointer(rewriter, loc, sourcePtr, as, elemTok);
      }
    }
    Value newPtr;
    {
      auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
      Type elemTy = resTy.getElementType();
      if (elemTy.isInteger(16)) {
        std::string castElemTypeStr = "int16_t";
        if (cast<IntegerType>(elemTy).isUnsigned())
          castElemTypeStr = "uint16_t";

        std::string qualifier = "__gm__";
        if (Attribute ms = srcType.getMemorySpace()) {
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(ms)) {
            qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
          }
        }

        auto typedPtrTy = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, qualifier + " " + castElemTypeStr));
        Value typedSourcePtr = rewriter.create<emitc::CastOp>(loc, typedPtrTy, sourcePtr);
        newPtr = rewriter.create<emitc::AddOp>(loc, typedPtrTy, typedSourcePtr, totalOffset);
      } else {
        newPtr = rewriter.create<emitc::AddOp>(loc, sourcePtr.getType(), sourcePtr, totalOffset);
      }
    }


    // -------------------------------------------------------------------------
    // Part 2: For non-GM memrefs, keep pointer (no GlobalTensor).
    // -------------------------------------------------------------------------
    bool isGlobal = true;
    if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
    }
    if (!isGlobal) {
      Type dstTy = getTypeConverter()->convertType(op.getType());
      if (!dstTy)
        return failure();
      if (newPtr.getType() != dstTy)
        newPtr = rewriter.create<emitc::CastOp>(loc, dstTy, newPtr);
      rewriter.replaceOp(op, newPtr);
      return success();
    }

    // -------------------------------------------------------------------------
    // Part 3: 生成 GlobalTensor 类型 (Shape/Stride Template Generation)
    // -------------------------------------------------------------------------
    
    // When emitting C++ with `declareVariablesAtTop`, value declarations are
    // hoisted before body statements. Avoid introducing local `using` aliases
    // for templated types (Shape/Stride/GlobalTensor) because those aliases
    // would appear after the hoisted declarations and break compilation
    // (`unknown type name`).
    //
    // Instead, use the fully spelled template types as EmitC opaque types.

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    
    // 1. 解析具体元素类型
    std::string elemTypeStr = getElemTypeStringForGT(resTy.getElementType());

    // 2. 生成 Shape 模板参数，之后会右对齐有效维度并补齐到 5 维（高维填 1）
    SmallVector<int64_t> shapeParamsVec;
    SmallVector<Value> sizeValues; // 每个维度对应的运行时 size（统一为 unsigned）
    auto resShape = resTy.getShape();
    auto mixedSizes = op.getMixedSizes();
    sizeValues.reserve(rank);
    for (int i = 0; i < resTy.getRank(); ++i) {
      if (resShape[i] == ShapedType::kDynamic) {
        shapeParamsVec.push_back(-1);
      } else {
        shapeParamsVec.push_back(resShape[i]);
      }
      // size 值：优先从 op.getMixedSizes() 取（可动态/静态），否则退化为类型里的静态 shape。
      if (i < (int)mixedSizes.size())
        sizeValues.push_back(ofrToEmitCValue(mixedSizes[i]));
      else
        sizeValues.push_back(
            mkU32(resShape[i] == ShapedType::kDynamic ? 1 : resShape[i]));
    }

    // 3. 生成 Stride 模板参数 + 运行时 stride 值（考虑 subview step）
    SmallVector<int64_t> strideTemplateVec;
    SmallVector<Value> strideValues; // 每个维度对应的运行时 stride（统一为 unsigned）
    strideTemplateVec.reserve(rank);
    strideValues.reserve(rank);
    auto subViewSteps = op.getMixedStrides();
    for (int i = 0; i < rank; ++i) {
      OpFoldResult srcStrideOfr =
          (i < (int)sourceStrides.size()) ? sourceStrides[i]
                                          : rewriter.getIndexAttr(1);
      OpFoldResult stepOfr = (i < (int)subViewSteps.size())
                                 ? subViewSteps[i]
                                 : rewriter.getIndexAttr(1);

      auto srcStatic = extractStaticInt(srcStrideOfr);
      auto stepStatic = extractStaticInt(stepOfr);
      if (srcStatic && stepStatic) {
        int64_t finalStride = (*srcStatic) * (*stepStatic);
        strideTemplateVec.push_back(finalStride);
        strideValues.push_back(mkU32(finalStride));
        continue;
      }

      strideTemplateVec.push_back(-1);
      Value srcV = ofrToEmitCValue(srcStrideOfr);
      Value stepV = ofrToEmitCValue(stepOfr);
      // 尽量避免乘以 1 生成冗余指令
      if (stepStatic && *stepStatic == 1)
        strideValues.push_back(srcV);
      else if (srcStatic && *srcStatic == 1)
        strideValues.push_back(stepV);
      else
        strideValues.push_back(
            rewriter.create<emitc::MulOp>(loc, u32Ty, srcV, stepV));
    }

    // 3.1 右对齐到 5 维：shape 补 1；已有维度继承原 stride；
    //      被补出来的高维按“紧密升维”规则连续推导：stride[i] = shape[i+1] * stride[i+1]
    SmallVector<int64_t, 5> finalShape;
    SmallVector<int64_t, 5> finalStride;
    buildGlobalTensorShapeAndStride(shapeParamsVec, strideTemplateVec,
                                    finalShape, finalStride);
    Value oneU32 = mkU32(1);
    SmallVector<Value, 5> finalShapeValues(5, oneU32);
    SmallVector<Value, 5> finalStrideValues(5, oneU32);
    int shift = 5 - rank;

    // 先放入原始 shape/stride（保持用户提供的值）
    for (int i = 0; i < rank && i < 5; ++i) {
      finalShapeValues[shift + i] = sizeValues[i];
      finalStrideValues[shift + i] = strideValues[i];
    }

    // 从低维到高维倒推补齐 stride（仅对补出来的前置维度生效）
    for (int i = 3; i >= 0; --i) {
      // 如果该维已由原始 rank 覆盖，则保持原值
      if (i >= shift)
        continue;
      if (finalStride[i] != -1) {
        finalStrideValues[i] = mkU32(finalStride[i]);
        continue;
      }
      // 动态推导：stride[i] = shape[i+1] * stride[i+1]
      if (finalShape[i + 1] == 1) {
        finalStrideValues[i] = finalStrideValues[i + 1];
      } else {
        finalStrideValues[i] = rewriter.create<emitc::MulOp>(
            loc, u32Ty, finalShapeValues[i + 1], finalStrideValues[i + 1]);
      }
    }

    std::string shapeParams = joinIntTemplateParams(finalShape);
    std::string strideParams = joinIntTemplateParams(finalStride);

    // Spelled-out C++ types.
    std::string shapeCppType = "pto::Shape<" + shapeParams + ">";
    std::string strideCppType = "pto::Stride<" + strideParams + ">";

    // 3.0 Layout: prefer the attribute from InferPTOLayout; only fall back to
    // local inference when the pass is disabled.
    std::string layoutEnum = "pto::Layout::ND";
    if (auto layout = resolveLayoutForGlobalTensor(op, op.getSource())) {
      layoutEnum = layoutToEmitCString(*layout);
    } else {
      bool allStatic =
          llvm::all_of(finalShape, [](int64_t value) { return value != -1; }) &&
          llvm::all_of(finalStride, [](int64_t value) { return value != -1; });

      int layoutTag = 0; // ND
      auto elemBytes = 4; // default float
      if (elemTypeStr.find("half") != std::string::npos ||
          elemTypeStr.find("f16") != std::string::npos ||
          elemTypeStr.find("bf16") != std::string::npos)
        elemBytes = 2;
      else if (elemTypeStr.find("double") != std::string::npos ||
               elemTypeStr.find("f64") != std::string::npos)
        elemBytes = 8;

      if (allStatic) {
        if (finalShape[2] == 16 &&
            finalShape[2] * finalShape[3] * elemBytes == 512 &&
            finalStride[4] == 1 && finalStride[3] == finalShape[4]) {
          layoutTag = 2; // NZ
        } else {
          bool isRow = finalStride[4] == 1;
          for (int i = 3; i >= 0; --i)
            isRow &= (finalStride[i] ==
                      multiplyOrDynamic(finalStride[i + 1], finalShape[i + 1]));
          bool isCol = finalStride[0] == 1;
          for (int i = 0; i < 4; ++i)
            isCol &= (finalStride[i + 1] ==
                      multiplyOrDynamic(finalStride[i], finalShape[i]));
          if (isCol)
            layoutTag = 1; // DN
          else
            layoutTag = isRow ? 0 : 0; // fallback ND
        }
      }

      if (layoutTag == 1)
        layoutEnum = "pto::Layout::DN";
      else if (layoutTag == 2)
        layoutEnum = "pto::Layout::NZ";
    }
    // GlobalTensor takes a Layout non-type template parameter; directly use the
    // enum constant.


    // -------------------------------------------------------------------------
    // Part 3: 显式对象实例化 (Explicit Object Instantiation)
    // -------------------------------------------------------------------------

    // A. Instantiate Shape object.
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeCppType);
    SmallVector<Value> shapeArgs;
    // 从 adaptor.getSizes() 获取 subview 的所有 dynamic sizes
    for (Value dynSize : adaptor.getSizes()) {
        shapeArgs.push_back(dynSize);
    }
    
    auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, 
        shapeTypeOpaque, // 返回类型
        shapeCppType,    // 调用的“函数名”即类名构造函数
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{}, 
        /*operands=*/ValueRange(shapeArgs)
    );
    
    // B. Instantiate Stride object.
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideCppType);
    // 仅传入动态 stride 维度对应的值，匹配 pto::Stride 的 N-parameter ctor（并满足其 static_assert）。
    SmallVector<Value> strideCtorArgs;
    strideCtorArgs.reserve(5);
    for (int i = 0; i < 5; ++i) {
      if (finalStride[i] == -1)
        strideCtorArgs.push_back(finalStrideValues[i]);
    }
    auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, strideTypeOpaque, strideCppType,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(strideCtorArgs));

    // C. Instantiate GlobalTensor object (ptr + shape + stride).
    std::string gtCppType = "GlobalTensor<" + elemTypeStr + ", " + shapeCppType +
                            ", " + strideCppType + ", " + layoutEnum + ">";
    auto gtType = emitc::OpaqueType::get(ctx, gtCppType);

    // 准备构造参数: [ptr, shape_instance, stride_instance]
    SmallVector<Value> gtConstructorArgs;
    gtConstructorArgs.push_back(newPtr);
    gtConstructorArgs.push_back(shapeInstOp.getResult(0)); // 拿到 shape_inst 的 SSA Value
    gtConstructorArgs.push_back(strideInstOp.getResult(0)); // 拿到 stride_inst 的 SSA Value

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        gtType, 
        gtCppType,
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(gtConstructorArgs)
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

std::string mlir::pto::getElemTypeStringForGT(Type elemTy) {
  return getEmitCScalarTypeToken(elemTy);
}

static bool hasStaticShape(MemRefType mrTy) {
  return llvm::none_of(mrTy.getShape(), [](int64_t dim) {
    return dim == ShapedType::kDynamic;
  });
}

static bool getStaticMemrefLayout(MemRefType mrTy, SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset) {
  if (failed(getStridesAndOffset(mrTy, strides, offset))) {
    strides.clear();
    int64_t stride = 1;
    ArrayRef<int64_t> shape = mrTy.getShape();
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides.push_back(stride);
      stride *= shape[i];
    }
    std::reverse(strides.begin(), strides.end());
    offset = 0;
  }
  return offset != ShapedType::kDynamic &&
         llvm::none_of(strides, [](int64_t strideValue) {
           return strideValue == ShapedType::kDynamic;
         });
}

Value mlir::pto::applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
                                     Location loc, Value basePtr,
                                     int64_t offset) {
  if (offset == 0)
    return basePtr;
  auto *ctx = rewriter.getContext();
  Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
  auto offVal = rewriter.create<emitc::ConstantOp>(
      loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(offset)));
  return rewriter.create<emitc::AddOp>(loc, basePtr.getType(), basePtr, offVal);
}

static int getGlobalTensorElementBytes(Type elemTy) {
  return static_cast<int>(getPTOStorageElemByteSize(elemTy));
}

static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return -1;
  return lhs * rhs;
}

void mlir::pto::buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D) {
  shape5D.assign(5, 1);
  stride5D.assign(5, 1);
  int rank = static_cast<int>(shape.size());
  int shift = 5 - rank;
  for (int i = 0; i < rank && i < 5; ++i) {
    shape5D[shift + i] = shape[i];
    stride5D[shift + i] = strides[i];
  }
  for (int i = 3; i >= 0; --i) {
    if (i >= shift)
      continue;
    stride5D[i] = multiplyOrDynamic(shape5D[i + 1], stride5D[i + 1]);
  }
}

std::string mlir::pto::joinIntTemplateParams(ArrayRef<int64_t> values) {
  std::string result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      result += ", ";
    result += std::to_string(values[i]);
  }
  return result;
}

SmallVector<int64_t> mlir::pto::buildRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = running;
    running = multiplyOrDynamic(running, shape[i]);
  }
  return strides;
}

static std::string getGlobalTensorTypeStringFromShape(Type elemTy,
                                                      ArrayRef<int64_t> shape,
                                                      StringRef layoutEnum) {
  SmallVector<int64_t> strides = buildRowMajorStrides(shape);
  return getGlobalTensorTypeStringFromShapeAndStrides(elemTy, shape, strides,
                                                      layoutEnum);
}

std::string mlir::pto::getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum) {
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  std::string elemTypeStr = getElemTypeStringForGT(elemTy);
  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  return "GlobalTensor<" + elemTypeStr + ", " + shapeType + ", " +
         strideType + ", " + layoutEnum.str() + ">";
}

static emitc::OpaqueType getGlobalTensorOpaqueTypeFromShape(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum) {
  return emitc::OpaqueType::get(
      ctx, getGlobalTensorTypeStringFromShape(elemTy, shape, layoutEnum));
}

static std::string inferFallbackGlobalTensorLayout(ArrayRef<int64_t> shape5D,
                                                   ArrayRef<int64_t> stride5D,
                                                   Type elemTy) {
  int elemBytes = getGlobalTensorElementBytes(elemTy);
  if (elemBytes == 0)
    return "pto::Layout::ND";
  if (shape5D[2] == 16 && multiplyOrDynamic(shape5D[2], shape5D[3]) * elemBytes == 512 &&
      stride5D[4] == 1 && stride5D[3] == shape5D[4]) {
    return "pto::Layout::NZ";
  }

  bool isRowMajor = stride5D[4] == 1;
  for (int i = 3; i >= 0 && isRowMajor; --i)
    isRowMajor = stride5D[i] == multiplyOrDynamic(stride5D[i + 1], shape5D[i + 1]);

  bool isColMajor = stride5D[0] == 1;
  for (int i = 0; i < 4 && isColMajor; ++i)
    isColMajor = stride5D[i + 1] == multiplyOrDynamic(stride5D[i], shape5D[i]);

  if (isColMajor)
    return "pto::Layout::DN";
  return isRowMajor ? "pto::Layout::ND" : "pto::Layout::ND";
}

static std::string resolveGlobalTensorLayout(Operation *anchor, Value basePtr,
                                             ArrayRef<int64_t> shape5D,
                                             ArrayRef<int64_t> stride5D,
                                             Type elemTy) {
  if (auto layout = resolveLayoutForGlobalTensor(anchor, basePtr))
    return layoutToEmitCString(*layout);
  return inferFallbackGlobalTensorLayout(shape5D, stride5D, elemTy);
}

struct GlobalTensorTypeNames {
  std::string shapeTypeName;
  std::string strideTypeName;
  std::string tensorTypeName;
  std::string layoutConstName;
};

static GlobalTensorTypeNames getGlobalTensorTypeNames(Operation *anchor) {
  std::string suffix = "_" + std::to_string(reinterpret_cast<uintptr_t>(anchor));
  return {
      "GTShape" + suffix,
      "GTStride" + suffix,
      "GT" + suffix,
      "GT" + suffix + "_layout",
  };
}
Value mlir::pto::buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy,
                                         Operation *anchor) {
  auto *ctx = rewriter.getContext();

  ArrayRef<int64_t> shape = mrTy.getShape();
  if (!hasStaticShape(mrTy))
    return Value();

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!getStaticMemrefLayout(mrTy, strides, offset))
    return Value();

  Value ptr = applyStaticMemrefOffset(rewriter, loc, basePtr, offset);
  GlobalTensorTypeNames names = getGlobalTensorTypeNames(anchor);
  std::string elemTypeStr = getElemTypeStringForGT(mrTy.getElementType());
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.shapeTypeName + " = pto::Shape<" +
               joinIntTemplateParams(shape5D) + ">;");
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.strideTypeName + " = pto::Stride<" +
               joinIntTemplateParams(stride5D) + ">;");

  std::string layoutEnum = resolveGlobalTensorLayout(
      anchor, basePtr, shape5D, stride5D, mrTy.getElementType());
  rewriter.create<emitc::VerbatimOp>(loc, "constexpr pto::Layout " +
                                              names.layoutConstName + " = " +
                                              layoutEnum + ";");

  auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, names.shapeTypeName);
  auto strideTypeOpaque = emitc::OpaqueType::get(ctx, names.strideTypeName);
  auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, shapeTypeOpaque, names.shapeTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});
  auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, strideTypeOpaque, names.strideTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.tensorTypeName + " = GlobalTensor<" + elemTypeStr +
               ", " + names.shapeTypeName + ", " + names.strideTypeName +
               ", " + names.layoutConstName + ">;");
  auto gtType = emitc::OpaqueType::get(ctx, names.tensorTypeName);

  SmallVector<Value> gtArgs;
  gtArgs.push_back(ptr);
  gtArgs.push_back(shapeInstOp.getResult(0));
  gtArgs.push_back(strideInstOp.getResult(0));

  auto gtInst = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, names.tensorTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange(gtArgs));

  return gtInst.getResult(0);
}

static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor) {
  auto mrTy = dyn_cast<MemRefType>(originalType);
  if (!mrTy)
    return loweredValue;

  bool isGlobal = true;
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  if (!isGlobal)
    return loweredValue;

  if (Value gt =
          buildGlobalTensorFromMemref(rewriter, loc, loweredValue, mrTy, anchor))
    return gt;
  return loweredValue;
}

Value mlir::pto::castToGMBytePointer(ConversionPatternRewriter &rewriter,
                                 Location loc, Value value) {
  auto *ctx = rewriter.getContext();
  auto targetTy =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ uint8_t"));
  if (value.getType() == targetTy)
    return value;

  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "__gm__ uint8_t*")});
  if (isSetFFTsPointerLikeType(value.getType())) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, targetTy, "reinterpret_cast",
                                     ArrayAttr{}, castTyAttr,
                                     ValueRange{value})
        .getResult(0);
  }
  return rewriter.create<emitc::CastOp>(loc, targetTy, value).getResult();
}

Value mlir::pto::materializeTensorViewDataPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value value,
    Type sourceType) {
  auto tvTy = dyn_cast<pto::TensorViewType>(sourceType);
  if (!tvTy)
    return value;

  auto *ctx = rewriter.getContext();
  std::string elemTypeStr = getElemTypeStringForGT(tvTy.getElementType());
  auto ptrTy = emitc::PointerType::get(
      emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "PTOAS__GLOBAL_TENSOR_DATA",
                                   ArrayAttr{}, ArrayAttr{}, ValueRange{value})
      .getResult(0);
}

static std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string blTok = "BLayout::RowMajor";
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
    if (static_cast<int32_t>(blAttr.getValue()) == 1)
      blTok = "BLayout::ColMajor";
  }
  return blTok;
}

static std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string slTok = "SLayout::NoneBox";
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
    int32_t slVal = static_cast<int32_t>(slAttr.getValue());
    slTok = (slVal == 1) ? "SLayout::RowMajor"
                         : (slVal == 2) ? "SLayout::ColMajor"
                                        : "SLayout::NoneBox";
  }
  return slTok;
}

static std::string tileBufPadToken(pto::TileBufConfigAttr configAttr) {
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
  return padTok;
}

pto::BLayout mlir::pto::getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr) {
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    return blAttr.getValue();
  return pto::BLayout::RowMajor;
}

int64_t mlir::pto::renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx) {
  assert(dimIdx >= 0 && dimIdx < 2 &&
         "renderTileTemplateDim expects a rank-2 rows/cols dimension index");
  if (rawDim == ShapedType::kDynamic)
    return rawDim;
  if (!pto::isPTOFloat4PackedType(elemTy))
    return rawDim;
  int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

FailureOr<Value> mlir::pto::buildAsyncScratchTileValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalScratch,
    Value emittedScratch) {
  Value scratch = peelUnrealized(emittedScratch);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(scratch.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return scratch;
  }

  auto memTy = dyn_cast<MemRefType>(originalScratch.getType());
  if (!memTy)
    return failure();

  ArrayRef<int64_t> shape = memTy.getShape();
  if (!memTy.hasStaticShape() || shape.empty() || shape.size() > 2)
    return failure();

  int64_t rows = shape.size() == 1 ? 1 : shape[0];
  int64_t cols = shape.size() == 1 ? shape[0] : shape[1];

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  if (auto bind = originalScratch.getDefiningOp<pto::BindTileOp>()) {
    configAttr = bind.getConfig();
  } else if (auto cast = originalScratch.getDefiningOp<pto::PointerCastOp>()) {
    if (auto config = cast.getConfig())
      configAttr = *config;
  }

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();

  Type elemTy = memTy.getElementType();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  int64_t templateRows = renderTileTemplateDim(rows, elemTy, blayout, 0);
  int64_t templateCols = renderTileTemplateDim(cols, elemTy, blayout, 1);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  std::string tileTypeStr =
      "Tile<TileType::Vec, " + elemTypeStr + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufBLayoutToken(configAttr) + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufSLayoutToken(configAttr) + ", " +
      std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ">";

  Value tile = rewriter
                   .create<emitc::VariableOp>(
                       loc, emitc::OpaqueType::get(ctx, tileTypeStr),
                       emitc::OpaqueAttr::get(ctx, ""))
                   .getResult();
  auto addr = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
  Value scratchAddr =
      rewriter
          .create<emitc::CallOpaqueOp>(loc, emitc::OpaqueType::get(ctx, "uint64_t"),
                                       "reinterpret_cast", ArrayAttr{}, addr,
                                       ValueRange{scratch})
          .getResult(0);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, scratchAddr});
  return tile;
}

//===----------------------------------------------------------------------===//
// pto.pointer_cast lowering
//===----------------------------------------------------------------------===
struct PTOMScatterToMSCATTER : public OpConversionPattern<pto::MScatterOp> {
  using OpConversionPattern<pto::MScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value mem = peelUnrealized(adaptor.getMem());

    Value memArg = maybeWrapGlobalMemrefAsGlobalTensor(
        rewriter, op.getLoc(), mem, op.getMem().getType(), op.getOperation());

    auto scatterAtomicTok = [&](pto::ScatterAtomicOp atomic) -> StringRef {
      switch (atomic) {
      case pto::ScatterAtomicOp::None:
        return "pto::ScatterAtomicOp::None";
      case pto::ScatterAtomicOp::Add:
        return "pto::ScatterAtomicOp::Add";
      case pto::ScatterAtomicOp::Max:
        return "pto::ScatterAtomicOp::Max";
      case pto::ScatterAtomicOp::Min:
        return "pto::ScatterAtomicOp::Min";
      }
      llvm_unreachable("unknown ScatterAtomicOp");
    };
    auto scatterOobTok = [&](pto::ScatterOOB mode) -> StringRef {
      switch (mode) {
      case pto::ScatterOOB::Undefined:
        return "pto::ScatterOOB::Undefined";
      case pto::ScatterOOB::Skip:
        return "pto::ScatterOOB::Skip";
      case pto::ScatterOOB::Clamp:
        return "pto::ScatterOOB::Clamp";
      case pto::ScatterOOB::Wrap:
        return "pto::ScatterOOB::Wrap";
      }
      llvm_unreachable("unknown ScatterOOB");
    };

    SmallVector<Attribute, 3> templateArgVec;
    const bool rowCoalesce =
        isRowCoalescedMGatherIndexType(op.getSrc().getType(), op.getIdx().getType());
    templateArgVec.push_back(emitc::OpaqueAttr::get(
        ctx, rowCoalesce ? "pto::Coalesce::Row" : "pto::Coalesce::Elem"));
    if (op.getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
        op.getScatterOob() != pto::ScatterOOB::Undefined) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, scatterAtomicTok(op.getScatterAtomicOp())));
      if (op.getScatterOob() != pto::ScatterOOB::Undefined)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, scatterOobTok(op.getScatterOob())));
    }
    ArrayAttr templateArgs = rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MSCATTER",
        ArrayAttr{}, templateArgs,
        ValueRange{memArg, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};
static void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx,
                                       DataFlowSolver &solver,
                                       PTOArch targetArch) {
  (void)solver;
  populatePTOToEmitCArithPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCRuntimeOpPatterns(patterns, typeConverter, ctx, targetArch);
  populatePTOToEmitCMemoryOpPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCTilePatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCSimpleOpPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCTileMaterializationPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCSyncPatterns(patterns, typeConverter, ctx, targetArch);
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
  populatePTOToEmitCKernelOpPatterns(patterns, typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
  populatePTOToEmitCCommPatterns(patterns, typeConverter, ctx, targetArch);
  populatePTOToEmitCControlFlowPatterns(patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct EmitPTOManualPass
    : public PassWrapper<EmitPTOManualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPTOManualPass)

  PTOArch targetArch;

  EmitPTOManualPass() : targetArch(PTOArch::A3) {}

  explicit EmitPTOManualPass(PTOArch arch) : targetArch(arch) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    mlir::cf::ControlFlowDialect, mlir::pto::PTODialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Start PTOToEmitC Pass\n");
    MLIRContext *ctx = &getContext();
    ModuleOp mop = getOperation();

    if (failed(pto::validatePTOEntryFunctions(mop)))
      return signalPassFailure();
    pto::annotatePTOEntryFunctions(mop);

    // A3 requires explicit FFTS base setup for inter-core sync ops.
    if (targetArch == PTOArch::A3) {
      bool hasMissingSetFFTs = false;
      for (auto func : mop.getOps<func::FuncOp>()) {
        if (!hasInterCoreSyncOp(func))
          continue;
        if (hasSetFFTsOp(func))
          continue;
        hasMissingSetFFTs = true;
        func.emitError()
            << "A3 inter-core sync requires explicit `pto.set_ffts` in the "
               "same function when using `pto.sync.set`/`pto.sync.wait`";
      }
      if (hasMissingSetFFTs)
        return signalPassFailure();
    }

        bool needsEventIdArrayHelper = false;
        bool needsTRandomHelper = false;
        bool needsGlobalTensorDataHelper = false;
        bool needsCommInclude = false;
        mop.walk([&](Operation *op) {
          if (isa<mlir::pto::DeclareEventIdArrayOp>(op))
            needsEventIdArrayHelper = true;
          if (isa<mlir::pto::TRandomOp>(op))
            needsTRandomHelper = true;
          if (isa<mlir::pto::PartitionViewOp>(op))
            needsGlobalTensorDataHelper = true;
          if (isa<mlir::pto::BuildAsyncSessionOp, mlir::pto::TPutAsyncOp,
                  mlir::pto::TGetAsyncOp, mlir::pto::TPrefetchAsyncOp,
                  mlir::pto::WaitAsyncEventOp, mlir::pto::TestAsyncEventOp,
                  mlir::pto::TPutOp,
                  mlir::pto::TGetOp, mlir::pto::TNotifyOp, mlir::pto::TWaitOp,
                  mlir::pto::TTestOp, mlir::pto::TBroadcastOp,
                  mlir::pto::CommTGatherOp, mlir::pto::CommTScatterOp,
                  mlir::pto::TReduceOp>(op))
            needsCommInclude = true;
        });

		    // 1. 插入头文件
	    auto loc = mop->getLoc();
	    OpBuilder builder(ctx);
	    builder.setInsertionPointToStart(mop.getBody());
	    builder.create<emitc::IncludeOp>(
	        loc, "pto/pto-inst.hpp", /*is_standard_include=*/false);
        if (needsCommInclude) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_M
#endif
)cpp"));
	      builder.create<emitc::IncludeOp>(
	          loc, "pto/comm/pto_comm_inst.hpp", /*is_standard_include=*/false);
        }
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr("using namespace pto;"));
        if (needsGlobalTensorDataHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}
)cpp"));
        }
        if (needsEventIdArrayHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <int N>
struct PTOAS_EventIdArray {
  static_assert(N > 0, "PTOAS_EventIdArray requires a positive static size");
  int32_t data[N] = {};

  AICORE inline int32_t &operator[](int32_t idx) { return data[idx]; }
  AICORE inline const int32_t &operator[](int32_t idx) const { return data[idx]; }
};
)cpp"));
        }
        if (needsTRandomHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <uint16_t Rounds, typename DstTile>
static AICORE inline void PTOAS__TRANDOM(
    DstTile &dst, uint32_t key0, uint32_t key1, uint32_t counter0,
    uint32_t counter1, uint32_t counter2, uint32_t counter3) {
  TRandomKey key = {key0, key1};
  TRandomCounter counter = {counter0, counter1, counter2, counter3};
  TRANDOM<Rounds>(dst, key, counter);
}
)cpp"));
        }
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr(R"cpp(
enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}
)cpp"));
	    // Only inject the bitcast helper when we actually lower ops that need it
	    // (e.g. arith.bitcast or arith.maximumf/minimumf tie-breaking on zeros).
	    bool needsBitcastHelper = false;
	    mop.walk([&](Operation *op) {
	      if (isa<arith::BitcastOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
	        needsBitcastHelper = true;
	        return WalkResult::interrupt();
	      }
	      return WalkResult::advance();
	    });
	    if (needsBitcastHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
		template <typename To, typename From>
		static inline To ptoas_bitcast(From from) {
		  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
		  To to;
		  __builtin_memcpy(&to, &from, sizeof(To));
		  return to;
		}
		)cpp"));
	    }

	    // 1.5 Pre-lower SCF constructs not handled by SCFToEmitC.
    if (failed(runPTOToEmitCSCFPreLowering(mop, ctx)))
      return signalPassFailure();

    PTOToEmitCTypeConverter typeConverter(ctx, targetArch);

    // 2. Pre-convert SCF structural op types (e.g. scf.if/scf.for results)
    // using the same type converter. This avoids creating emitc.variable with
    // unsupported types such as memref.
    {
      RewritePatternSet scfTypePatterns(ctx);
      ConversionTarget scfTypeTarget(*ctx);
      scf::populateSCFStructuralTypeConversionsAndLegality(
          typeConverter, scfTypePatterns, scfTypeTarget);
      scfTypeTarget.markUnknownOpDynamicallyLegal(
          [](Operation *) { return true; });

      if (failed(applyPartialConversion(mop, scfTypeTarget,
                                        std::move(scfTypePatterns)))) {
        mop.emitError("failed to reconcile SCF structural types");
        return signalPassFailure();
      }
    }

    // 3. 配置转换目标
    ConversionTarget target(*ctx);

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>(); 
    
    // If we introduced CFG branches (e.g. from scf.while), make sure they are
    // updated to use legalized operand types.
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                  typeConverter);
        });

    // [关键] 允许 Cast 存在，最后统一清理
    target.addLegalOp<UnrealizedConversionCastOp>(); 

    target.addIllegalOp<func::ReturnOp>();
    target.addIllegalOp<func::FuncOp>(); 
    target.addIllegalOp<func::CallOp>();

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();

    auto solver = std::make_unique<DataFlowSolver>();
    solver->load<dataflow::DeadCodeAnalysis>();
    solver->load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(ctx);
    populatePTOToEmitCPatterns(patterns, typeConverter, ctx, *solver, targetArch);

    // 4. 执行转换
    if (failed(applyPartialConversion(mop, target, std::move(patterns)))) {
      llvm::errs() << "Conversion FAILED! Rolling back executed.\n";
      return signalPassFailure();
    }

    // =========================================================================
    // 5. [终极清理] 
    // 顺序至关重要：
    // Step A: 先移除所有 Cast，让 Loop 的 Operand 类型变成底层类型 (如 int32)
    // Step B: 再根据新的 Operand 类型，修复 Loop IV 的类型
    // =========================================================================
    
    // --- Step A: 清理 UnrealizedConversionCastOp ---
    // Prefer dropping redundant/unused casts; otherwise lower to emitc.cast
    // so the C++ emitter can print it.
    auto isEmitCTileLikeType = [](Type ty) {
      auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
      if (!opaqueTy)
        return false;
      StringRef value = opaqueTy.getValue();
      return value.contains("Tile<") || value.contains("ConvTile<");
    };

    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    bool castCleanupFailed = false;
    mop.walk([&](UnrealizedConversionCastOp cast) {
      if (castCleanupFailed)
        return;

      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast.emitError() << "unsupported unrealized_conversion_cast shape";
        castCleanupFailed = true;
        return;
      }

      Value input = cast.getOperand(0);
      Value output = cast.getResult(0);
      Type inTy = input.getType();
      Type outTy = output.getType();

      if (output.use_empty()) {
        castsToErase.push_back(cast);
        return;
      }

      if (inTy == outTy) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      // SCF/CFG type conversion can transiently materialize pointer->memref
      // bridge casts. At this stage, the producing value is already in the
      // lowered EmitC pointer form; keep it and drop the bridge cast.
      if (isEmitCPointerLikeType(inTy) && isa<BaseMemRefType>(outTy)) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      // SCF structural type conversion may leave a bridge from the converted
      // EmitC tile value back to the original pto.tile_buf type for PTO op
      // users. After PTO ops are lowered, the EmitC tile value is the value we
      // want to keep.
      if (isEmitCTileLikeType(inTy) && isa<pto::TileBufType>(outTy)) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      if (emitc::isSupportedEmitCType(inTy) && emitc::isSupportedEmitCType(outTy)) {
        OpBuilder builder(cast);
        auto c = builder.create<emitc::CastOp>(cast.getLoc(), outTy, input);
        output.replaceAllUsesWith(c.getResult());
        castsToErase.push_back(cast);
        return;
      }

      cast.emitError() << "cannot lower unrealized_conversion_cast(" << inTy
                       << " -> " << outTy << ") to emitc.cast";
      castCleanupFailed = true;
    });

    for (auto cast : castsToErase)
      cast.erase();

    if (castCleanupFailed)
      return signalPassFailure();

    // --- Step A2: Sink casts of emitc.variable "reads" to their use sites ---
    //
    // SCFToEmitC lowers scf.if/scf.for results via mutable `emitc.variable` and
    // `emitc.assign`. During type conversion, casts from the variable handle to
    // the converted type may be materialized right after the variable
    // declaration, effectively snapshotting the value *before* assignments. That
    // produces wrong C++ (use-before-init / stale reads).
    //
    // Fix by re-materializing the cast at each use site so it reads the variable
    // at the point of use.
    {
      SmallVector<emitc::CastOp> castOpsToSink;
      mop.walk([&](emitc::CastOp castOp) {
        if (castOp.getSource().getDefiningOp<emitc::VariableOp>())
          castOpsToSink.push_back(castOp);
      });

      for (emitc::CastOp castOp : castOpsToSink) {
        Value src = castOp.getSource();
        Type dstTy = castOp.getResult().getType();
        Value oldRes = castOp.getResult();

        // Replace each use with a freshly inserted cast right before the user.
        for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
          Operation *user = use.getOwner();
          OpBuilder b(user);
          b.setInsertionPoint(user);
          auto newCast = b.create<emitc::CastOp>(castOp.getLoc(), dstTy, src);
          use.set(newCast.getResult());
        }

        castOp.erase();
      }
    }

    // --- Step B: 修复 Loop 归纳变量 (IV) ---
    // 此时 emitc.for 的 operand 已经是 int32 了，我们检查 IV 是否匹配，不匹配则修正
    mop.walk([&](emitc::ForOp forOp) {
       Type boundTy = forOp.getLowerBound().getType(); 
       BlockArgument iv = forOp.getBody()->getArgument(0); 
       
       if (iv.getType() != boundTy) {
         iv.setType(boundTy); // 强制将 IV 类型 (index) 修改为与边界一致 (int32)
       }
    });
    
    // --- Step C: 消除冗余 Tile 变量 (Dead Code Elimination) [新增] ---
    // 逻辑：如果一个 emitc.variable 没有被读取（use_empty），
    // 那么它自己，以及给它赋值的 TASSIGN 都可以删除。
    // 注意：TASSIGN(v15, v9) 会把 v15 作为 Operand 0 使用，所以 v15 不是严格的 use_empty。
    // 我们需要检查：v15 是否除了 TASSIGN 之外没有其他 User。

    llvm::SmallVector<emitc::VariableOp> deadVars;
    mop.walk([&](emitc::VariableOp varOp) {
        // 检查该变量的所有 User
        bool isRead = false;
        for (Operation* user : varOp.getResult().getUsers()) {
            // 如果 User 是 TASSIGN 且变量是第0个参数(dst)，不算"读取"
            if (auto call = dyn_cast<emitc::CallOpaqueOp>(user)) {
                if (call.getCallee() == "TASSIGN" && call.getOperand(0) == varOp.getResult()) {
                    continue; // 这是一个赋值操作，不算有效使用
                }
            }
            // 如果还有其他用途（如 TLOAD, TMOV, TMATMUL），则该变量有用
            isRead = true;
            break;
        }

        if (!isRead) {
            deadVars.push_back(varOp);
        }
    });

    for (auto varOp : deadVars) {
        // 1. 先删除所有使用该变量的 TASSIGN
        llvm::SmallVector<Operation*> usersToErase;
        for (Operation* user : varOp.getResult().getUsers()) {
             // 我们上面已经确认过，剩下的 user 只能是 TASSIGN
             usersToErase.push_back(user);
        }
        for (auto u : usersToErase) u->erase();

        // 2. 删除变量定义本身
        varOp.erase();
    }

    llvm::SmallVector<emitc::ConstantOp> deadConsts;
    mop.walk([&](emitc::ConstantOp constOp) {
      if (constOp.getResult().use_empty())
        deadConsts.push_back(constOp);
    });
    for (auto constOp : deadConsts)
      constOp.erase();

    // =========================================================================
  }
  };
} // namespace

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass() {
  return std::make_unique<EmitPTOManualPass>();
}

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass(PTOArch arch) {
  return std::make_unique<EmitPTOManualPass>(arch);
}
