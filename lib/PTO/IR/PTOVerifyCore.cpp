// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Split from PTO.cpp to keep source file nbnc under the codecheck threshold.
// This file is included by lib/PTO/IR/PTO.cpp and intentionally has no local includes.

using namespace mlir;
using namespace mlir::pto;

void mlir::pto::PartitionViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", offsets = [";
  printer.printOperands(getOffsets());
  printer << "], sizes = [";
  printer.printOperands(getSizes());
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << getSource().getType();

  auto inferredResultType = inferPartitionViewResultTypeFromSizes(
      dyn_cast<mlir::pto::TensorViewType>(getSource().getType()), getSizes());
  if (succeeded(inferredResultType) && *inferredResultType == getResult().getType())
    return;

  printer << " -> " << getResult().getType();
}

static std::optional<int64_t> getConstantIntegerValueEx(
    Value v, bool includeIndexAndIntOpsInConstFold) {
  if (includeIndexAndIntOpsInConstFold) {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>())
      return c.value();
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>())
      return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static LogicalResult verifyNonNegativeIndexRowCol(
    Operation &op, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold) {
  if (!indexRow.getType().isIndex() || !indexCol.getType().isIndex())
    return op.emitOpError("expects indexRow and indexCol to be index type");
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  if (row && *row < 0)
    return op.emitOpError("expects indexRow to be non-negative");
  if (col && *col < 0)
    return op.emitOpError("expects indexCol to be non-negative");
  return success();
}

static LogicalResult verifyExtractStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + dstShape[0] > srcShape[0])
    return op.emitOpError("expects indexRow + dst.rows <= src.rows");
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + dstShape[1] > srcShape[1])
    return op.emitOpError("expects indexCol + dst.cols <= src.cols");
  return success();
}

static LogicalResult verifyInsertStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getValidShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + srcShape[0] > dstShape[0])
    return op.emitOpError("expects indexRow + src.rows <= dst.rows");
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + srcShape[1] > dstShape[1])
    return op.emitOpError("expects indexCol + src.cols <= dst.cols");
  return success();
}

static unsigned getElemByteSize(Type ty) {
  return getPTOStorageElemByteSize(ty);
}

static bool readBLayoutValue(Attribute attr, int32_t &out) {
  if (auto layout = dyn_cast_or_null<BLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layout.getValue());
    return true;
  }
  if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(value.getInt());
    return true;
  }
  return false;
}

static bool readSLayoutValue(Attribute attr, int32_t &out) {
  if (auto layout = dyn_cast_or_null<SLayoutAttr>(attr)) {
    out = static_cast<int32_t>(layout.getValue());
    return true;
  }
  if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(value.getInt());
    return true;
  }
  return false;
}

static LogicalResult verifyTileBufPositiveShape(Operation *op,
                                                ArrayRef<int64_t> shape,
                                                StringRef name) {
  if (shape.size() != 2)
    return op->emitOpError() << "expects " << name << " to be rank-2";
  if (shape[0] != ShapedType::kDynamic && shape[0] <= 0)
    return op->emitOpError() << "expects " << name << " rows to be positive";
  if (shape[1] != ShapedType::kDynamic && shape[1] <= 0)
    return op->emitOpError() << "expects " << name << " cols to be positive";
  return success();
}

static LogicalResult verifyNoneBoxTileBufLayout(Operation *op, StringRef name,
                                                int32_t blayout,
                                                int64_t rows, int64_t cols,
                                                unsigned elemBytes) {
  constexpr int64_t kAlignedBytes = 32;
  auto checkByteAlignment = [&](int64_t dim, StringRef layoutName,
                                StringRef byteExpr) -> LogicalResult {
    if (dim == ShapedType::kDynamic)
      return success();
    int64_t bytes = dim * static_cast<int64_t>(elemBytes);
    if (bytes % kAlignedBytes == 0)
      return success();
    return op->emitOpError()
           << "expects " << name << " " << layoutName
           << " none_box tile " << byteExpr
           << " to be 32-byte aligned, but got " << bytes << " bytes";
  };
  if (blayout == static_cast<int32_t>(BLayout::RowMajor))
    return checkByteAlignment(cols, "row-major",
                              "row byte size (cols * sizeof(dtype))");
  return checkByteAlignment(rows, "col-major",
                            "column byte size (rows * sizeof(dtype))");
}

static LogicalResult getBoxedTileInnerShape(Operation *op, StringRef name,
                                            int32_t slayout, int32_t fractal,
                                            unsigned elemBytes,
                                            int64_t &innerRows,
                                            int64_t &innerCols) {
  constexpr int64_t kAlignedBytes = 32;
  if (elemBytes == 0)
    return op->emitOpError() << "expects " << name
                             << " to have a non-zero element byte size";
  switch (fractal) {
  case 1024:
    innerRows = 16;
    innerCols = 16;
    return success();
  case 32:
    innerRows = 16;
    innerCols = 2;
    return success();
  case 512:
    if (kAlignedBytes % elemBytes != 0) {
      return op->emitOpError() << "expects " << name
                               << " element byte size to divide 32 for boxed "
                                  "fractal-512 tile layout";
    }
    if (slayout == static_cast<int32_t>(SLayout::RowMajor)) {
      innerRows = 16;
      innerCols = kAlignedBytes / static_cast<int64_t>(elemBytes);
      return success();
    }
    if (slayout == static_cast<int32_t>(SLayout::ColMajor)) {
      innerRows = kAlignedBytes / static_cast<int64_t>(elemBytes);
      innerCols = 16;
      return success();
    }
    break;
  default:
    break;
  }
  return op->emitOpError() << "expects " << name
                           << " to use a supported boxed tile layout";
}

static LogicalResult verifyBoxedTileBufLayout(Operation *op, pto::TileBufType tb,
                                              StringRef name, int64_t rows,
                                              int64_t cols, unsigned elemBytes,
                                              int32_t slayout,
                                              int32_t fractal) {
  int64_t innerRows = 0;
  int64_t innerCols = 0;
  if (failed(getBoxedTileInnerShape(op, name, slayout, fractal, elemBytes,
                                    innerRows, innerCols)))
    return failure();

  auto loc = getPTOMemorySpaceEnum(tb);
  bool allowUnalignedRows =
      (loc && *loc == pto::AddressSpace::VEC) || fractal == 32 || rows == 1;
  if (!allowUnalignedRows && rows != ShapedType::kDynamic &&
      rows % innerRows != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile rows to be a multiple of innerRows (" << innerRows
           << "), but got " << rows;
  }
  if (cols != ShapedType::kDynamic && cols % innerCols != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile cols to be a multiple of innerCols (" << innerCols
           << "), but got " << cols;
  }
  return success();
}

static LogicalResult verifyTileBufLayoutConstraints(Operation *op,
                                                    pto::TileBufType tb,
                                                    StringRef name) {
  auto shape = tb.getShape();
  if (failed(verifyTileBufPositiveShape(op, shape, name)))
    return failure();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  unsigned elemBytes = getElemByteSize(tb.getElementType());
  if (elemBytes == 0)
    return op->emitOpError() << "expects " << name
                             << " element type to have a byte size";

  auto cfg = tb.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(tb.getContext());
  int32_t blayout = 0;
  int32_t slayout = 0;
  if (!readBLayoutValue(cfg.getBLayout(), blayout) ||
      !readSLayoutValue(cfg.getSLayout(), slayout)) {
    return op->emitOpError() << "expects " << name
                             << " to have concrete tile layout attributes";
  }

  if (slayout == static_cast<int32_t>(SLayout::NoneBox))
    return verifyNoneBoxTileBufLayout(op, name, blayout, rows, cols, elemBytes);
  int32_t fractal = static_cast<int32_t>(cfg.getSFractalSize().getInt());
  return verifyBoxedTileBufLayout(op, tb, name, rows, cols, elemBytes, slayout,
                                  fractal);
}

[[maybe_unused]] static bool isSupportedLoadStoreElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isBF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32 || width == 64;
  }
  return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 16 || width == 32;
  }
  return false;
}

static bool isSupportedGatherElemTypeA5(Type ty) {
  if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16())
    return true;
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == 8;
  }
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  return false;
}

static std::optional<mlir::pto::Layout>
inferLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
            unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;

  // NZ / fractal: rank>=5, check middle dims (sh3/sh4/sh5 per spec)
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return mlir::pto::Layout::NZ;
  }

  // ND: row-major contiguous
  bool isRowMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    if (strides[i] != strides[i + 1] * shape[i + 1]) {
      isRowMajor = false;
      break;
    }
  }
  if (isRowMajor && strides.back() == 1)
    return mlir::pto::Layout::ND;

  // DN: col-major
  bool isColMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    if (strides[i + 1] != strides[i] * shape[i]) {
      isColMajor = false;
      break;
    }
  }
  if (isColMajor && strides.front() == 1)
    return mlir::pto::Layout::DN;

  return mlir::pto::Layout::ND; // fallback
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value) {
  if (!value)
    return std::nullopt;
  if (auto part = value.getDefiningOp<pto::PartitionViewOp>())
    return getLogicalViewLayout(part.getSource());
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!tvTy)
      return std::nullopt;
    SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
    SmallVector<int64_t> strides;
    strides.reserve(make.getStrides().size());
    for (Value stride : make.getStrides()) {
      auto cst = getConstIndexValue(stride);
      if (!cst)
        return std::nullopt;
      strides.push_back(*cst);
    }
    return inferLayout(shape, strides, getElemByteSize(tvTy.getElementType()));
  }
  return std::nullopt;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type) {
  if (!type)
    return std::nullopt;
  int32_t sl = type.getSLayoutValueI32();
  int32_t bl = type.getBLayoutValueI32();
  if (sl != static_cast<int32_t>(pto::SLayout::NoneBox))
    return pto::Layout::NZ;
  if (bl == static_cast<int32_t>(pto::BLayout::RowMajor))
    return pto::Layout::ND;
  if (bl == static_cast<int32_t>(pto::BLayout::ColMajor))
    return pto::Layout::DN;
  return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  if (auto mr = dyn_cast<MemRefType>(ty))
    (void)mr;
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout != pto::Layout::ND)
      return op->emitOpError() << "expects " << name
                               << " to use an ND-style tile layout";
  }
  return success();
}

static LogicalResult verifyRowReductionDstLayout(Operation *op, Type ty,
                                                 StringRef name) {
  auto verifyBaseLayout = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(op, ty, name)))
      return failure();
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC)
      return op->emitOpError()
             << "expects " << name << " to be in the vec address space";
    if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
      if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return op->emitOpError()
               << "expects " << name << " to use the none_box slayout";
      if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
        return op->emitOpError()
               << "expects " << name
               << " to use the row_major or col_major blayout";
      }
    }
    return success();
  };
  auto verifyTileLayout = [&](pto::TileBufType tb) -> LogicalResult {
    auto layout = getTileBufLogicalLayout(tb);
    if (!layout || *layout == pto::Layout::ND)
      return success();
    if (*layout != pto::Layout::DN) {
      return op->emitOpError()
             << "expects " << name
             << " to use a DN-style column vector tile or legacy ND-style tile";
    }
    auto shape = getShapeVec(ty);
    if (shape.size() == 2 && shape[1] != ShapedType::kDynamic && shape[1] != 1) {
      return op->emitOpError()
             << "expects DN-style " << name << " to have shape[1] == 1";
    }
    return success();
  };

  if (failed(verifyBaseLayout()))
    return failure();
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return verifyTileLayout(tb);
  return success();
}

static LogicalResult verifyPositiveStaticDims(Operation *op,
                                             ArrayRef<int64_t> dims,
                                             StringRef name,
                                             StringRef kind) {
  for (auto [idx, dim] : llvm::enumerate(dims)) {
    if (dim != ShapedType::kDynamic && dim <= 0) {
      return op->emitOpError()
             << "expects " << name << " " << kind << "[" << idx
             << "] to be positive";
    }
  }
  return success();
}

static LogicalResult verifyPositiveRankedMemrefShape(Operation *op, MemRefType mr,
                                                     StringRef name) {
  if (!mr.hasRank())
    return op->emitOpError() << "expects " << name << " memref to be ranked";
  for (int64_t dim : mr.getShape()) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " memref shape to be positive";
  }
  return success();
}

struct TLoadVerifyInfo {
  pto::PartitionTensorViewType srcPart;
  pto::TileBufType dstTile;
  Type srcElem;
  Type dstElem;
};

static FailureOr<TLoadVerifyInfo> verifyTLoadCommon(Operation *op, Value src,
                                                    Value dst,
                                                    bool allowLowPrecision) {
  auto srcPart = dyn_cast<pto::PartitionTensorViewType>(src.getType());
  auto dstTile = dyn_cast<pto::TileBufType>(dst.getType());
  if (!srcPart || !dstTile) {
    op->emitOpError(
        "expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision)) ||
      failed(verifyPositiveStaticDims(op, srcPart.getShape(), "src", "shape")) ||
      failed(
          verifyPositiveStaticDims(op, dstTile.getValidShape(), "dst", "valid_shape"))) {
    return failure();
  }
  return TLoadVerifyInfo{srcPart, dstTile, srcPart.getElementType(),
                         dstTile.getElementType()};
}

static bool isA2A3TLoadDstElemType(Type elem) {
  return elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32) ||
         elem.isInteger(64) || elem.isF16() || elem.isBF16() || elem.isF32();
}

static LogicalResult verifyTLoadA2A3(Operation *op, const TLoadVerifyInfo &info) {
  if (isPTOLowPrecisionType(info.srcElem) || isPTOLowPrecisionType(info.dstElem))
    return op->emitOpError(
        "expects A2/A3 tload low-precision element types to be unsupported");
  if (!isA2A3TLoadDstElemType(info.dstElem)) {
    return op->emitOpError(
        "expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
  }

  auto dstSpace = getPTOMemorySpaceEnum(info.dstTile);
  if (!dstSpace ||
      (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
    return op->emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
  }
  if (getElemByteSize(info.srcElem) != getElemByteSize(info.dstElem)) {
    return op->emitOpError(
        "expects src and dst element types to have the same bitwidth");
  }
  return success();
}

static LogicalResult verifyTLoadA5(Operation *op, const TLoadVerifyInfo &info) {
  unsigned srcBytes = getElemByteSize(info.srcElem);
  unsigned dstBytes = getElemByteSize(info.dstElem);
  if (srcBytes != dstBytes) {
    return op->emitOpError(
        "expects src and dst element types to have the same element size");
  }
  if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8)) {
    return op->emitOpError(
        "expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
  }
  if (!isA5TLoadStoreTransferElemType(info.srcElem)) {
    return op->emitOpError(
        "expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  if (!isA5TLoadStoreTransferElemType(info.dstElem)) {
    return op->emitOpError(
        "expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  if (info.dstElem.isInteger(64)) {
    auto pad = info.dstTile.getPadValueI32();
    if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
        pad != static_cast<int32_t>(pto::PadValue::Zero)) {
      return op->emitOpError(
          "expects A5 i64/u64 tload dst pad to be null or zero");
    }
  }
  return success();
}

static FailureOr<Type> verifyTPrefetchSrcElemType(Operation *op, Type srcTy) {
  if (auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy)) {
    if (failed(verifyPositiveStaticDims(op, srcPart.getShape(), "src", "shape")))
      return failure();
    return srcPart.getElementType();
  }
  if (auto srcMr = dyn_cast<MemRefType>(srcTy)) {
    if (failed(verifyPositiveRankedMemrefShape(op, srcMr, "src")))
      return failure();
    return srcMr.getElementType();
  }
  op->emitOpError("expects src to be !pto.partition_tensor_view or memref");
  return failure();
}

static FailureOr<Type> verifyTPrefetchDstElemType(Operation *op, Type dstTy,
                                                  bool allowLowPrecision) {
  if (auto dstTile = dyn_cast<pto::TileBufType>(dstTy)) {
    if (failed(verifyTileBufCommon(op, dstTile, "dst", allowLowPrecision)) ||
        failed(
            verifyPositiveStaticDims(op, dstTile.getValidShape(), "dst", "valid_shape"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace ||
        (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
      return op->emitOpError("expects dst to use loc=vec or loc=mat"), failure();
    }
    return dstTile.getElementType();
  }
  if (auto dstMr = dyn_cast<MemRefType>(dstTy)) {
    auto dstSpace = getPTOMemorySpaceEnum(dstMr);
    if (!dstSpace ||
        (*dstSpace != pto::AddressSpace::VEC && *dstSpace != pto::AddressSpace::MAT)) {
      return op->emitOpError("expects dst memref to use loc=vec or loc=mat"),
             failure();
    }
    if (failed(verifyPositiveRankedMemrefShape(op, dstMr, "dst")) ||
        failed(verifyTileBufCommon(op, dstMr, "dst", allowLowPrecision))) {
      return failure();
    }
    return dstMr.getElementType();
  }
  op->emitOpError("expects dst to be !pto.tile_buf or memref");
  return failure();
}

static LogicalResult verifyTPrefetchElemTypes(Operation *op, Type srcElem,
                                              Type dstElem,
                                              bool allowLowPrecision) {
  if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
    return op->emitOpError(
        "expects src and dst element types to have the same element size");
  }
  if (!allowLowPrecision &&
      (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))) {
    return op->emitOpError(
        "expects A2/A3 tprefetch low-precision element types to be unsupported");
  }
  if (allowLowPrecision &&
      (!isA5TLoadStoreTransferElemType(srcElem) ||
       !isA5TLoadStoreTransferElemType(dstElem))) {
    return op->emitOpError(
        "expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
  }
  return success();
}

static std::optional<int64_t> getStaticElemCount(ArrayRef<int64_t> shape) {
  int64_t total = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return std::nullopt;
    if (total > std::numeric_limits<int64_t>::max() / dim)
      return std::nullopt;
    total *= dim;
  }
  return total;
}

struct TStoreVerifyInfo {
  pto::TileBufType srcTile;
  pto::PartitionTensorViewType dstPart;
  Type srcElem;
  Type dstElem;
};

static FailureOr<TStoreVerifyInfo> verifyTStoreCommon(Operation *op, Value src,
                                                      Value dst,
                                                      bool allowLowPrecision) {
  auto srcTile = dyn_cast<pto::TileBufType>(src.getType());
  auto dstPart = dyn_cast<pto::PartitionTensorViewType>(dst.getType());
  if (!srcTile || !dstPart) {
    op->emitOpError(
        "expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
    return failure();
  }
  if (failed(verifyTileBufCommon(op, srcTile, "src", allowLowPrecision)) ||
      failed(verifyPositiveStaticDims(op, dstPart.getShape(), "dst", "shape")) ||
      failed(
          verifyPositiveStaticDims(op, srcTile.getValidShape(), "src", "valid_shape"))) {
    return failure();
  }

  auto dstElemCount = getStaticElemCount(dstPart.getShape());
  auto srcValidElemCount = getStaticElemCount(srcTile.getValidShape());
  if (dstElemCount && srcValidElemCount && *dstElemCount != *srcValidElemCount) {
    op->emitOpError() << "expects dst static element count (" << *dstElemCount
                      << ") to match src valid_shape static element count ("
                      << *srcValidElemCount << ")";
    return failure();
  }
  return TStoreVerifyInfo{srcTile, dstPart, srcTile.getElementType(),
                          dstPart.getElementType()};
}

static bool isLoadStoreElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isI8Like(Type ty) { return ty.isInteger(8); }

static LogicalResult verifyTStoreModeSource(Operation *op,
                                            pto::AddressSpace srcSpace,
                                            bool hasPreQuant,
                                            pto::ReluPreMode reluMode) {
  if (hasPreQuant && srcSpace != pto::AddressSpace::ACC)
    return op->emitOpError("expects preQuantScalar form to use loc=acc src");
  if (reluMode != pto::ReluPreMode::NoRelu && srcSpace != pto::AddressSpace::ACC) {
    return op->emitOpError("expects reluPreMode form to use loc=acc src");
  }
  return success();
}

static LogicalResult verifyTStoreA2A3AccDstType(Operation *op, Type srcElem,
                                                Type dstElem, bool hasPreQuant) {
  if (hasPreQuant) {
    if (srcElem.isInteger(32)) {
      if (!(isI8Like(dstElem) || dstElem.isF16())) {
        return op->emitOpError(
            "expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
      }
    } else if (srcElem.isF32() && !isI8Like(dstElem)) {
      return op->emitOpError(
          "expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
    }
    return success();
  }
  if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16())) {
    return op->emitOpError(
        "expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
  }
  return success();
}

static LogicalResult verifyTStoreA2A3(TStoreOp op, const TStoreVerifyInfo &info,
                                      bool hasPreQuant,
                                      pto::ReluPreMode reluMode) {
  auto srcSpace = getPTOMemorySpaceEnum(info.srcTile);
  if (!srcSpace ||
      (*srcSpace != pto::AddressSpace::VEC && *srcSpace != pto::AddressSpace::MAT &&
       *srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError(
        "expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
  }
  if (failed(verifyTStoreModeSource(op.getOperation(), *srcSpace, hasPreQuant,
                                    reluMode))) {
    return failure();
  }

  if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
    if (isPTOLowPrecisionType(info.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
    }
    if (!isLoadStoreElemType(info.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
    }
    if (getElemByteSize(info.srcElem) != getElemByteSize(info.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
    }
    return success();
  }

  if (!(info.srcElem.isInteger(32) || info.srcElem.isF32()))
    return op.emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
  if (failed(verifyTStoreA2A3AccDstType(op.getOperation(), info.srcElem,
                                        info.dstElem, hasPreQuant))) {
    return failure();
  }

  auto srcShape = info.srcTile.getShape();
  if (srcShape[1] != ShapedType::kDynamic &&
      (srcShape[1] < 1 || srcShape[1] > 4095)) {
    return op.emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
  }
  auto srcValid = info.srcTile.getValidShape();
  if (srcValid[1] != ShapedType::kDynamic &&
      (srcValid[1] < 1 || srcValid[1] > 4095)) {
    return op.emitOpError(
        "expects A2/A3 acc tstore src valid_shape[1] to be in [1, 4095]");
  }
  return success();
}

static LogicalResult verifyTStoreA5AccDstType(Operation *op, Type srcElem,
                                              Type dstElem, bool hasPreQuant) {
  if (hasPreQuant) {
    if (!isA5AccStorePreQuantDstType(srcElem, dstElem)) {
      return op->emitOpError(
          "expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
    }
    return success();
  }
  if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
        dstElem.isBF16())) {
    return op->emitOpError(
        "expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
  }
  return success();
}

static LogicalResult verifyTStoreA5(TStoreOp op, const TStoreVerifyInfo &info,
                                    bool hasPreQuant,
                                    pto::ReluPreMode reluMode) {
  auto srcSpace = getPTOMemorySpaceEnum(info.srcTile);
  if (!srcSpace ||
      (*srcSpace != pto::AddressSpace::VEC && *srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
  }
  if (failed(verifyTStoreModeSource(op.getOperation(), *srcSpace, hasPreQuant,
                                    reluMode))) {
    return failure();
  }

  if (*srcSpace == pto::AddressSpace::VEC) {
    if (!isA5TLoadStoreTransferElemType(info.srcElem)) {
      return op.emitOpError(
          "expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    if (getElemByteSize(info.srcElem) != getElemByteSize(info.dstElem)) {
      return op.emitOpError(
          "expects A5 vec tstore src and dst element types to have the same bitwidth");
    }
    return success();
  }

  if (!(info.srcElem.isInteger(32) || info.srcElem.isF32()))
    return op.emitOpError("expects A5 acc tstore src element type to be i32 or f32");
  return verifyTStoreA5AccDstType(op.getOperation(), info.srcElem, info.dstElem,
                                  hasPreQuant);
}

static LogicalResult verifyRowReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
    return op->emitOpError("expects src valid_shape[0] to be non-zero");
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
    return op->emitOpError("expects src valid_shape[1] to be non-zero");
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0])
    return op->emitOpError("expects src and dst to have the same valid_shape[0]");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1)
    return op->emitOpError("expects dst valid_shape[1] to be 1");
  return success();
}

static bool isSupportedRowReductionElemType(Type elem) {
  return elem.isInteger(16) || elem.isInteger(32) || elem.isF16() ||
         elem.isF32();
}

static LogicalResult verifyTRowReductionNoTmpCommon(Operation *op, Type srcTy,
                                                    Type dstTy,
                                                    StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

static LogicalResult verifyTRowReductionWithTmpCommon(Operation *op, Type srcTy,
                                                      Type tmpTy, Type dstTy,
                                                      StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy)))
    return failure();
  if (!isSupportedRowReductionElemType(getElemTy(srcTy)))
    return op->emitOpError(elemTypeError);
  return success();
}

static LogicalResult verifyTRowArgReductionCommon(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
    return failure();
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem))
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32)
    return op->emitOpError("expects dst element type to be i32 or ui32");
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
    return failure();
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
  }
  return success();
}

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name)))
    return failure();
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy)
    return emitOpError("result must be pto.tensor_view<...>");

  auto pty = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!pty)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  if (pty.getElementType() != tvTy.getElementType())
    return emitOpError() << "ptr element type must match tensor_view element type, but got ptr="
                         << pty.getElementType() << " view=" << tvTy.getElementType();

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank)
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();

  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });

  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

static LogicalResult verifyPartitionViewDimension(
    PartitionViewOp op, int64_t dimIdx, ArrayRef<int64_t> srcShape,
    ArrayRef<int64_t> resShape, bool sameRank) {
  auto offVal = getConstIndexValue(op.getOffsets()[dimIdx]);
  auto sizeVal = getConstIndexValue(op.getSizes()[dimIdx]);
  if (offVal && *offVal < 0)
    return op.emitOpError() << "offset at dim " << dimIdx
                            << " must be non-negative, got " << *offVal;
  if (sizeVal && *sizeVal <= 0)
    return op.emitOpError() << "size at dim " << dimIdx
                            << " must be positive, got " << *sizeVal;
  if (sameRank && sizeVal) {
    int64_t resDim = resShape[dimIdx];
    if (resDim != ShapedType::kDynamic && *sizeVal != resDim) {
      return op.emitOpError() << "size/result mismatch at dim " << dimIdx
                              << ": size operand=" << *sizeVal
                              << " result type dim=" << resDim;
    }
  }
  int64_t srcDim = srcShape[dimIdx];
  if (srcDim == ShapedType::kDynamic)
    return success();
  if (sizeVal && *sizeVal > srcDim) {
    return op.emitOpError() << "size at dim " << dimIdx << " (" << *sizeVal
                            << ") exceeds static source dim (" << srcDim
                            << ")";
  }
  if (offVal && sizeVal && (*offVal + *sizeVal > srcDim)) {
    return op.emitOpError() << "offset+size at dim " << dimIdx << " ("
                            << (*offVal + *sizeVal)
                            << ") exceeds static source dim (" << srcDim
                            << ")";
  }
  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  auto resTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!srcTy || !resTy)
    return emitOpError("expects tensor_view source and partition_tensor_view result");
  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError() << "element type mismatch between source and result: src="
                         << srcTy.getElementType() << " result="
                         << resTy.getElementType();
  }

  int64_t srcRank = srcTy.getRank();
  if ((int64_t)getOffsets().size() != srcRank)
    return emitOpError() << "offset count (" << getOffsets().size()
                         << ") must match source rank (" << srcRank << ")";
  if ((int64_t)getSizes().size() != srcRank)
    return emitOpError() << "size count (" << getSizes().size()
                         << ") must match source rank (" << srcRank << ")";

  ArrayRef<int64_t> srcShape = srcTy.getShape();
  ArrayRef<int64_t> resShape = resTy.getShape();
  bool sameRank = resTy.getRank() == srcRank;
  for (int64_t i = 0; i < srcRank; ++i) {
    if (failed(
            verifyPartitionViewDimension(*this, i, srcShape, resShape, sameRank)))
      return failure();
  }
  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy)
    return emitOpError("result must be !pto.ptr<...>");

  if (ptrTy != resTy)
    return emitOpError("result type must match ptr operand type");

  return success();
}

static LogicalResult verifyPtrLikeForAddressCast(Operation *op, Type type,
                                                 StringRef name) {
  if (isa<mlir::pto::PtrType>(type))
    return success();

  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy)
    return op->emitOpError()
           << "expects " << name << " to be !pto.ptr<...> or a GM memref";

  if (memTy.getRank() != 1)
    return op->emitOpError()
           << "expects lowered memref " << name << " to be rank-1";

  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError()
           << "expects lowered memref " << name << " to use GM address space";

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64())
    return true;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32 || intTy.getWidth() == 64;
  if (mlir::pto::isPTOFloat8Type(type))
    return true;
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type))
    return true;
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != 64)
    return emitOpError("result must be i64");

  return verifyPtrLikeForAddressCast(getOperation(), getPtr().getType(),
                                     "ptr operand");
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != 64)
    return emitOpError("address operand must be i64");

  if (failed(verifyPtrLikeForAddressCast(getOperation(), getResult().getType(),
                                         "result")))
    return failure();

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem))
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getResult().getType() != arrayTy.getElementType())
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getValue().getType() != arrayTy.getElementType())
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  return success();
}
AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType)
    return {};
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!scopeAttr)
    return {};
  return scopeAttr;
}

bool mlir::pto::isScalarPtrOrMemRef(Type type) {
  if (auto pty = dyn_cast<mlir::pto::PtrType>(type))
    return true;
  if (auto memTy = dyn_cast<MemRefType>(type))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return false;
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName));
}

static constexpr StringLiteral kEffectivePTOEntryAttrName =
    "pto.internal.entry";

static SmallVector<func::FuncOp> getPTOFunctionDefinitions(ModuleOp module) {
  SmallVector<func::FuncOp> defs;
  if (!module)
    return defs;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!func.isDeclaration())
      defs.push_back(func);
  }
  return defs;
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  if (auto attr = func->getAttrOfType<BoolAttr>(kEffectivePTOEntryAttrName))
    return attr.getValue();
  if (hasExplicitPTOEntryAttr(func))
    return true;

  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  SmallVector<func::FuncOp> defs = getPTOFunctionDefinitions(module);
  return defs.size() == 1 && defs.front() == func;
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return success();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func))
      continue;
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func))
      continue;
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return;

  SmallVector<func::FuncOp> defs = getPTOFunctionDefinitions(module);
  for (auto func : module.getOps<func::FuncOp>())
    func->removeAttr(kEffectivePTOEntryAttrName);

  if (defs.empty())
    return;
  if (defs.size() == 1) {
    defs.front()->setAttr(kEffectivePTOEntryAttrName,
                          BoolAttr::get(module.getContext(), true));
    return;
  }

  for (auto func : defs) {
    func->setAttr(kEffectivePTOEntryAttrName,
                  BoolAttr::get(module.getContext(),
                                hasExplicitPTOEntryAttr(func)));
  }
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//  - If operands are memref/tensor: verify strictly.
//  - Otherwise (tile_view/tile etc): accept (so old IR can still parse).
//===----------------------------------------------------------------------===//

[[maybe_unused]] static LogicalResult verifyMemrefToTensorLoad(Operation *op, Value src, Value res) {
  auto mr = dyn_cast<MemRefType>(src.getType());
  auto rt = dyn_cast<RankedTensorType>(res.getType());
  if (!mr)
    return success(); // non-memref case: don't block old IR
  if (!rt)
    return op->emitOpError("when src is memref, result must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  if (mr.hasStaticShape()) {
    if (!rt.hasStaticShape())
      return op->emitOpError("memref has static shape but result tensor is not static");
    if (mr.getShape() != rt.getShape())
      return op->emitOpError() << "shape mismatch: memref=" << mr << " tensor=" << rt;
  } else {
    // For dynamic memref dims: if tensor dim is static, allow it; if it's dynamic too, also fine.
    // We only reject when a memref static dim conflicts with tensor static dim.
    for (int64_t i = 0; i < mr.getRank(); ++i) {
      int64_t md = mr.getDimSize(i);
      int64_t td = rt.getDimSize(i);
      if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
        return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyMemrefTensorStore(Operation *op, Value dst, Value src) {
  auto mr = dyn_cast<MemRefType>(dst.getType());
  if (!mr)
    return success(); // non-memref case: old tile IR allowed
  auto rt = dyn_cast<RankedTensorType>(src.getType());
  if (!rt)
    return op->emitOpError("when dst is memref, src must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  for (int64_t i = 0; i < mr.getRank(); ++i) {
    int64_t md = mr.getDimSize(i);
    int64_t td = rt.getDimSize(i);
    if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
      return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
  }
  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result")))
    return failure();

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR)
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));

  if (hasVC != needVC)
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));

  return success();
}

LogicalResult MaterializeTileOp::verify() {
  auto sourceTy = cast<MemRefType>(getSource().getType());
  auto resultTy = cast<TileBufType>(getResult().getType());

  if (sourceTy.getRank() != 2)
    return emitOpError("source memref must be rank-2 to materialize a tile handle");
  if (resultTy.getRank() != 2)
    return emitOpError("result tile_buf must be rank-2");
  if (failed(verifyTileBufLayoutConstraints(*this, resultTy, "result")))
    return failure();

  auto viewSemantics = (*this)->getAttrOfType<StringAttr>("pto.view_semantics");
  bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
  if (!isSubview && sourceTy.getShape() != resultTy.getShape())
    return emitOpError() << "source/result shape mismatch: source="
                         << sourceTy << " result=" << resultTy;

  if (sourceTy.getElementType() != resultTy.getElementType())
    return emitOpError() << "source/result element type mismatch: source="
                         << sourceTy.getElementType()
                         << " result=" << resultTy.getElementType();

  if (sourceTy.getMemorySpace() != resultTy.getMemorySpace())
    return emitOpError() << "source/result memory space mismatch";

  if (getConfig() != resultTy.getConfigAttr())
    return emitOpError("config attribute must match the result tile_buf config");

  auto shape = resultTy.getShape();
  auto validShape = resultTy.getValidShape();
  if (validShape.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic &&
        validShape[i] != ShapedType::kDynamic && validShape[i] > shape[i]) {
      return emitOpError() << "valid_shape[" << i << "] must be <= shape["
                           << i << "]";
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }
  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common =
        verifyTLoadCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    return verifyTLoadA2A3(*this, *common);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common =
        verifyTLoadCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    return verifyTLoadA5(*this, *common);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TPrefetchOp::verify() {
  auto verifyByArch = [&](bool allowLowPrecision) -> LogicalResult {
    auto srcElem = verifyTPrefetchSrcElemType(*this, getSrc().getType());
    if (failed(srcElem))
      return failure();
    auto dstElem =
        verifyTPrefetchDstElemType(*this, getDst().getType(), allowLowPrecision);
    if (failed(dstElem))
      return failure();
    return verifyTPrefetchElemTypes(*this, *srcElem, *dstElem,
                                    allowLowPrecision);
  };
  return dispatchVerifierByArch(
      getOperation(),
      [&]() { return verifyByArch(/*allowLowPrecision=*/false); },
      [&]() { return verifyByArch(/*allowLowPrecision=*/true); });
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  Type workspaceTy = getWorkspace().getType();
  Type elemTy = nullptr;
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    elemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError("expects workspace memref to be in GM address space");
    elemTy = memTy.getElementType();
  } else {
    return emitOpError("expects workspace to be !pto.ptr<i8> or GM memref<i8>");
  }
  if (!isByteIntegerType(elemTy))
    return emitOpError("expects workspace element type to be an 8-bit integer");
  return success();
}

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src")))
    return failure();
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto mr = llvm::dyn_cast<mlir::MemRefType>(getFfts().getType());
  if (!mr)
    return emitOpError("expects a memref operand");

  if (!mr.getElementType().isInteger(64) && !mr.getElementType().isInteger(8))
    return emitOpError("expects element type i64 (or i8)");

  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2)
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE3:
      return success();
    default:
      return emitOpError()
             << "A5 sync.set expects pipe to be one of <PIPE_FIX>, <PIPE_MTE3>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

static ParseResult parseSyncAllOptionalOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes) {
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) ||
        parser.parseColonTypeList(operandTypes) || parser.parseRParen())
      return failure();
    if (operands.size() != operandTypes.size()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
    }
  }
  return success();
}

static ParseResult parseSyncAllModeAndCoreType(OpAsmParser &parser,
                                               OperationState &result,
                                               SyncAllModeAttr &mode,
                                               SyncCoreTypeAttr &coreType) {
  Attribute modeAttr;
  Attribute coreTypeAttr;
  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr))
    return failure();
  mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode)
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";
  coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType)
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";
  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);
  return parser.parseOptionalAttrDict(result.attributes);
}

static void addSyncAllSegmentSizes(OpAsmParser &parser, OperationState &result,
                                   int32_t gm, int32_t ub, int32_t l1,
                                   int32_t used) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {gm, ub, l1, used}));
}

static ParseResult resolveSyncAllSoftOperands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    ArrayRef<Type> operandTypes, int32_t gm, int32_t ub, int32_t l1) {
  int32_t required = gm + ub + l1;
  bool hasUsedCores = operands.size() == static_cast<size_t>(required + 1);
  if (operands.size() != static_cast<size_t>(required) && !hasUsedCores)
    return failure();
  for (int32_t i = 0; i < required; ++i) {
    if (parser.resolveOperand(operands[i], operandTypes[i], result.operands))
      return failure();
  }
  if (hasUsedCores &&
      parser.resolveOperand(operands[required], operandTypes[required],
                            result.operands))
    return failure();
  addSyncAllSegmentSizes(parser, result, gm, ub, l1, hasUsedCores ? 1 : 0);
  return success();
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> operandTypes;
  SyncAllModeAttr mode;
  SyncCoreTypeAttr coreType;
  if (failed(parseSyncAllOptionalOperands(parser, operands, operandTypes)) ||
      failed(parseSyncAllModeAndCoreType(parser, result, mode, coreType)))
    return failure();

  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty())
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    addSyncAllSegmentSizes(parser, result, 0, 0, 0, 0);
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  switch (coreType.getValue()) {
  case pto::SyncCoreType::AIVOnly:
    if (operands.size() != 2 && operands.size() != 3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIV-only syncall to have gm_workspace, "
                "ub_workspace, and optional used_cores";
    return resolveSyncAllSoftOperands(parser, result, operands, operandTypes, 1,
                                      1, 0);
  case pto::SyncCoreType::AICOnly:
    if (operands.size() != 2 && operands.size() != 3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIC-only syncall to have gm_workspace, "
                "l1_workspace, and optional used_cores";
    return resolveSyncAllSoftOperands(parser, result, operands, operandTypes, 1,
                                      0, 1);
  case pto::SyncCoreType::Mix:
    if (operands.size() != 3 && operands.size() != 4)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft mixed syncall to have gm_workspace, "
                "ub_workspace, l1_workspace, and optional used_cores";
    return resolveSyncAllSoftOperands(parser, result, operands, operandTypes, 1,
                                      1, 1);
  }

  llvm_unreachable("unhandled SyncCoreType");
}

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVector<Value, 4> operands;
  if (getGmWorkspace())
    operands.push_back(getGmWorkspace());
  if (getUbWorkspace())
    operands.push_back(getUbWorkspace());
  if (getL1Workspace())
    operands.push_back(getL1Workspace());
  if (getUsedCores())
    operands.push_back(getUsedCores());

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p,
                          [&](Value operand) { p.printType(operand.getType()); });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}

LogicalResult mlir::pto::SyncWaitOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.wait expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TStoreOp::verify() {
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  auto reluMode = getReluPreMode();

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common =
        verifyTStoreCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    return verifyTStoreA2A3(*this, *common, hasPreQuant, reluMode);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common =
        verifyTStoreCommon(*this, getSrc(), getDst(), /*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    return verifyTStoreA5(*this, *common, hasPreQuant, reluMode);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
