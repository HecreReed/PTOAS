// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOAttrs.cpp ------------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Parser/Parser.h"          // parseAttribute
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;

TileBufConfigAttr TileBufConfigAttr::getDefault(MLIRContext *ctx) {
  Builder b(ctx);
  BLayoutAttr bl = BLayoutAttr::get(ctx, BLayout::RowMajor);
  SLayoutAttr sl = SLayoutAttr::get(ctx, SLayout::NoneBox);
  PadValueAttr pv = PadValueAttr::get(ctx, PadValue::Null);
  CompactModeAttr compact = CompactModeAttr::get(ctx, CompactMode::Null);
  IntegerAttr sz = b.getI32IntegerAttr(512);
  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

bool TileBufConfigAttr::isDefault() const {
  auto d = getDefault(getContext());
  return getBLayout() == d.getBLayout() &&
         getSLayout() == d.getSLayout() &&
         getSFractalSize() == d.getSFractalSize() &&
         getPad() == d.getPad() &&
         getCompactMode() == d.getCompactMode();
}

static int32_t getLayoutInt(Attribute a, int32_t def) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a))
    return static_cast<int32_t>(bl.getValue());
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a))
    return static_cast<int32_t>(sl.getValue());
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a))
    return static_cast<int32_t>(pv.getValue());
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a))
    return static_cast<int32_t>(cm.getValue());
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a))
    return static_cast<int32_t>(ia.getInt());
  return def;
}

static LogicalResult verifyLayoutLikeAttr(
    function_ref<InFlightDiagnostic()> emitError, Attribute attr, StringRef name,
    function_ref<bool(Attribute)> predicate) {
  if (attr && predicate(attr))
    return success();
  return emitError() << name << " must be the expected enum attr or i32 integer attr",
         failure();
}

static LogicalResult verifyValueRange(function_ref<InFlightDiagnostic()> emitError,
                                      int32_t value, int32_t min, int32_t max,
                                      StringRef name) {
  if (value >= min && value <= max)
    return success();
  return emitError() << "unsupported " << name << " value: " << value, failure();
}

LogicalResult TileBufConfigAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, Attribute bLayout,
    Attribute sLayout, IntegerAttr sFractalSize, Attribute pad,
    Attribute compactMode) {
  auto isBLayoutLike = [](Attribute attr) {
    return mlir::isa<BLayoutAttr, IntegerAttr>(attr);
  };
  auto isSLayoutLike = [](Attribute attr) {
    return mlir::isa<SLayoutAttr, IntegerAttr>(attr);
  };
  auto isPadLike = [](Attribute attr) {
    return mlir::isa<PadValueAttr, IntegerAttr>(attr);
  };
  auto isCompactLike = [](Attribute attr) {
    return mlir::isa<CompactModeAttr, IntegerAttr>(attr);
  };

  if (failed(verifyLayoutLikeAttr(emitError, bLayout, "blayout", isBLayoutLike)) ||
      failed(verifyLayoutLikeAttr(emitError, sLayout, "slayout", isSLayoutLike)) ||
      failed(verifyLayoutLikeAttr(emitError, pad, "pad", isPadLike)) ||
      failed(verifyLayoutLikeAttr(emitError, compactMode, "compact_mode",
                                  isCompactLike))) {
    return failure();
  }

  if (!sFractalSize || !sFractalSize.getType().isInteger(32))
    return emitError() << "s_fractal_size must be i32", failure();

  int32_t s = static_cast<int32_t>(sFractalSize.getInt());
  if (s != 32 && s != 16 && s != 512 && s != 1024)
    return emitError() << "unsupported s_fractal_size: " << s, failure();

  if (failed(verifyValueRange(emitError, getLayoutInt(bLayout, -1), 0, 1,
                              "blayout")) ||
      failed(verifyValueRange(emitError, getLayoutInt(sLayout, -1), 0, 2,
                              "slayout")) ||
      failed(verifyValueRange(emitError, getLayoutInt(pad, -1), 0, 3,
                              "pad")) ||
      failed(verifyValueRange(emitError, getLayoutInt(compactMode, -1), 0, 2,
                              "compact_mode"))) {
    return failure();
  }
  return success();
}

// Helper: parse Attribute and convert to BLayoutAttr/SLayoutAttr/PadValueAttr
static BLayoutAttr toBLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto bl = mlir::dyn_cast<BLayoutAttr>(a))
    return bl;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a))
    return BLayoutAttr::get(ctx, static_cast<BLayout>(ia.getInt()));
  return {};
}
static SLayoutAttr toSLayoutAttr(MLIRContext *ctx, Attribute a) {
  if (auto sl = mlir::dyn_cast<SLayoutAttr>(a))
    return sl;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a))
    return SLayoutAttr::get(ctx, static_cast<SLayout>(ia.getInt()));
  return {};
}
static PadValueAttr toPadValueAttr(MLIRContext *ctx, Attribute a) {
  if (auto pv = mlir::dyn_cast<PadValueAttr>(a))
    return pv;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a))
    return PadValueAttr::get(ctx, static_cast<PadValue>(ia.getInt()));
  return {};
}
static CompactModeAttr toCompactModeAttr(MLIRContext *ctx, Attribute a) {
  if (auto cm = mlir::dyn_cast<CompactModeAttr>(a)) return cm;
  if (auto ia = mlir::dyn_cast<IntegerAttr>(a))
    return CompactModeAttr::get(ctx, static_cast<CompactMode>(ia.getInt()));
  return {};
}

static ParseResult parseTileBufConfigField(AsmParser &p, MLIRContext *ctx,
                                           StringRef key, BLayoutAttr &bl,
                                           SLayoutAttr &sl, IntegerAttr &sz,
                                           PadValueAttr &pv,
                                           CompactModeAttr &compact) {
  if (key == "blayout") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    bl = toBLayoutAttr(ctx, a);
    return bl ? success() : failure();
  }
  if (key == "slayout") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    sl = toSLayoutAttr(ctx, a);
    return sl ? success() : failure();
  }
  if (key == "s_fractal_size") {
    int32_t v;
    if (p.parseInteger(v))
      return failure();
    sz = IntegerAttr::get(IntegerType::get(ctx, 32), v);
    return success();
  }
  if (key == "pad") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    pv = toPadValueAttr(ctx, a);
    return pv ? success() : failure();
  }
  if (key == "compact") {
    Attribute a;
    if (p.parseAttribute(a))
      return failure();
    compact = toCompactModeAttr(ctx, a);
    return compact ? success() : failure();
  }
  p.emitError(p.getCurrentLocation(), "unknown key in tile_buf_config: ") << key;
  return failure();
}

Attribute TileBufConfigAttr::parse(AsmParser &p, Type) {
  MLIRContext *ctx = p.getContext();
  auto def = TileBufConfigAttr::getDefault(ctx);
  BLayoutAttr bl = def.getBLayout();
  SLayoutAttr sl = def.getSLayout();
  IntegerAttr sz = def.getSFractalSize();
  PadValueAttr pv = def.getPad();
  CompactModeAttr compact = def.getCompactMode();

  if (p.parseLess()) return {};

  if (succeeded(p.parseOptionalGreater()))
    return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);

  bool done = false;
  while (!done) {
    StringRef key;
    if (p.parseKeyword(&key) || p.parseEqual())
      return {};
    if (failed(parseTileBufConfigField(p, ctx, key, bl, sl, sz, pv, compact)))
      return {};
    if (succeeded(p.parseOptionalGreater())) {
      done = true;
      continue;
    }
    if (p.parseComma())
      return {};
  }

  return TileBufConfigAttr::get(ctx, bl, sl, sz, pv, compact);
}

void TileBufConfigAttr::print(AsmPrinter &p) const {
  p << "<";
  p << "blayout=" << getBLayout();
  p << ", slayout=" << getSLayout();
  p << ", s_fractal_size=" << (int32_t)getSFractalSize().getInt();
  p << ", pad=" << getPad();
  p << ", compact=" << getCompactMode();
  p << ">";
}
