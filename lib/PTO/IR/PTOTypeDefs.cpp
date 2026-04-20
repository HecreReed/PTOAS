// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTypeDefs.cpp --------------------------------------------*- C++ -*-===//
#include "PTO/IR/PTO.h"
#include "mlir/IR/DialectImplementation.h"
#include <limits>
#include <mutex>
#include <unordered_map>

using namespace mlir;
using namespace mlir::pto;

namespace {
std::mutex parserTargetArchMutex;
std::unordered_map<const MLIRContext *, PTOParserTargetArch>
    parserTargetArchByContext;
}

void mlir::pto::setPTOParserTargetArch(MLIRContext *context,
                                       PTOParserTargetArch arch) {
  if (!context)
    return;

  std::lock_guard<std::mutex> lock(parserTargetArchMutex);
  if (arch == PTOParserTargetArch::Unspecified) {
    parserTargetArchByContext.erase(context);
    return;
  }
  parserTargetArchByContext[context] = arch;
}

PTOParserTargetArch mlir::pto::getPTOParserTargetArch(MLIRContext *context) {
  if (!context)
    return PTOParserTargetArch::Unspecified;

  std::lock_guard<std::mutex> lock(parserTargetArchMutex);
  auto it = parserTargetArchByContext.find(context);
  if (it == parserTargetArchByContext.end())
    return PTOParserTargetArch::Unspecified;
  return it->second;
}

mlir::pto::ScopedPTOParserTargetArch::ScopedPTOParserTargetArch(
    MLIRContext *context, PTOParserTargetArch arch)
    : context(context), previousArch(getPTOParserTargetArch(context)) {
  setPTOParserTargetArch(context, arch);
}

mlir::pto::ScopedPTOParserTargetArch::~ScopedPTOParserTargetArch() {
  setPTOParserTargetArch(context, previousArch);
}

static SmallVector<int64_t, 4> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  SmallVector<int64_t, 4> canonical;
  canonical.reserve(validShape.size());
  for (int64_t dim : validShape)
    canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  return canonical;
}

static LogicalResult parseTileBufKeyEq(AsmParser &parser,
                                       StringRef expectedKey) {
  if (failed(parser.parseKeyword(expectedKey)))
    return failure();
  return parser.parseEqual();
}

static LogicalResult parseTileBufComma(AsmParser &parser) {
  return parser.parseComma();
}

static LogicalResult parseTileBufKeywordField(AsmParser &parser, StringRef key,
                                              std::string &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();
  if (failed(parser.parseKeywordOrString(&value)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufTypeField(AsmParser &parser, StringRef key,
                                           Type &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();
  if (failed(parser.parseType(value)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufIntegerField(AsmParser &parser, StringRef key,
                                              int64_t &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();
  if (failed(parser.parseInteger(value)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufValidDim(AsmParser &parser, StringRef key,
                                          int64_t &value) {
  if (failed(parseTileBufKeyEq(parser, key)))
    return failure();

  if (succeeded(parser.parseOptionalQuestion())) {
    value = -1;
    return success();
  }

  if (failed(parser.parseInteger(value)))
    return failure();
  if (value < -1) {
    parser.emitError(parser.getCurrentLocation(),
                     key + " must be '?', -1, or a non-negative integer");
    return failure();
  }
  return success();
}

static LogicalResult parseTileBufValidShapeFields(AsmParser &parser,
                                                  int64_t &vrow,
                                                  int64_t &vcol) {
  if (failed(parseTileBufValidDim(parser, "v_row", vrow)))
    return failure();
  if (failed(parseTileBufComma(parser)))
    return failure();
  if (failed(parseTileBufValidDim(parser, "v_col", vcol)))
    return failure();
  return parseTileBufComma(parser);
}

static LogicalResult parseTileBufPadField(AsmParser &parser, uint32_t &padInt) {
  int64_t parsedPad = 0;
  if (failed(parseTileBufKeyEq(parser, "pad")))
    return failure();
  if (failed(parser.parseInteger(parsedPad)))
    return failure();
  if (parsedPad < 0 || parsedPad > std::numeric_limits<uint32_t>::max()) {
    parser.emitError(parser.getCurrentLocation(),
                     "pad must be a non-negative 32-bit integer");
    return failure();
  }
  padInt = static_cast<uint32_t>(parsedPad);
  return success();
}

static std::optional<AddressSpace> resolveTileBufMemorySpace(StringRef locStr) {
  return ::llvm::StringSwitch<::std::optional<AddressSpace>>(locStr)
      .Case("mat", AddressSpace::MAT)
      .Case("left", AddressSpace::LEFT)
      .Case("right", AddressSpace::RIGHT)
      .Case("acc", AddressSpace::ACC)
      .Case("vec", AddressSpace::VEC)
      .Case("bias", AddressSpace::BIAS)
      .Case("scaling", AddressSpace::SCALING)
      .Default(::std::nullopt);
}

static BLayout resolveTileBufBLayout(MLIRContext *context,
                                     AddressSpace memorySpace,
                                     BLayout parsedLayout) {
  if (memorySpace != AddressSpace::LEFT)
    return parsedLayout;

  switch (getPTOParserTargetArch(context)) {
  case PTOParserTargetArch::A3:
    return BLayout::RowMajor;
  case PTOParserTargetArch::A5:
    return BLayout::ColMajor;
  case PTOParserTargetArch::Unspecified:
    return parsedLayout;
  }
  return parsedLayout;
}

TileBufConfigAttr TileBufType::getConfigAttr() const {
  // 情况 A：getConfig() 已经是 TileBufConfigAttr
  if constexpr (std::is_same_v<decltype(getConfig()), TileBufConfigAttr>) {
    auto cfg = getConfig();
    if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());
    return cfg;
  } else {
    // 情况 B：getConfig() 是 Attribute
    auto cfg = llvm::dyn_cast_or_null<TileBufConfigAttr>(getConfig());
    if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());
    return cfg;
  }
}
bool TileBufType::hasNonDefaultConfig() const {
  return !getConfigAttr().isDefault();
}

mlir::Attribute TileBufType::getBLayoutAttr() const { return getConfigAttr().getBLayout(); }
mlir::Attribute TileBufType::getSLayoutAttr() const { return getConfigAttr().getSLayout(); }
mlir::Attribute TileBufType::getPadValueAttr() const { return getConfigAttr().getPad(); }
mlir::Attribute TileBufType::getCompactModeAttr() const {
  return getConfigAttr().getCompactMode();
}

// ✅ numeric getters（可选）
int32_t TileBufType::getSFractalSizeI32() const {
  return (int32_t)getConfigAttr().getSFractalSize().getInt();
}

int32_t TileBufType::getBLayoutValueI32() const {
  if (auto a = llvm::dyn_cast<BLayoutAttr>(getBLayoutAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getSLayoutValueI32() const {
  if (auto a = llvm::dyn_cast<SLayoutAttr>(getSLayoutAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getPadValueI32() const {
  if (auto a = llvm::dyn_cast<PadValueAttr>(getPadValueAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

int32_t TileBufType::getCompactModeI32() const {
  if (auto a = llvm::dyn_cast<CompactModeAttr>(getCompactModeAttr()))
    return static_cast<int32_t>(a.getValue());
  return 0;
}

namespace {

struct ParsedTileBufFields {
  std::string locStr;
  Type dtype;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t vrow = -1;
  int64_t vcol = -1;
  std::string blayoutStr;
  std::string slayoutStr;
  int64_t fractal = 0;
  uint32_t padInt = 0;
  uint32_t compactInt = 0;
};

struct ParsedTileBufCompactFlags {
  bool seenValid = false;
  bool seenBLayout = false;
  bool seenSLayout = false;
  bool seenFractal = false;
  bool seenPad = false;
  bool seenCompact = false;
};

struct SymbolizedTileBufConfig {
  AddressSpace memorySpace = AddressSpace::Zero;
  BLayout bLayout = BLayout::RowMajor;
  SLayout sLayout = SLayout::NoneBox;
  PadValue pad = PadValue::Null;
  CompactMode compact = CompactMode::Null;
};

struct TileBufPrintState {
  int64_t rows = ShapedType::kDynamic;
  int64_t cols = ShapedType::kDynamic;
  int64_t vrow = ShapedType::kDynamic;
  int64_t vcol = ShapedType::kDynamic;
  TileBufConfigAttr config;
  TileBufConfigAttr defaultConfig;
  bool printValid = false;
  bool printBLayout = false;
  bool printSLayout = false;
  bool printFractal = false;
  bool printPad = false;
  bool printCompact = false;
};

static LogicalResult parseTileBufUInt32Value(AsmParser &parser, StringRef key,
                                             uint32_t &value) {
  int64_t parsedValue = 0;
  if (failed(parser.parseInteger(parsedValue)))
    return failure();
  if (parsedValue < 0 ||
      parsedValue > std::numeric_limits<uint32_t>::max()) {
    parser.emitError(parser.getCurrentLocation())
        << key << " must be a non-negative 32-bit integer";
    return failure();
  }
  value = static_cast<uint32_t>(parsedValue);
  return success();
}

static LogicalResult parseLegacyTileBufFields(AsmParser &parser,
                                              ParsedTileBufFields &fields) {
  if (failed(parser.parseEqual()))
    return failure();
  if (failed(parser.parseKeywordOrString(&fields.locStr)))
    return failure();
  if (failed(parser.parseComma()))
    return failure();

  if (failed(parseTileBufTypeField(parser, "dtype", fields.dtype)) ||
      failed(parseTileBufIntegerField(parser, "rows", fields.rows)) ||
      failed(parseTileBufIntegerField(parser, "cols", fields.cols)) ||
      failed(parseTileBufValidShapeFields(parser, fields.vrow, fields.vcol)) ||
      failed(parseTileBufKeywordField(parser, "blayout", fields.blayoutStr)) ||
      failed(parseTileBufKeywordField(parser, "slayout", fields.slayoutStr)) ||
      failed(parseTileBufIntegerField(parser, "fractal", fields.fractal)) ||
      failed(parseTileBufPadField(parser, fields.padInt))) {
    return failure();
  }

  return success();
}

static LogicalResult initializeCompactTileBufDefaults(AsmParser &parser,
                                                      ParsedTileBufFields &fields) {
  auto defaultConfig = TileBufConfigAttr::getDefault(parser.getContext());
  auto defaultBLayout = llvm::dyn_cast<BLayoutAttr>(defaultConfig.getBLayout());
  auto defaultSLayout = llvm::dyn_cast<SLayoutAttr>(defaultConfig.getSLayout());
  auto defaultPad = llvm::dyn_cast<PadValueAttr>(defaultConfig.getPad());
  auto defaultCompact =
      llvm::dyn_cast<CompactModeAttr>(defaultConfig.getCompactMode());
  if (!defaultBLayout || !defaultSLayout || !defaultPad || !defaultCompact) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to load default tile_buf config");
    return failure();
  }

  fields.blayoutStr = stringifyBLayout(defaultBLayout.getValue()).str();
  fields.slayoutStr = stringifySLayout(defaultSLayout.getValue()).str();
  fields.fractal = defaultConfig.getSFractalSize().getInt();
  fields.padInt = static_cast<uint32_t>(defaultPad.getValue());
  fields.compactInt = static_cast<uint32_t>(defaultCompact.getValue());
  return success();
}

static LogicalResult parseCompactTileBufValidField(AsmParser &parser,
                                                   ParsedTileBufFields &fields) {
  SmallVector<int64_t, 2> validShape;
  if (failed(parser.parseDimensionList(validShape, /*allowDynamic=*/true,
                                       /*withTrailingX=*/false))) {
    return failure();
  }
  if (validShape.size() != 2) {
    parser.emitError(parser.getCurrentLocation(),
                     "tile_buf valid must have exactly two dims");
    return failure();
  }
  fields.vrow = validShape[0];
  fields.vcol = validShape[1];
  return success();
}

static LogicalResult parseCompactTileBufLayoutField(AsmParser &parser,
                                                    StringRef key,
                                                    std::string &value,
                                                    bool &seen) {
  if (seen) {
    parser.emitError(parser.getCurrentLocation())
        << "duplicate " << key << " in tile_buf compact syntax";
    return failure();
  }
  seen = true;
  return parser.parseKeywordOrString(&value);
}

static LogicalResult parseCompactTileBufIntegerOption(AsmParser &parser,
                                                      StringRef key,
                                                      int64_t &value,
                                                      bool &seen) {
  if (seen) {
    parser.emitError(parser.getCurrentLocation())
        << "duplicate " << key << " in tile_buf compact syntax";
    return failure();
  }
  seen = true;
  return parser.parseInteger(value);
}

static LogicalResult parseCompactTileBufUInt32Option(AsmParser &parser,
                                                     StringRef key,
                                                     uint32_t &value,
                                                     bool &seen) {
  if (seen) {
    parser.emitError(parser.getCurrentLocation())
        << "duplicate " << key << " in tile_buf compact syntax";
    return failure();
  }
  seen = true;
  return parseTileBufUInt32Value(parser, key, value);
}

static LogicalResult parseCompactTileBufField(AsmParser &parser, StringRef key,
                                              ParsedTileBufFields &fields,
                                              ParsedTileBufCompactFlags &flags) {
  if (key == "valid") {
    if (flags.seenValid) {
      parser.emitError(parser.getCurrentLocation(),
                       "duplicate valid in tile_buf compact syntax");
      return failure();
    }
    flags.seenValid = true;
    return parseCompactTileBufValidField(parser, fields);
  }
  if (key == "blayout") {
    return parseCompactTileBufLayoutField(parser, key, fields.blayoutStr,
                                          flags.seenBLayout);
  }
  if (key == "slayout") {
    return parseCompactTileBufLayoutField(parser, key, fields.slayoutStr,
                                          flags.seenSLayout);
  }
  if (key == "fractal") {
    return parseCompactTileBufIntegerOption(parser, key, fields.fractal,
                                            flags.seenFractal);
  }
  if (key == "pad") {
    return parseCompactTileBufUInt32Option(parser, key, fields.padInt,
                                           flags.seenPad);
  }
  if (key == "compact") {
    return parseCompactTileBufUInt32Option(parser, key, fields.compactInt,
                                           flags.seenCompact);
  }

  parser.emitError(parser.getCurrentLocation(),
                   "unknown key in tile_buf compact syntax: ")
      << key;
  return failure();
}

static LogicalResult parseCompactTileBufFields(AsmParser &parser,
                                               StringRef firstToken,
                                               ParsedTileBufFields &fields) {
  fields.locStr = firstToken.str();

  if (failed(parser.parseComma()))
    return failure();

  SmallVector<int64_t, 2> shape;
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/false)))
    return failure();
  if (failed(parser.parseType(fields.dtype)))
    return failure();
  if (shape.size() != 2) {
    parser.emitError(parser.getCurrentLocation(),
                     "tile_buf compact syntax expects exactly two shape dims");
    return failure();
  }

  fields.rows = shape[0];
  fields.cols = shape[1];
  fields.vrow = fields.rows;
  fields.vcol = fields.cols;

  if (failed(initializeCompactTileBufDefaults(parser, fields)))
    return failure();

  ParsedTileBufCompactFlags flags;

  while (succeeded(parser.parseOptionalComma())) {
    StringRef key;
    if (failed(parser.parseKeyword(&key)) || failed(parser.parseEqual()))
      return failure();
    if (failed(parseCompactTileBufField(parser, key, fields, flags)))
      return failure();
  }

  return success();
}

static LogicalResult
symbolizeTileBufConfig(AsmParser &parser, const ParsedTileBufFields &fields,
                       SymbolizedTileBufConfig &config) {
  auto memorySpace = resolveTileBufMemorySpace(fields.locStr);
  if (!memorySpace.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown loc: ") << fields.locStr;
    return failure();
  }

  auto bl = symbolizeBLayout(fields.blayoutStr);
  auto sl = symbolizeSLayout(fields.slayoutStr);
  auto pv = symbolizePadValue(fields.padInt);
  auto compact = symbolizeCompactMode(fields.compactInt);
  if (!bl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown blayout: ")
        << fields.blayoutStr;
    return failure();
  }
  if (!sl.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown slayout: ")
        << fields.slayoutStr;
    return failure();
  }
  if (!pv.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown pad: ") << fields.padInt;
    return failure();
  }
  if (!compact.has_value()) {
    parser.emitError(parser.getNameLoc(), "unknown compact: ")
        << fields.compactInt;
    return failure();
  }

  config.memorySpace = memorySpace.value();
  config.bLayout =
      resolveTileBufBLayout(parser.getContext(), config.memorySpace, bl.value());
  config.sLayout = sl.value();
  config.pad = pv.value();
  config.compact = compact.value();
  return success();
}

static TileBufConfigAttr buildTileBufConfigAttr(MLIRContext *ctx,
                                                const ParsedTileBufFields &fields,
                                                const SymbolizedTileBufConfig &config) {
  auto blAttr = BLayoutAttr::get(ctx, config.bLayout);
  auto slAttr = SLayoutAttr::get(ctx, config.sLayout);
  auto fractalAttr =
      IntegerAttr::get(IntegerType::get(ctx, 32), fields.fractal);
  auto padAttr = PadValueAttr::get(ctx, config.pad);
  auto compactAttr = CompactModeAttr::get(ctx, config.compact);
  return TileBufConfigAttr::get(ctx, blAttr, slAttr, fractalAttr, padAttr,
                                compactAttr);
}

static Type buildTileBufType(AsmParser &parser,
                             const ParsedTileBufFields &fields) {
  MLIRContext *ctx = parser.getContext();

  if (fields.rows < 0 || fields.cols < 0) {
    parser.emitError(parser.getNameLoc(), "rows/cols must be non-negative");
    return Type();
  }

  SymbolizedTileBufConfig config;
  if (failed(symbolizeTileBufConfig(parser, fields, config)))
    return Type();

  SmallVector<int64_t, 2> shape{fields.rows, fields.cols};
  SmallVector<int64_t, 2> validShape{fields.vrow, fields.vcol};
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto memorySpaceAttr = AddressSpaceAttr::get(ctx, config.memorySpace);
  auto cfg = buildTileBufConfigAttr(ctx, fields, config);

  return TileBufType::get(ctx, shape, fields.dtype, memorySpaceAttr,
                          llvm::ArrayRef<int64_t>(canonicalValidShape), cfg);
}

} // namespace

// ---- TileBufType custom asm ----
// !pto.tile_buf<<loc=.., dtype=.., rows=.., cols=.., blayout=.., valid=..x..,
//                slayout=.., fractal=.., pad=.., compact=..>>
Type TileBufType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();

  std::string firstToken;
  if (failed(parser.parseKeywordOrString(&firstToken)))
    return Type();

  ParsedTileBufFields fields;
  const bool isLegacySyntax = firstToken == "loc";
  if (isLegacySyntax) {
    if (failed(parseLegacyTileBufFields(parser, fields)))
      return Type();
  } else {
    if (failed(parseCompactTileBufFields(parser, firstToken, fields)))
      return Type();
  }

  if (isLegacySyntax && succeeded(parser.parseOptionalComma())) {
    if (failed(parseTileBufKeyEq(parser, "compact")) ||
        failed(parseTileBufUInt32Value(parser, "compact", fields.compactInt))) {
      return Type();
    }
  }

  if (failed(parser.parseGreater()))
    return Type();

  return buildTileBufType(parser, fields);
}

static llvm::StringRef stringifyLocFromMemorySpace(mlir::Attribute memorySpace) {
  auto asAttr = llvm::dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  switch (asAttr.getAddressSpace()) {
    case AddressSpace::MAT: return "mat";
    case AddressSpace::LEFT: return "left";
    case AddressSpace::RIGHT: return "right";
    case AddressSpace::ACC: return "acc";
    case AddressSpace::VEC: return "vec";
    case AddressSpace::BIAS: return "bias";
    case AddressSpace::SCALING: return "scaling";
    default: return "illegal";
  }
}

static llvm::StringRef stringifyLocFromPad(mlir::Attribute pad) {
  auto padAttr = llvm::dyn_cast_or_null<PadValueAttr>(pad);
  if (!padAttr) return "9999";

  switch (padAttr.getValue()) {
    case PadValue::Null: return "0";
    case PadValue::Zero: return "1";
    case PadValue::Max: return "2";
    case PadValue::Min: return "3";
    default:
      return "9999";
  }
}

static llvm::StringRef stringifyCompactModeInt(mlir::Attribute compactMode) {
  auto compactAttr = llvm::dyn_cast_or_null<CompactModeAttr>(compactMode);
  if (!compactAttr)
    return "9999";

  switch (compactAttr.getValue()) {
  case CompactMode::Null:
    return "0";
  case CompactMode::Normal:
    return "1";
  case CompactMode::RowPlusOne:
    return "2";
  default:
    return "9999";
  }
}

static void printTileBufDim(AsmPrinter &printer, int64_t dim) {
  if (dim == ShapedType::kDynamic)
    printer << "?";
  else
    printer << dim;
}

static TileBufPrintState buildTileBufPrintState(TileBufType type) {
  TileBufPrintState state;
  auto shape = type.getShape();
  state.rows = shape.size() > 0 ? shape[0] : ShapedType::kDynamic;
  state.cols = shape.size() > 1 ? shape[1] : ShapedType::kDynamic;
  state.config = type.getConfigAttr();
  if (!state.config)
    state.config = mlir::pto::TileBufConfigAttr::getDefault(type.getContext());
  state.defaultConfig = TileBufConfigAttr::getDefault(type.getContext());

  auto validShape = type.getValidShape();
  state.vrow = state.rows;
  state.vcol = state.cols;
  if (validShape.size() >= 2) {
    state.vrow = validShape[0];
    state.vcol = validShape[1];
  }

  auto blayout = llvm::dyn_cast<BLayoutAttr>(state.config.getBLayout());
  auto slayout = llvm::dyn_cast<SLayoutAttr>(state.config.getSLayout());
  auto pad = llvm::dyn_cast<PadValueAttr>(state.config.getPad());
  auto compact = llvm::dyn_cast<CompactModeAttr>(state.config.getCompactMode());
  auto defaultBLayout =
      llvm::dyn_cast<BLayoutAttr>(state.defaultConfig.getBLayout());
  auto defaultSLayout =
      llvm::dyn_cast<SLayoutAttr>(state.defaultConfig.getSLayout());
  auto defaultPad = llvm::dyn_cast<PadValueAttr>(state.defaultConfig.getPad());
  auto defaultCompact =
      llvm::dyn_cast<CompactModeAttr>(state.defaultConfig.getCompactMode());

  state.printValid = state.vrow != state.rows || state.vcol != state.cols;
  state.printBLayout =
      blayout && defaultBLayout &&
      blayout.getValue() != defaultBLayout.getValue();
  state.printSLayout =
      slayout && defaultSLayout &&
      slayout.getValue() != defaultSLayout.getValue();
  state.printFractal = state.config.getSFractalSize().getInt() !=
                       state.defaultConfig.getSFractalSize().getInt();
  state.printPad = pad && defaultPad && pad.getValue() != defaultPad.getValue();
  state.printCompact =
      compact && defaultCompact &&
      compact.getValue() != defaultCompact.getValue();
  return state;
}

static void printTileBufShapeAndType(AsmPrinter &printer, TileBufType type,
                                     const TileBufPrintState &state) {
  printer << "<" << stringifyLocFromMemorySpace(type.getMemorySpace()) << ", ";
  printTileBufDim(printer, state.rows);
  printer << "x";
  printTileBufDim(printer, state.cols);
  printer << "x";
  printer.printType(type.getElementType());
}

static void printTileBufOptionalFields(AsmPrinter &printer,
                                       const TileBufPrintState &state) {
  auto config = state.config;
  if (state.printValid) {
    printer << ", valid=";
    printTileBufDim(printer, state.vrow);
    printer << "x";
    printTileBufDim(printer, state.vcol);
  }
  if (state.printBLayout)
    printer << ", blayout="
            << stringifyBLayout(
                   llvm::cast<BLayoutAttr>(config.getBLayout()).getValue());
  if (state.printSLayout)
    printer << ", slayout="
            << stringifySLayout(
                   llvm::cast<SLayoutAttr>(config.getSLayout()).getValue());
  if (state.printFractal)
    printer << ", fractal=" << config.getSFractalSize().getInt();
  if (state.printPad)
    printer << ", pad=" << stringifyLocFromPad(config.getPad());
  if (state.printCompact)
    printer << ", compact="
            << stringifyCompactModeInt(config.getCompactMode());
}

void mlir::pto::TileBufType::print(mlir::AsmPrinter &printer) const {
  TileBufPrintState state = buildTileBufPrintState(*this);
  printTileBufShapeAndType(printer, *this, state);
  printTileBufOptionalFields(printer, state);
  printer << ">";
}
