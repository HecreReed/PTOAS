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

// === TMatmulBiasMxOp ===
// Read: a, b, bias, Write: dst
void TMatmulMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

static bool isInsideSectionCube(Operation *op) {
  return op->getParentOfType<pto::SectionCubeOp>() != nullptr;
}

static bool isInsideSectionVector(Operation *op) {
  return op->getParentOfType<pto::SectionVectorOp>() != nullptr;
}

static std::optional<FunctionKernelKind>
getEnclosingFunctionKernelKind(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return std::nullopt;

  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  return kernelKindAttr.getKernelKind();
}

static bool isInsideSectionOrAttributedKernel(Operation *op) {
  return isInsideSectionCube(op) || isInsideSectionVector(op) ||
         getEnclosingFunctionKernelKind(op).has_value();
}

static LogicalResult verifySplitAttr(Operation *op, int64_t split) {
  if (split < 0 || split > 2)
    return op->emitOpError("expects 'split' to be 0, 1, or 2");
  return success();
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  if (!kernelKind || *kernelKind != expected) {
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function";
  }
  return success();
}

struct FrontendInitAttrState {
  bool sawId = false;
  bool sawDirMask = false;
  bool sawSlotSize = false;
  bool sawLocalSlotNum = false;
  bool sawNoSplit = false;
};

static ParseResult parseFrontendInitI32AttrClause(OpAsmParser &parser,
                                                  NamedAttrList &attrs,
                                                  bool &seen, StringRef keyword,
                                                  StringRef attrName,
                                                  Type attrType) {
  if (seen) {
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate '" << keyword << "' clause";
  }
  IntegerAttr valueAttr;
  if (parser.parseAttribute(valueAttr, attrType, attrName, attrs))
    return failure();
  seen = true;
  return success();
}

static ParseResult parseFrontendInitBoolAttrClause(OpAsmParser &parser,
                                                   NamedAttrList &attrs,
                                                   bool &seen,
                                                   StringRef keyword,
                                                   StringRef attrName) {
  if (seen) {
    return parser.emitError(parser.getCurrentLocation())
           << "duplicate '" << keyword << "' clause";
  }
  BoolAttr valueAttr;
  if (parser.parseAttribute(valueAttr, attrName, attrs))
    return failure();
  seen = true;
  return success();
}

static ParseResult parseFrontendInitializePipeAttrClause(
    OpAsmParser &parser, StringRef keyword, NamedAttrList &attrs,
    FrontendInitAttrState &state) {
  if (keyword == "id") {
    return parseFrontendInitI32AttrClause(parser, attrs, state.sawId, keyword,
                                          "id",
                                          parser.getBuilder().getI32Type());
  }
  if (keyword == "dir_mask") {
    return parseFrontendInitI32AttrClause(parser, attrs, state.sawDirMask,
                                          keyword, "dir_mask",
                                          parser.getBuilder().getI8Type());
  }
  if (keyword == "slot_size") {
    return parseFrontendInitI32AttrClause(
        parser, attrs, state.sawSlotSize, keyword, "slot_size",
        parser.getBuilder().getI32Type());
  }
  if (keyword == "local_slot_num") {
    return parseFrontendInitI32AttrClause(
        parser, attrs, state.sawLocalSlotNum, keyword, "local_slot_num",
        parser.getBuilder().getI32Type());
  }
  if (keyword == "nosplit") {
    return parseFrontendInitBoolAttrClause(parser, attrs, state.sawNoSplit,
                                           keyword, "nosplit");
  }
  return parser.emitError(parser.getCurrentLocation())
         << "unexpected keyword '" << keyword << "'";
}

static ParseResult parseFrontendInitializePipeAttrs(OpAsmParser &parser,
                                                    NamedAttrList &attrs,
                                                    FrontendInitAttrState &state) {
  if (parser.parseLBrace())
    return failure();
  while (failed(parser.parseOptionalRBrace())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual() ||
        failed(parseFrontendInitializePipeAttrClause(parser, keyword, attrs,
                                                     state))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalRBrace()))
      break;
    if (parser.parseComma())
      return failure();
  }
  if (!state.sawDirMask)
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  if (!state.sawSlotSize)
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  if (!state.sawId)
    attrs.set("id", parser.getBuilder().getI32IntegerAttr(0));
  return success();
}

struct FrontendInitOperandState {
  OpAsmParser::UnresolvedOperand gmSlotBuffer;
  OpAsmParser::UnresolvedOperand gmSlotTensor;
  OpAsmParser::UnresolvedOperand c2vConsumerBuf;
  OpAsmParser::UnresolvedOperand v2cConsumerBuf;
  Type gmSlotBufferTy;
  Type gmSlotTensorTy;
  Type c2vConsumerBufTy;
  Type v2cConsumerBufTy;
  bool hasGmSlotBuffer = false;
  bool hasGmSlotTensor = false;
  bool hasC2vConsumerBuf = false;
  bool hasV2cConsumerBuf = false;
};

static ParseResult parseFrontendInitializePipeOperandClause(
    OpAsmParser &parser, StringRef keyword, FrontendInitOperandState &state) {
  if (keyword == "gm_slot_buffer") {
    if (state.hasGmSlotBuffer) {
      return parser.emitError(parser.getCurrentLocation(),
                              "duplicate 'gm_slot_buffer' operand");
    }
    if (parser.parseOperand(state.gmSlotBuffer) ||
        parser.parseColonType(state.gmSlotBufferTy))
      return failure();
    state.hasGmSlotBuffer = true;
    return success();
  }
  if (keyword == "gm_slot_tensor") {
    if (state.hasGmSlotTensor) {
      return parser.emitError(parser.getCurrentLocation(),
                              "duplicate 'gm_slot_tensor' operand");
    }
    if (parser.parseOperand(state.gmSlotTensor) ||
        parser.parseColonType(state.gmSlotTensorTy))
      return failure();
    state.hasGmSlotTensor = true;
    return success();
  }
  if (keyword == "c2v_consumer_buf") {
    if (state.hasC2vConsumerBuf) {
      return parser.emitError(parser.getCurrentLocation(),
                              "duplicate 'c2v_consumer_buf' operand");
    }
    if (parser.parseOperand(state.c2vConsumerBuf) ||
        parser.parseColonType(state.c2vConsumerBufTy))
      return failure();
    state.hasC2vConsumerBuf = true;
    return success();
  }
  if (keyword == "v2c_consumer_buf") {
    if (state.hasV2cConsumerBuf) {
      return parser.emitError(parser.getCurrentLocation(),
                              "duplicate 'v2c_consumer_buf' operand");
    }
    if (parser.parseOperand(state.v2cConsumerBuf) ||
        parser.parseColonType(state.v2cConsumerBufTy))
      return failure();
    state.hasV2cConsumerBuf = true;
    return success();
  }
  return parser.emitError(parser.getCurrentLocation())
         << "unexpected initialize_pipe operand '" << keyword << "'";
}

static ParseResult parseFrontendInitializePipeOperands(
    OpAsmParser &parser, FrontendInitOperandState &state) {
  if (parser.parseLParen())
    return failure();
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual() ||
        failed(parseFrontendInitializePipeOperandClause(parser, keyword, state))) {
      return failure();
    }
    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (parser.parseComma())
      return failure();
  }
  return success();
}

static ParseResult resolveFrontendInitializePipeOperands(
    OpAsmParser &parser, OperationState &result, NamedAttrList &attrs,
    const FrontendInitOperandState &state) {
  result.addAttributes(attrs);
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {state.hasGmSlotBuffer ? 1 : 0, state.hasGmSlotTensor ? 1 : 0,
           state.hasC2vConsumerBuf ? 1 : 0, state.hasV2cConsumerBuf ? 1 : 0}));
  if (state.hasGmSlotBuffer &&
      parser.resolveOperand(state.gmSlotBuffer, state.gmSlotBufferTy,
                            result.operands))
    return failure();
  if (state.hasGmSlotTensor &&
      parser.resolveOperand(state.gmSlotTensor, state.gmSlotTensorTy,
                            result.operands))
    return failure();
  if (state.hasC2vConsumerBuf &&
      parser.resolveOperand(state.c2vConsumerBuf, state.c2vConsumerBufTy,
                            result.operands))
    return failure();
  if (state.hasV2cConsumerBuf &&
      parser.resolveOperand(state.v2cConsumerBuf, state.v2cConsumerBufTy,
                            result.operands))
    return failure();
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  NamedAttrList attrs;
  FrontendInitAttrState attrState;
  FrontendInitOperandState operandState;
  if (failed(parseFrontendInitializePipeAttrs(parser, attrs, attrState)) ||
      failed(parseFrontendInitializePipeOperands(parser, operandState)) ||
      parser.parseOptionalAttrDict(attrs) ||
      failed(resolveFrontendInitializePipeOperands(parser, result, attrs,
                                                  operandState))) {
    return failure();
  }
  return success();
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&](StringRef keyword, auto value) {
    if (needsComma)
      p << ", ";
    p << keyword << " = " << value;
    needsComma = true;
  };

  if (op.getId() != 0)
    printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr())
    printClause("local_slot_num", localSlotNumAttr.getInt());
  if (auto noSplitAttr = op.getNosplitAttr())
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  p << "}";

  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&](StringRef keyword, Value value) {
    if (needsOperandComma)
      p << ", ";
    p << keyword << " = " << value << " : " << value.getType();
    needsOperandComma = true;
  };
  if (op.getGmSlotBuffer()) {
    printOperandClause("gm_slot_buffer", op.getGmSlotBuffer());
  }
  if (op.getGmSlotTensor())
    printOperandClause("gm_slot_tensor", op.getGmSlotTensor());
  if (op.getC2vConsumerBuf())
    printOperandClause("c2v_consumer_buf", op.getC2vConsumerBuf());
  if (op.getV2cConsumerBuf())
    printOperandClause("v2c_consumer_buf", op.getV2cConsumerBuf());
  p << ")";
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "local_slot_num",
                       "nosplit", "operandSegmentSizes"});
}

static std::optional<uint64_t>
getStaticElementCount(ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static bool isSameOrHalfSlotByteSize(uint64_t tensorBytes, uint64_t slotBytes) {
  return tensorBytes == slotBytes || tensorBytes * 2 == slotBytes;
}

static LogicalResult verifyFrontendGlobalSlotTensor(Operation *op, Value tensor,
                                                    int8_t dirMask,
                                                    int32_t slotSize) {
  (void)dirMask;
  auto tvTy = dyn_cast<TensorViewType>(tensor.getType());
  if (!tvTy)
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");

  ArrayRef<int64_t> shape = tvTy.getShape();
  if (shape.empty())
    return op->emitOpError(
        "expects 'gm_slot_tensor' to describe one slot entry tensor");

  if (auto elemCount = getStaticElementCount(shape)) {
    uint64_t elemBytes = getElemByteSize(tvTy.getElementType());
    if (elemBytes != 0) {
      uint64_t tensorBytes = *elemCount * elemBytes;
      if (!isSameOrHalfSlotByteSize(tensorBytes,
                                    static_cast<uint64_t>(slotSize))) {
        return op->emitOpError()
               << "expects 'slot_size' to equal gm_slot_tensor byte size "
                  "or twice gm_slot_tensor byte size for split GlobalTensor "
                  "entries (got slot_size = "
               << slotSize << ", gm_slot_tensor byte size = " << tensorBytes
               << ")";
      }
    }
  }

  return success();
}

template <typename InitOpT>
static unsigned countFrontendInitOpsWithSameId(func::FuncOp funcOp, uint32_t id) {
  unsigned sameIdInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == id)
        ++sameIdInitCount;
      return;
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == id)
        ++sameIdInitCount;
    }
  });
  return sameIdInitCount;
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitGlobalTensorForm(
    InitOpT op, int8_t dirMask, bool hasC2vConsumerBuf, bool hasV2cConsumerBuf) {
  if (op.getGmSlotBuffer() || hasC2vConsumerBuf || hasV2cConsumerBuf) {
    return op.emitOpError(
        "globaltensor pipe init expects only 'gm_slot_tensor' and no "
        "'gm_slot_buffer', 'c2v_consumer_buf', or 'v2c_consumer_buf'");
  }
  if (op.getLocalSlotNumAttr())
    return op.emitOpError("globaltensor pipe init does not use 'local_slot_num'");
  if (getTargetArch(op.getOperation()) == PTOArch::A5) {
    return op.emitOpError(
        "globaltensor pipe entries are supported for a2/a3 l2g2l pipes");
  }
  return verifyFrontendGlobalSlotTensor(op.getOperation(), op.getGmSlotTensor(),
                                        dirMask, op.getSlotSize());
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitLocalPipeForm(
    InitOpT op, int8_t dirMask, bool hasC2vConsumerBuf, bool hasV2cConsumerBuf) {
  if (hasC2vConsumerBuf != hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects 'c2v_consumer_buf' and 'v2c_consumer_buf' to be provided together");
  }
  if (!hasC2vConsumerBuf) {
    return op.emitOpError(
        "expects local pipe init to provide 'c2v_consumer_buf' and "
        "'v2c_consumer_buf'; use 'gm_slot_tensor' for globaltensor pipe entries");
  }
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return op.emitOpError("expects 'local_slot_num' to be greater than 0");
    int32_t loweredSlotNum = dirMask == 3 ? 4 : 8;
    if (localSlotNum > loweredSlotNum) {
      return op.emitOpError()
             << "expects 'local_slot_num' to be less than or equal to "
             << loweredSlotNum << " for dir_mask = " << static_cast<int>(dirMask);
    }
  }
  return success();
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (failed(verifyFrontendKernelKind(op.getOperation(), expected, kernelName)))
    return failure();

  auto funcOp = op->template getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op.emitOpError("must be nested under a func.func");

  if (op.getId() < 0)
    return op.emitOpError("expects 'id' to be non-negative");

  unsigned sameIdInitCount = countFrontendInitOpsWithSameId<InitOpT>(
      funcOp, op.getId());
  if (sameIdInitCount > 1) {
    return op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
  }

  int8_t dirMask = op.getDirMask();
  if (dirMask != 1 && dirMask != 2 && dirMask != 3)
    return op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (op.getSlotSize() <= 0)
    return op.emitOpError("expects 'slot_size' to be greater than 0");

  bool hasGlobalSlotTensor = static_cast<bool>(op.getGmSlotTensor());
  bool hasC2vConsumerBuf = static_cast<bool>(op.getC2vConsumerBuf());
  bool hasV2cConsumerBuf = static_cast<bool>(op.getV2cConsumerBuf());
  if (hasGlobalSlotTensor) {
    return verifyFrontendInitGlobalTensorForm(op, dirMask, hasC2vConsumerBuf,
                                              hasV2cConsumerBuf);
  }
  return verifyFrontendInitLocalPipeForm(op, dirMask, hasC2vConsumerBuf,
                                         hasV2cConsumerBuf);
}

ParseResult AicInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AicInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ParseResult AivInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AivInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

static ReserveBufferOp findReserveBufferByName(func::FuncOp funcOp,
                                               StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name)
      return WalkResult::advance();
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested under a func.func");

  if (getSize() <= 0)
    return emitOpError("expects 'size' to be greater than 0");

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT)
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");

  if (!getAutoAlloc() && !getBaseAttr())
    return emitOpError("expects 'base' when 'auto' is false");

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0)
    return emitOpError("expects 'base' to be non-negative when present");

  unsigned sameNameCount = 0;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName())
      ++sameNameCount;
  });
  if (sameNameCount > 1)
    return emitOpError("requires 'name' to be unique within the function");

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested under a func.func");

  auto peerFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      getOperation(), getPeerFuncAttr());
  if (!peerFunc)
    return emitOpError("expects 'peer_func' to reference an existing func.func");

  unsigned sameImportCount = 0;
  funcOp.walk([&](ImportReservedBufferOp importOp) {
    if (importOp.getName() == getName() &&
        importOp.getPeerFuncAttr() == getPeerFuncAttr()) {
      ++sameImportCount;
    }
  });
  if (sameImportCount > 1) {
    return emitOpError(
        "requires (name, peer_func) to be unique within the function");
  }

  if (!findReserveBufferByName(peerFunc, getName()))
    return emitOpError("expects matching peer reserve_buffer to exist");

  return success();
}

static FailureOr<Operation *> lookupFrontendInitOpById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName)))
    return failure();
  if (id < 0)
    return op->emitOpError("expects 'id' to be non-negative");
  return verifySplitAttr(op, split);
}

static FailureOr<int8_t> lookupFrontendInitDirMaskById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr))
    return failure();
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr))
    return aic.getDirMask();
  return cast<AivInitializePipeOp>(*initOr).getDirMask();
}

static LogicalResult verifyFrontendDataOpDirection(Operation *op, int32_t id,
                                                   bool expectC2V) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op->emitOpError("must be nested under a func.func");

  auto dirMaskOr = lookupFrontendInitDirMaskById(op, funcOp, id);
  if (failed(dirMaskOr))
    return failure();

  int8_t dirMask = *dirMaskOr;
  if (expectC2V && dirMask != 1 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != 2 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 2 or 3";
  }
  return success();
}

static Value getFrontendInitGmSlotTensor(Operation *initOp) {
  if (auto aic = dyn_cast<AicInitializePipeOp>(initOp))
    return aic.getGmSlotTensor();
  return cast<AivInitializePipeOp>(initOp).getGmSlotTensor();
}

static LogicalResult verifyFrontendTensorEntryMatchesInit(Operation *op,
                                                          int32_t id,
                                                          Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op->emitOpError("must be nested under a func.func");

  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr))
    return failure();
  Value gmSlotTensor = getFrontendInitGmSlotTensor(*initOr);
  if (!gmSlotTensor) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with 'gm_slot_tensor' when the "
              "pipe entry is !pto.tensor_view";
  }

  auto slotTensorTy = dyn_cast<TensorViewType>(gmSlotTensor.getType());
  if (!slotTensorTy)
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  if (slotTensorTy.getElementType() != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match gm_slot_tensor element type";
  }
  if (slotTensorTy.getRank() != entryViewTy.getRank()) {
    return op->emitOpError()
           << "expects pipe entry rank to match gm_slot_tensor rank";
  }

  ArrayRef<int64_t> slotShape = slotTensorTy.getShape();
  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim)
      continue;
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match gm_slot_tensor dimension " << slotDim;
  }
  return success();
}

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopOp(FrontendPopOpT op,
                                         FunctionKernelKind expected,
                                         StringRef kernelName,
                                         bool expectC2V) {
  if (failed(verifyFrontendSplitOp(op.getOperation(), expected, kernelName,
                                   op.getId(),
                                   op.getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(op.getOperation(), op.getId(),
                                           expectC2V)))
    return failure();
  if (failed(verifyFrontendTensorEntryMatchesInit(op.getOperation(), op.getId(),
                                                  op.getTile().getType())))
    return failure();

  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol)
    return op.emitOpError(
        "expects valid_row and valid_col operands to be provided together");
  if (!hasValidRow)
    return success();

  if (isa<TensorViewType>(op.getTile().getType()))
    return op.emitOpError(
        "does not accept valid_row/valid_col when result is !pto.tensor_view");

  auto tileTy = dyn_cast<TileBufType>(op.getTile().getType());
  if (!tileTy)
    return op.emitOpError(
        "expects tile result to be !pto.tile_buf when valid_row/valid_col operands are provided");
  if (!tileTy.hasDynamicValid())
    return op.emitOpError(
        "expects tile result to have dynamic validShape (?, ?) when valid_row/valid_col operands are provided");
  return success();
}

static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  if (dirMask != 1 && dirMask != 2 && dirMask != 3)
    return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (slotSize <= 0)
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  if (slotNum != 4 && slotNum != 8)
    return op->emitOpError("expects 'slot_num' to be 4 or 8");
  if (flagBase && *flagBase < 0)
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  if (flagBase) {
    int32_t flagWidth = dirMask == 3 ? 4 : 2;
    if (*flagBase + flagWidth > kMaxHardwareFlagIds) {
      return op->emitOpError()
             << "requires 'flag_base' and dir_mask to fit within "
             << kMaxHardwareFlagIds << " hardware flag ids";
    }
  }

  return success();
}

static LogicalResult verifyPipeHandleProducer(Operation *op, Value pipeHandle) {
  if (!isa<pto::PipeType>(pipeHandle.getType()))
    return op->emitOpError("expects pipe operand type !pto.pipe");
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
  }
  return success();
}

static bool getTensorLikeElementAndShape(Type ty, Type &elementType,
                                         ArrayRef<int64_t> &shape) {
  if (auto tvTy = dyn_cast<TensorViewType>(ty)) {
    elementType = tvTy.getElementType();
    shape = tvTy.getShape();
    return true;
  }
  if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
    elementType = memrefTy.getElementType();
    shape = memrefTy.getShape();
    return true;
  }
  return false;
}

static FailureOr<InitializeL2G2LPipeOp> verifyTensorEntryInternalPipeHandle(
    Operation *op, Value pipeHandle) {
  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    op->emitOpError()
        << "expects !pto.tensor_view pipe entry to use a pipe produced by "
           "pto.initialize_l2g2l_pipe";
    return failure();
  }
  if (initOp.getLocalAddr()) {
    op->emitOpError()
        << "expects !pto.tensor_view pipe entry to use global-only "
           "pto.initialize_l2g2l_pipe without local_addr";
    return failure();
  }
  return initOp;
}

static LogicalResult verifyTensorEntrySlotType(Operation *op,
                                               TensorViewType entryViewTy,
                                               InitializeL2G2LPipeOp initOp) {
  Type slotElementType;
  ArrayRef<int64_t> slotShape;
  if (!getTensorLikeElementAndShape(initOp.getGmAddr().getType(),
                                    slotElementType, slotShape)) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use "
              "pto.initialize_l2g2l_pipe gm_addr with tensor/memref slot type";
  }
  if (slotElementType != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match initialize_l2g2l_pipe "
              "gm_addr element type";
  }
  if (slotShape.size() != static_cast<size_t>(entryViewTy.getRank())) {
    return op->emitOpError()
           << "expects pipe entry rank to match initialize_l2g2l_pipe gm_addr "
              "rank";
  }

  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic || entryDim == ShapedType::kDynamic ||
        slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match initialize_l2g2l_pipe gm_addr dimension " << slotDim;
  }
  return success();
}

static int8_t getTensorEntrySplit(Operation *op) {
  if (auto alloc = dyn_cast<TAllocOp>(op))
    return alloc.getSplit();
  if (auto push = dyn_cast<TPushOp>(op))
    return push.getSplit();
  if (auto pop = dyn_cast<TPopOp>(op))
    return pop.getSplit();
  if (auto free = dyn_cast<TFreeOp>(op))
    return free.getSplit();
  return 0;
}

static LogicalResult verifyTensorEntryByteSize(Operation *op,
                                               TensorViewType entryViewTy,
                                               InitializeL2G2LPipeOp initOp) {
  auto entryElemCount = getStaticElementCount(entryViewTy.getShape());
  if (!entryElemCount)
    return success();

  uint64_t elemBytes = getElemByteSize(entryViewTy.getElementType());
  if (elemBytes == 0)
    return success();

  uint64_t entryBytes = *entryElemCount * elemBytes;
  int8_t split = getTensorEntrySplit(op);
  uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
  bool isSplitEntry = split != 0;
  bool byteSizeMatches =
      entryBytes == slotBytes || (isSplitEntry && entryBytes * 2 == slotBytes);
  if (!byteSizeMatches) {
    return op->emitOpError()
           << "expects pipe entry byte size to match initialize_l2g2l_pipe "
              "slot_size"
           << (isSplitEntry ? " or half slot_size for split entries" : "")
           << " (got entry byte size = " << entryBytes
           << ", slot_size = " << initOp.getSlotSize() << ")";
  }
  return success();
}

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto initOp = verifyTensorEntryInternalPipeHandle(op, pipeHandle);
  if (failed(initOp) ||
      failed(verifyTensorEntrySlotType(op, entryViewTy, *initOp)) ||
      failed(verifyTensorEntryByteSize(op, entryViewTy, *initOp)))
    return failure();
  return success();
}

static LogicalResult verifyAsyncSessionScratch(Operation *op, Type scratchTy) {
  if (!isa<pto::TileBufType, MemRefType>(scratchTy))
    return op->emitOpError("expects scratch to be tile_buf or memref type");

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC)
    return op->emitOpError("expects scratch to be in vec address space");

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > 2)
    return op->emitOpError("expects scratch to be rank-1 or rank-2");
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError("expects scratch to have a static shape");
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes)
    return op->emitOpError("expects scratch byte size to be statically known");
  if (*scratchBytes < sizeof(uint64_t))
    return op->emitOpError("expects scratch to provide at least 8 bytes");
  return success();
}

static LogicalResult verifyAsyncSessionWorkspace(Operation *op, Type workspaceTy) {
  Type workspaceElemTy;
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    workspaceElemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    workspaceElemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return op->emitOpError("expects workspace to be in GM address space");
  } else {
    return op->emitOpError("expects workspace to be !pto.ptr or memref type");
  }
  if (!isByteIntegerType(workspaceElemTy))
    return op->emitOpError("expects workspace element type to be an 8-bit integer");
  return success();
}

static LogicalResult verifyAsyncSessionAttrs(BuildAsyncSessionOp op) {
  if (auto syncIdAttr = op.getSyncIdAttr()) {
    int64_t syncId = syncIdAttr.getInt();
    if (syncId < 0 || syncId > 7)
      return op.emitOpError("expects sync_id in range [0, 7]");
  }
  if (auto blockBytesAttr = op.getBlockBytesAttr()) {
    if (blockBytesAttr.getInt() <= 0)
      return op.emitOpError("expects block_bytes to be greater than 0");
  }
  if (auto commBlockOffsetAttr = op.getCommBlockOffsetAttr()) {
    if (commBlockOffsetAttr.getInt() < 0)
      return op.emitOpError("expects comm_block_offset to be non-negative");
  }
  if (auto queueNumAttr = op.getQueueNumAttr()) {
    if (queueNumAttr.getInt() <= 0)
      return op.emitOpError("expects queue_num to be greater than 0");
  }
  if (auto channelGroupIdxAttr = op.getChannelGroupIdxAttr()) {
    APInt value = channelGroupIdxAttr.getValue();
    if (value.isNegative())
      return op.emitOpError("expects channel_group_idx to be non-negative");
    if (value.ugt(UINT32_MAX))
      return op.emitOpError("expects channel_group_idx to fit in uint32");
  }
  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  if (failed(verifyAsyncSessionScratch(getOperation(), getScratch().getType())) ||
      failed(verifyAsyncSessionWorkspace(getOperation(), getWorkspace().getType())) ||
      failed(verifyAsyncSessionAttrs(*this))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy)
    return op->emitOpError("expects src and dst to have element types");
  if (dstElemTy != srcElemTy)
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src")))
    return failure();
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError("expects src and dst to have the same static shape");
  return success();
}

static LogicalResult verifyCommTransferWithStaging(Operation *op, Value dst,
                                                   Value src, Value ping,
                                                   Value pong) {
  if (shouldBypassDecodedMemrefVerifier(op))
    return success();
  if (failed(verifyCommGlobalLike(op, dst, "dst")) ||
      failed(verifyCommGlobalLike(op, src, "src")) ||
      failed(verifyCommStagingTileLike(op, ping, "ping")) ||
      failed(verifyCommPingPongSameType(op, ping, pong, "ping", "pong")))
    return failure();
  if (getElemTy(dst.getType()) != getElemTy(src.getType()))
    return op->emitOpError("expects src and dst to have the same element type");
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError("expects src and dst to have the same static shape");
  if (getElemTy(ping.getType()) != getElemTy(src.getType()))
    return op->emitOpError("expects staging tile element type to match src/dst");
  return success();
}

static LogicalResult verifyRootedCommTileTransfer(Operation *op, Value src,
                                                  OperandRange group,
                                                  uint32_t root, Value ping,
                                                  Value pong) {
  if (shouldBypassDecodedMemrefVerifier(op))
    return success();
  if (failed(verifyCommGlobalLike(op, src, "src")) ||
      failed(verifyCommStagingTileLike(op, ping, "ping")) ||
      failed(verifyCommPingPongSameType(op, ping, pong, "ping", "pong")) ||
      failed(verifyCommGlobalGroup(op, group, "group")))
    return failure();
  if (root >= static_cast<uint32_t>(group.size()))
    return op->emitOpError("expects root to index into group operands");
  if (getElemTy(ping.getType()) != getElemTy(src.getType()))
    return op->emitOpError("expects staging tile element type to match src");
  return success();
}

static LogicalResult verifyFrontendPipeTileAccess(Operation *op, Value pipeHandle,
                                                  int64_t split, Type tileTy) {
  if (!isInsideSectionOrAttributedKernel(op))
    return op->emitOpError(
        "must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(op, pipeHandle)))
    return failure();
  if (failed(verifySplitAttr(op, split)))
    return failure();
  return verifyTensorEntryMatchesInternalPipeInit(op, pipeHandle, tileTy);
}

template <typename PongRangeT>
static void addOptionalPongWriteEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    PongRangeT pongRange) {
  if (auto it = pongRange.begin(); it != pongRange.end())
    addEffect(effects, &*it, MemoryEffects::Write::get());
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TPutOp::verify() {
  return verifyCommTransferWithStaging(getOperation(), getDst(), getSrc(),
                                       getPing(), getPong());
}

LogicalResult TGetOp::verify() {
  return verifyCommTransferWithStaging(getOperation(), getDst(), getSrc(),
                                       getPing(), getPong());
}

LogicalResult TNotifyOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto valueTy = dyn_cast<IntegerType>(getValue().getType());
  if (!valueTy || valueTy.getWidth() != 32)
    return emitOpError("expects value to be i32");
  return success();
}

LogicalResult TWaitOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

LogicalResult TTestOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  if (!isa<MemRefType, pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";

  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    if (!memTy.hasRank())
      return op->emitOpError() << "expects " << name << " to be ranked";
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
  }

  auto elemTy = dyn_cast<IntegerType>(getElemTy(ty));
  if (!elemTy || elemTy.getWidth() != 32)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

static LogicalResult verifySyncAllTileWorkspace(Operation *op, Value workspace,
                                                StringRef name,
                                                pto::AddressSpace expectedSpace) {
  Type ty = workspace.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be tile_buf or memref type";

  if (isa<pto::TileBufType>(ty) && failed(verifyTileBufCommon(op, ty, name)))
    return failure();

  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != expectedSpace)
    return op->emitOpError() << "expects " << name << " to be in "
                             << (expectedSpace == pto::AddressSpace::VEC
                                     ? "vec"
                                     : "mat")
                             << " address space";

  Type elemTy = getElemTy(ty);
  auto intTy = dyn_cast_or_null<IntegerType>(elemTy);
  if (!intTy || intTy.getWidth() != 32)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  auto shape = getShapeVec(ty);
  if (shape.empty() || shape.size() > 2)
    return op->emitOpError() << "expects " << name
                             << " to be rank-1 or rank-2";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

static LogicalResult verifySyncAllHardMode(SyncAllOp op, bool hasGm, bool hasUb,
                                           bool hasL1) {
  if (hasGm || hasUb || hasL1 || op.getUsedCores()) {
    return op.emitOpError(
        "expects hard syncall to have no workspace operands or used_cores");
  }
  return success();
}

static LogicalResult verifySyncAllUsedCores(SyncAllOp op) {
  if (auto used = op.getUsedCores()) {
    auto intTy = dyn_cast<IntegerType>(used.getType());
    if (!intTy || intTy.getWidth() != 32)
      return op.emitOpError("expects used_cores to be i32");
  }
  return success();
}

static LogicalResult verifySyncAllSoftWorkspaces(SyncAllOp op, bool hasUb,
                                                 bool hasL1) {
  switch (op.getCoreType().getValue()) {
  case pto::SyncCoreType::AIVOnly:
    if (!hasUb || hasL1) {
      return op.emitOpError(
          "expects soft AIV-only syncall to use gm_workspace + ub_workspace only");
    }
    return verifySyncAllTileWorkspace(op.getOperation(), op.getUbWorkspace(),
                                      "ub_workspace", pto::AddressSpace::VEC);
  case pto::SyncCoreType::AICOnly:
    if (hasUb || !hasL1) {
      return op.emitOpError(
          "expects soft AIC-only syncall to use gm_workspace + l1_workspace only");
    }
    return verifySyncAllTileWorkspace(op.getOperation(), op.getL1Workspace(),
                                      "l1_workspace", pto::AddressSpace::MAT);
  case pto::SyncCoreType::Mix:
    if (!hasUb || !hasL1) {
      return op.emitOpError(
          "expects soft mixed syncall to use gm_workspace + ub_workspace + l1_workspace");
    }
    if (failed(verifySyncAllTileWorkspace(op.getOperation(), op.getUbWorkspace(),
                                          "ub_workspace",
                                          pto::AddressSpace::VEC))) {
      return failure();
    }
    return verifySyncAllTileWorkspace(op.getOperation(), op.getL1Workspace(),
                                      "l1_workspace", pto::AddressSpace::MAT);
  }
  llvm_unreachable("unhandled SyncCoreType");
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  bool hasUb = static_cast<bool>(getUbWorkspace());
  bool hasL1 = static_cast<bool>(getL1Workspace());
  if (getMode().getValue() == pto::SyncAllMode::Hard)
    return verifySyncAllHardMode(*this, hasGm, hasUb, hasL1);

  if (!hasGm)
    return emitOpError("expects soft syncall to provide gm_workspace");
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace")) ||
      failed(verifySyncAllUsedCores(*this)) ||
      failed(verifySyncAllSoftWorkspaces(*this, hasUb, hasL1)))
    return failure();
  return success();
}

LogicalResult TBroadcastOp::verify() {
  if (failed(verifyRootedCommTileTransfer(getOperation(), getSrc(), getGroup(),
                                          getRoot(), getPing(), getPong())))
    return failure();
  if (getSrc().getType() != getGroup().front().getType())
    return emitOpError("expects src type to match group member type");
  return success();
}

LogicalResult CommTGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects staging tile element type to match dst");
  return success();
}

LogicalResult CommTScatterOp::verify() {
  if (failed(verifyRootedCommTileTransfer(getOperation(), getSrc(), getGroup(),
                                          getRoot(), getPing(), getPong())))
    return failure();
  if (getElemTy(getSrc().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects src element type to match group member type");
  return success();
}

LogicalResult TReduceOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getAcc().getType() != getRecvPing().getType())
    return emitOpError("expects acc and recv_ping to have identical types");
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  return success();
}

LogicalResult AicInitializePipeOp::verify() {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube");
}

LogicalResult AivInitializePipeOp::verify() {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector");
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPushToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPopFromAicOp::verify() {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                             /*expectC2V=*/true);
}

LogicalResult TPopFromAivOp::verify() {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

LogicalResult TFreeFromAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult TFreeFromAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt)))
    return failure();

  if (!getLocalAddr()) {
    if (getPeerLocalAddr())
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    if (getLocalSlotNumAttr())
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum())
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
  }

  if (getDirMask() == 3 && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != 3 && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                              getSlotNum(),
                              getFlagBaseAttr()
                                  ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                  : std::nullopt)))
    return failure();

  if (getDirMask() == 3 && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != 3 && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult TPushOp::verify() {
  if (failed(verifyFrontendPipeTileAccess(getOperation(), getPipeHandle(),
                                          getSplit(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError("tile type must map to a supported producer pipe");
  return success();
}

LogicalResult TAllocOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

LogicalResult TPopOp::verify() {
  if (failed(verifyFrontendPipeTileAccess(getOperation(), getPipeHandle(),
                                          getSplit(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError(
        "tile type and target arch must map to a supported consumer pipe");
  return success();
}

LogicalResult TFreeOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

static ParseResult parseTFreeEntryMode(OpAsmParser &parser,
                                       OpAsmParser::UnresolvedOperand &pipe,
                                       Type &firstTy, Type &pipeTy) {
  if (parser.parseOperand(pipe) || parser.parseColonType(firstTy) ||
      parser.parseComma() || parser.parseType(pipeTy) || parser.parseRParen())
    return failure();
  return success();
}

static ParseResult parseTFreePipeOnlyMode(OpAsmParser &parser,
                                          OpAsmParser::UnresolvedOperand first,
                                          OpAsmParser::UnresolvedOperand &pipe,
                                          Type &pipeTy) {
  if (parser.parseColonType(pipeTy) || parser.parseRParen())
    return failure();
  pipe = first;
  return success();
}

static ParseResult parseTFreeSplitAttrs(OpAsmParser &parser,
                                        NamedAttrList &attrs) {
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual())
    return failure();
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs))
    return failure();
  return success();
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;

  if (parser.parseLParen() || parser.parseOperand(first))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    hasEntry = true;
    if (failed(parseTFreeEntryMode(parser, pipe, firstTy, pipeTy)))
      return failure();
  } else if (failed(parseTFreePipeOnlyMode(parser, first, pipe, pipeTy))) {
    return failure();
  }

  NamedAttrList attrs;
  if (parseTFreeSplitAttrs(parser, attrs))
    return failure();

  result.addAttributes(attrs);
  if (hasEntry && parser.resolveOperand(first, firstTy, result.operands))
    return failure();
  if (parser.resolveOperand(pipe, pipeTy, result.operands))
    return failure();
  return success();
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

void BuildAsyncSessionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScratchMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getWorkspaceMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TGetAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
}

void TGetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
}

void TNotifyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getValueMutable(), MemoryEffects::Read::get());
}

void TWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
}

void TTestOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TBroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  addOptionalPongWriteEffect(effects, getPongMutable());
}

void CommTGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
}

void CommTScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  addOptionalPongWriteEffect(effects, getPongMutable());
}

void TReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Write::get());
}

void WaitAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TestAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2G2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getGmAddrMutable(), MemoryEffects::Read::get());
  auto localAddr = getLocalAddrMutable();
  if (!localAddr.empty())
    addEffect(effects, &*localAddr.begin(), MemoryEffects::Read::get());
  auto peerLocalAddr = getPeerLocalAddrMutable();
  if (!peerLocalAddr.empty())
    addEffect(effects, &*peerLocalAddr.begin(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getLocalAddrMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPushOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEntryMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TPopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getTileMutable(), MemoryEffects::Write::get());
}

void TFreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto entry = getEntryMutable();
  if (!entry.empty())
    addEffect(effects, &*entry.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}
