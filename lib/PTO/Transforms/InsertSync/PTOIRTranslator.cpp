// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "pto-ir-translator"

using namespace mlir;
using namespace mlir::pto;

template <typename T>
static T *requireSyncIRNode(InstanceElement *element, llvm::StringRef context) {
  if (auto *typed = dyn_cast_or_null<T>(element))
    return typed;
  llvm::report_fatal_error(context);
}

static void requireMatchingValueCounts(size_t lhsCount, size_t rhsCount,
                                       llvm::StringRef context) {
  if (lhsCount == rhsCount)
    return;
  llvm::report_fatal_error(context);
}

static std::pair<int64_t, int64_t> getStaticOffsetAndSizeFromSubview(
    memref::SubViewOp subView, MemRefType srcType) {
  int64_t baseOffset;
  SmallVector<int64_t, 4> strides;
  if (failed(mlir::getStridesAndOffset(srcType, strides, baseOffset)))
    return {-1, -1};

  int64_t newSize = 1;
  for (int64_t size : subView.getStaticSizes()) {
    if (size == ShapedType::kDynamic)
      return {-1, -1};
    newSize *= size;
  }

  auto staticOffsets = subView.getStaticOffsets();
  if (staticOffsets.empty() || staticOffsets.size() > strides.size())
    return {-1, -1};

  int64_t totalOffset = 0;
  for (size_t i = 0; i < staticOffsets.size(); ++i) {
    int64_t off = staticOffsets[i];
    if (off == ShapedType::kDynamic || strides[i] == ShapedType::kDynamic)
      return {-1, -1};
    totalOffset += off * strides[i];
  }
  return {totalOffset, newSize};
}

static std::pair<int64_t, int64_t>
getStaticOffsetAndSizeFromReinterpretCast(memref::ReinterpretCastOp castOp) {
  auto staticOffsets = castOp.getStaticOffsets();
  if (staticOffsets.empty() || staticOffsets[0] == ShapedType::kDynamic)
    return {0, 0};
  return {staticOffsets[0], 0};
}

// [辅助函数] 尝试从 Operation 中计算相对于 Source 的字节偏移量和新大小
// 返回值: pair<offsetInBytes, sizeInBytes>
// 如果无法计算静态值，返回 {-1, -1} 表示这是动态的
static std::pair<int64_t, int64_t> getStaticOffsetAndSize(Operation *op, Value src) {
  auto srcType = dyn_cast<MemRefType>(src.getType());
  if (!srcType)
    return {0, 0};

  int64_t elemSize = srcType.getElementType().getIntOrFloatBitWidth() / 8;
  if (elemSize == 0)
    elemSize = 1;

  // === Case 1: memref.subview ===
  if (auto subView = dyn_cast<memref::SubViewOp>(op)) {
    auto [offset, size] = getStaticOffsetAndSizeFromSubview(subView, srcType);
    if (offset < 0 || size < 0)
      return {offset, size};
    return {offset * elemSize, size * elemSize};
  }

  // === Case 2: memref.reinterpret_cast ===
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    auto [offset, size] = getStaticOffsetAndSizeFromReinterpretCast(castOp);
    return {offset * elemSize, size};
  }

  return {0, 0};
}
 
// ============================================================================
// 1. 构建入口
// ============================================================================
void PTOIRTranslator::Build() {
  Region &funcRegion = func_.getBody();
  UpdateKernelArgMemInfo();
  RecursionIR(&funcRegion);
}
 
// ============================================================================
// 2. 更新 Kernel 参数内存信息 (GM Global Memory)
// ============================================================================
void PTOIRTranslator::UpdateKernelArgMemInfo() {
  auto funcParamSize = func_.getNumArguments();
  for (size_t i = 0; i < funcParamSize; i++) {
    Value funcArg = func_.getArgument(i);
    Type argType = funcArg.getType();
 
    if (!isa<pto::PtrType>(argType) && !isa<MemRefType>(argType)) {
      continue;
    }
 
    std::unique_ptr<BaseMemInfo> newMemInfo = std::make_unique<BaseMemInfo>(
        funcArg,                  // baseBuffer
        funcArg,                  // rootBuffer
        pto::AddressSpace::GM,    // Scope
        SmallVector<uint64_t>{0}, // Base Addresses
        0                         // Allocate Size
    );
 
    buffer2MemInfoMap_[funcArg].emplace_back(newMemInfo->clone());
  }
}
 
// ============================================================================
// 3. 递归遍历 IR (核心分发逻辑)
// ============================================================================
std::optional<WalkResult> PTOIRTranslator::HandleMemoryInfoOp(Operation *op) {
  if (auto allocOp = dyn_cast<pto::AllocTileOp>(op)) {
    if (failed(UpdateAllocTileOpMemInfo(allocOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }
  if (auto memAllocOp = dyn_cast<memref::AllocOp>(op)) {
    if (failed(UpdateMemrefAllocOpMemInfo(memAllocOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }
  if (auto declareOp = dyn_cast<pto::DeclareTileMemRefOp>(op)) {
    if (failed(UpdateDeclareTileMemRefOpMemInfo(declareOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }
  if (auto castOp = dyn_cast<pto::PointerCastOp>(op)) {
    if (failed(UpdatePointerCastOpMemInfo(castOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  }
  return std::nullopt;
}

std::optional<WalkResult> PTOIRTranslator::HandleAliasOp(Operation *op) {
  if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
    UpdateAliasBufferInfo(makeViewOp.getResult(), makeViewOp.getPtr());
    return WalkResult::advance();
  }
  if (auto bindTileOp = dyn_cast<pto::BindTileOp>(op)) {
    UpdateAliasBufferInfo(bindTileOp.getResult(), bindTileOp.getSource());
    return WalkResult::advance();
  }
  if (auto subViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
    UpdateAliasBufferInfo(subViewOp.getResult(), subViewOp.getSource());
    return WalkResult::advance();
  }
  if (auto memrefSubView = dyn_cast<memref::SubViewOp>(op)) {
    UpdateAliasBufferInfo(memrefSubView.getResult(), memrefSubView.getSource());
    return WalkResult::advance();
  }
  if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    UpdateAliasBufferInfo(castOp.getResult(), castOp.getSource());
    return WalkResult::advance();
  }
  if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op)) {
    UpdateAliasBufferInfo(collapseOp.getResult(), collapseOp.getSrc());
    return WalkResult::advance();
  }
  if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    UpdateAliasBufferInfo(expandOp.getResult(), expandOp.getSrc());
    return WalkResult::advance();
  }
  return std::nullopt;
}

std::optional<WalkResult> PTOIRTranslator::HandleControlFlowOp(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    UpdateForOpInfo(forOp);
    return WalkResult::skip();
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    UpdateWhileOpInfo(whileOp);
    return WalkResult::skip();
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    UpdateIfOpInfo(ifOp);
    return WalkResult::skip();
  }
  return std::nullopt;
}

std::optional<WalkResult> PTOIRTranslator::HandleLeafOp(Operation *op) {
  if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    UpdateYieldOpInfo(yieldOp);
    return WalkResult::advance();
  }
  if (isa<pto::OpPipeInterface>(op)) {
    UpdatePTOOpInfo(op);
    return WalkResult::advance();
  }
  return std::nullopt;
}

void PTOIRTranslator::RecursionIR(Region *region) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto walkResult = HandleMemoryInfoOp(op))
      return *walkResult;
    if (auto walkResult = HandleAliasOp(op))
      return *walkResult;
    if (auto walkResult = HandleControlFlowOp(op))
      return *walkResult;
    if (auto walkResult = HandleLeafOp(op))
      return *walkResult;
    return WalkResult::advance();
  });
 
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PTO InjectSync Traverse IR Failed!");
  }
}
 
// ============================================================================
// 4. 处理 AllocTile / PointerCast
// ============================================================================
static uint64_t getStaticTileAllocSizeBytes(pto::TileBufType tileType) {
  if (!tileType)
    return 0;

  uint64_t numElements = 1;
  for (int64_t dim : tileType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return 0;
    numElements *= static_cast<uint64_t>(dim);
  }
  int64_t elemSize = tileType.getElementType().getIntOrFloatBitWidth() / 8;
  return numElements * static_cast<uint64_t>(std::max<int64_t>(elemSize, 0));
}

static pto::AddressSpace resolveAllocTileAddressSpace(pto::TileBufType tileType) {
  if (!tileType)
    return pto::AddressSpace::MAT;
  if (auto attr = tileType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr))
      return ptoAttr.getAddressSpace();
  }
  return pto::AddressSpace::MAT;
}

LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
  Value res = op.getResult();

  auto tileType = dyn_cast<pto::TileBufType>(res.getType());
  uint64_t baseAddr = 0;

  // If alloc_tile carries an explicit address, record it when it's a constant.
  if (Value addr = op.getAddr()) {
    llvm::APInt apIntValue;
    if (matchPattern(addr, m_ConstantInt(&apIntValue))) {
      baseAddr = static_cast<uint64_t>(apIntValue.getSExtValue());
    }
  }

  uint64_t sizeInBytes = getStaticTileAllocSizeBytes(tileType);
  pto::AddressSpace space = resolveAllocTileAddressSpace(tileType);
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      space,
      SmallVector<uint64_t>{baseAddr},
      sizeInBytes);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
LogicalResult PTOIRTranslator::UpdatePointerCastOpMemInfo(pto::PointerCastOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();
 
  if (op.getAddrs().empty()) {
    return op.emitError("PointerCast must have at least one address operand");
  }
  Value rootSrc = op.getAddrs().front(); 
 
  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }
 
  pto::AddressSpace space = pto::AddressSpace::GM; 
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }
 
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,          
      rootSrc,      
      space,
      SmallVector<uint64_t>{0}, 
      sizeInBytes
  );
 
  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}

LogicalResult
PTOIRTranslator::UpdateDeclareTileMemRefOpMemInfo(pto::DeclareTileMemRefOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType)
    return failure();

  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elemSize == 0)
      elemSize = 1;

    int64_t numElements = 1;
    for (auto dim : memRefType.getShape())
      numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }

  pto::AddressSpace space = pto::AddressSpace::MAT;
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr))
      space = ptoAttr.getAddressSpace();
  }

  // declare_tile_memref is only a symbolic placeholder. Use its SSA result as
  // both base/root so later bind_tile aliases and tpop consumers can be
  // connected by InsertSync without inventing a fake allocation.
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,
      res,
      space,
      SmallVector<uint64_t>{0},
      sizeInBytes);

  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
// ============================================================================
// 5. [P0 修改] 更新 PTO Op 信息 (通用接口版)
// ============================================================================
void PTOIRTranslator::CollectMemEffectDependences(
    Operation *op, SmallVector<const BaseMemInfo *> &defVec,
    SmallVector<const BaseMemInfo *> &useVec) {
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: Op " << op->getName()
                            << " has Pipe but no MemoryEffects interface.\n");
    return;
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memEffect.getEffects(effects);
  for (auto &effect : effects) {
    Value val = effect.getValue();
    if (!val)
      continue;
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      UpdateDefUseVec({val}, useVec);
    } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
      UpdateDefUseVec({val}, defVec);
    }
  }
}

void PTOIRTranslator::SetCompoundCoreType(
    CompoundInstanceElement &compoundElement, PipelineType pipe) {
  compoundElement.compoundCoreType =
      (pipe == pto::PipelineType::PIPE_M ||
       pipe == pto::PipelineType::PIPE_MTE1)
          ? pto::TCoreType::CUBE
          : pto::TCoreType::VECTOR;
}

void PTOIRTranslator::UpdatePTOOpInfo(Operation *op) {
  pto::PipelineType pipe = getOpPipeline(op);
  if (pipe == pto::PipelineType::PIPE_UNASSIGNED)
    return;

  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
  CollectMemEffectDependences(op, defVec, useVec);

  auto compoundElement = std::make_unique<CompoundInstanceElement>(
      index, defVec, useVec, pipe, op->getName());
  compoundElement->elementOp = op;
  SetCompoundCoreType(*compoundElement, pipe);
  syncIR_.emplace_back(std::move(compoundElement));
  index++;
}
 
// ============================================================================
// 6. [P0 修改] 获取 Op 的 Pipeline 类型
// ============================================================================
pto::PipelineType PTOIRTranslator::getOpPipeline(Operation *op) {
  // 1. 优先尝试通过接口获取
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op)) {
    // 注意：假设 pto::Pipe (ODS Enum) 和 pto::PipelineType (C++ Enum) 的数值定义是一致的
    // 或者在这里做一个 switch-case 映射
    // 目前假设直接 cast 是安全的 (0=S, 1=V, 2=M ...)
    return static_cast<pto::PipelineType>(pipeOp.getPipe());
  }
 
  // 2. 如果没实现接口，返回 Unassigned
  return pto::PipelineType::PIPE_UNASSIGNED;
}
 
// ============================================================================
// 7. 控制流处理 (SCF Support)
// ============================================================================
 
void PTOIRTranslator::UpdateForOpInfo(scf::ForOp forOp) {
  auto forBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  forBeginElement->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forBeginElement));
  
  std::unique_ptr<InstanceElement> &forElement = syncIR_[index];
  index++;
  
  auto *forBeginPtr =
      requireSyncIRNode<LoopInstanceElement>(forElement.get(),
                                             "Sync IR construction failed");
  
  if (!forOp.getInitArgs().empty()) {
    const size_t initArgCount = forOp.getInitArgs().size();
    const size_t regionIterArgCount = forOp.getRegionIterArgs().size();
    requireMatchingValueCounts(
        initArgCount, regionIterArgCount,
        "scf.for init arg count does not match region iter arg count");
    for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
      UpdateAliasBufferInfo(forOp.getRegionIterArgs()[i], arg);
    }
  }
 
  RecursionIR(&forOp.getRegion());
 
  forBeginPtr->endId = index;
  auto forEnd = forBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = forOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}
 
void PTOIRTranslator::UpdateWhileOpInfo(scf::WhileOp whileOp) {
  auto loopBeginElement = std::make_unique<LoopInstanceElement>(index, index, index);
  loopBeginElement->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(loopBeginElement));
  
  auto *loopBeginPtr = dyn_cast<LoopInstanceElement>(syncIR_.back().get());
  index++;
 
  if (!whileOp.getInits().empty()) {
    for (auto [initArg, blockArg] : llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
      UpdateAliasBufferInfo(blockArg, initArg);
    }
    auto conditionOp = whileOp.getConditionOp();
    for (auto [yieldedArg, blockArg] : llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
      UpdateAliasBufferInfo(blockArg, yieldedArg);
    }
  }
 
  RecursionIR(&whileOp.getBefore());
  RecursionIR(&whileOp.getAfter());
 
  loopBeginPtr->endId = index;
  auto forEnd = loopBeginPtr->CloneFor(KindOfLoop::LOOP_END);
  forEnd->elementOp = whileOp.getOperation();
  syncIR_.emplace_back(std::move(forEnd));
  index++;
}
 
void PTOIRTranslator::AppendIfBranchPlaceholder(BranchInstanceElement *branchPtr,
                                                Operation *op,
                                                bool isVirtualElse,
                                                Operation *parentIfOp) {
  auto placeHolder = std::make_unique<PlaceHolderInstanceElement>(
      index, branchPtr->GetIndex());
  placeHolder->elementOp = op;
  placeHolder->isVirtualElse = isVirtualElse;
  if (isVirtualElse)
    placeHolder->parentIfOp = parentIfOp;
  syncIR_.emplace_back(std::move(placeHolder));
  index++;
}

void PTOIRTranslator::AppendIfEnd(BranchInstanceElement *ifPtr, scf::IfOp ifOp) {
  auto ifEndElement = ifPtr->CloneBranch(KindOfBranch::IF_END);
  ifEndElement->elementOp = ifOp.getOperation();
  syncIR_.emplace_back(std::move(ifEndElement));
  index++;
}

void PTOIRTranslator::UpdateIfOpInfo(scf::IfOp ifOp) {
  auto ifBeginElement =
      std::make_unique<BranchInstanceElement>(index, index,
                                              KindOfBranch::IF_BEGIN);
  ifBeginElement->elementOp = ifOp.getOperation();
  auto *ifPtr = ifBeginElement.get();
  syncIR_.emplace_back(std::move(ifBeginElement));
  index++;

  RecursionIR(&ifOp.getThenRegion());
  AppendIfBranchPlaceholder(ifPtr, ifOp.getThenRegion().front().getTerminator(),
                            /*isVirtualElse=*/false,
                            /*parentIfOp=*/nullptr);
  ifPtr->branchId = index;

  auto ifElseElement = ifPtr->CloneBranch(KindOfBranch::ELSE_BEGIN);
  ifElseElement->elementOp = ifOp.getOperation();
  auto *elsePtr = ifElseElement.get();
  syncIR_.emplace_back(std::move(ifElseElement));
  index++;

  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion());
    AppendIfBranchPlaceholder(elsePtr, ifOp.getElseRegion().front().getTerminator(),
                              /*isVirtualElse=*/false,
                              /*parentIfOp=*/nullptr);
  } else {
    AppendIfBranchPlaceholder(elsePtr, ifOp.getOperation(),
                              /*isVirtualElse=*/true, ifOp.getOperation());
  }

  elsePtr->endId = index;
  ifPtr->endId = index;
  AppendIfEnd(ifPtr, ifOp);
}
 
void PTOIRTranslator::UpdateYieldOpInfo(scf::YieldOp yieldOp) {
  auto *parentOp = yieldOp->getParentOp();
  if (!parentOp || isa<scf::WhileOp>(parentOp)) return;
 
  const size_t parentResultCount = parentOp->getResults().size();
  const size_t yieldOperandCount = yieldOp->getOpOperands().size();
  requireMatchingValueCounts(parentResultCount, yieldOperandCount,
                             "yield/result count mismatch while updating aliases");
  for (auto [yieldVal, resultVal] : llvm::zip(yieldOp->getOpOperands(), parentOp->getResults())) {
    UpdateAliasBufferInfo(resultVal, yieldVal.get());
  }
}
 
// ============================================================================
// 8. 辅助函数
// ============================================================================
void PTOIRTranslator::UpdateAliasBufferInfo(Value result, Value source) {
  if (!result || !source) return;
  if (!buffer2MemInfoMap_.contains(source)) return;
 
  int64_t deltaOffset = 0;
  int64_t newSize = -1; 
 
  if (auto op = result.getDefiningOp()) {
    auto info = getStaticOffsetAndSize(op, source);
    if (info.first != -1) {
        deltaOffset = info.first;
        if (info.second > 0) newSize = info.second;
    } 
  }
 
  auto &resultMemInfoVec = buffer2MemInfoMap_[result];
  
  for (auto &parentInfo : buffer2MemInfoMap_[source]) {
    auto newInfo = parentInfo->clone(result);
 
    if (!newInfo->baseAddresses.empty()) {
        newInfo->baseAddresses[0] += deltaOffset;
    } else {
        newInfo->baseAddresses.push_back(deltaOffset);
    }
 
    if (newSize > 0) {
        newInfo->allocateSize = newSize;
    }
 
    resultMemInfoVec.emplace_back(std::move(newInfo));
  }
}
 
// ============================================================================
// 实现 UpdateMemrefAllocOpMemInfo
// ============================================================================
LogicalResult PTOIRTranslator::UpdateMemrefAllocOpMemInfo(memref::AllocOp op) {
  Value res = op.getResult();
  auto memRefType = dyn_cast<MemRefType>(res.getType());
  if (!memRefType) return failure();
 
  // 1. 计算大小 (Bytes)
  uint64_t sizeInBytes = 0;
  if (memRefType.hasStaticShape()) {
    int64_t elemSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;
    if (elemSize == 0) elemSize = 1; // bool case
    
    int64_t numElements = 1;
    for (auto dim : memRefType.getShape()) numElements *= dim;
    sizeInBytes = numElements * elemSize;
  }
 
  // 2. 解析地址空间 (Scope)
  // 默认视为 MAT/UB (Local Memory)，这是 alloc 的常见用途
  // 如果有显式属性，则覆盖
  pto::AddressSpace space = pto::AddressSpace::MAT; 
  
  if (auto attr = memRefType.getMemorySpace()) {
    if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr)) {
      space = ptoAttr.getAddressSpace();
    }
  }
 
  // 3. 注册 Buffer 信息
  // 对于 alloc，它自己就是 Root
  auto newMemInfo = std::make_unique<BaseMemInfo>(
      res,                      // baseBuffer
      res,                      // rootBuffer (Self is root)
      space,
      SmallVector<uint64_t>{0}, // Base Addresses (Offset 0)
      sizeInBytes
  );
 
  buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
  return success();
}
 
void PTOIRTranslator::UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec) {
  for (Value v : values) {
    if (buffer2MemInfoMap_.contains(v)) {
      for (auto &memInfo : buffer2MemInfoMap_[v]) {
        vec.push_back(memInfo.get());
      }
    }
  }
}
 
// ============================================================================
// 9. 调试与打印支持
// ============================================================================
 
std::string PTOIRTranslator::getPipelineName(pto::PipelineType pipe) {
  switch (pipe) {
  case pto::PipelineType::PIPE_MTE1: return "MTE1";
  case pto::PipelineType::PIPE_MTE2: return "MTE2";
  case pto::PipelineType::PIPE_MTE3: return "MTE3";
  case pto::PipelineType::PIPE_M:    return "CUBE";
  case pto::PipelineType::PIPE_V:    return "VECTOR";
  case pto::PipelineType::PIPE_S:    return "SCALAR";
  case pto::PipelineType::PIPE_ALL:  return "BARRIER";
  default: return "UNKNOWN";
  }
}
 
void PTOIRTranslator::printMemInfoList(llvm::raw_ostream &os, 
                                       const SmallVector<const BaseMemInfo *> &list, 
                                       AsmState &state) {
  os << "[";
  bool first = true;
  for (const auto *info : list) {
    if (!first) os << ", ";
    info->rootBuffer.printAsOperand(os, state);
    // [Fix] 打印 MAT 或 VEC 或 GM
    if (info->scope == pto::AddressSpace::GM) os << "(GM)";
    else if (info->scope == pto::AddressSpace::MAT) os << "(MAT)";
    else if (info->scope == pto::AddressSpace::VEC) os << "(VEC)";
    else os << "(Other)"; // 处理 LEFT/RIGHT/ACC 等其他情况
    first = false;
  }
  os << "]";
}
 
void PTOIRTranslator::print() {
  llvm::errs() << "\n=== PTO IR Translator Dump ===\n";
  
  AsmState state(func_); 
 
  llvm::errs() << "--- Buffer Analysis (Value -> Root) ---\n";
  for (auto &it : buffer2MemInfoMap_) {
    Value v = it.first;
    auto &infoList = it.second;
    
    llvm::errs() << "  ";
    v.printAsOperand(llvm::errs(), state);
    llvm::errs() << " -> ";
    
    for (auto &mem : infoList) {
        mem->rootBuffer.printAsOperand(llvm::errs(), state);
        llvm::errs() << " ";
    }
    llvm::errs() << "\n";
  }
 
  llvm::errs() << "\n--- SyncIR Structure ---\n";
  for (const auto &element : syncIR_) {
    unsigned id = element->GetIndex();
    llvm::errs() << llvm::formatv("{0,4}: ", id); 
 
    switch (element->GetKind()) {
    case InstanceElement::KindTy::COMPOUND: {
      auto *comp = dyn_cast<CompoundInstanceElement>(element.get());
      llvm::errs() << "COMPOUND [" << getPipelineName(comp->kPipeValue) << "] ";
      llvm::errs() << comp->opName.getStringRef() << "\n";
      
      llvm::errs() << "      DEF: ";
      printMemInfoList(llvm::errs(), comp->defVec, state);
      llvm::errs() << "\n      USE: ";
      printMemInfoList(llvm::errs(), comp->useVec, state);
      llvm::errs() << "\n";
      break;
    }
    case InstanceElement::KindTy::LOOP: 
        llvm::errs() << "LOOP\n"; break;
    case InstanceElement::KindTy::BRANCH: 
        llvm::errs() << "BRANCH\n"; break;
    case InstanceElement::KindTy::PLACE_HOLDER: 
        llvm::errs() << "PLACE_HOLDER\n"; break;
    }
  }
  llvm::errs() << "==============================\n\n";
}
