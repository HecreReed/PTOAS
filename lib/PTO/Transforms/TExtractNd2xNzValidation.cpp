// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Driver-level validation helpers for the ND-to-2xNZ dual-output TEXTRACT form
// (design doc sections 5.1 items 11-13, 5.3.1, 7). These are plain driver
// functions, not MLIR passes; see the header for placement contracts.

#include "PTO/Transforms/TExtractNd2xNzValidation.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

// A half-open physical byte range in a specific local memory space. Ranges in
// different spaces never alias even when their numeric addresses coincide
// (design doc 5.4: PhysicalRange = (addressSpace, [baseByte, endByte))).
struct ByteRange {
  std::optional<pto::AddressSpace> space;
  int64_t base = 0;
  int64_t end = 0;
  bool resolved = false;
};

// Skip single-input view/cast chains that preserve the underlying buffer:
// subview, bitcast, treshape, and single-input unrealized_conversion_cast.
Value skipViews(Value v) {
  while (auto *def = v.getDefiningOp()) {
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() == 1) {
        v = cast.getInputs().front();
        continue;
      }
      break;
    }
    if (auto b = dyn_cast<pto::BitcastOp>(def)) {
      v = b.getSrc();
      continue;
    }
    if (auto r = dyn_cast<pto::TReshapeOp>(def)) {
      v = r.getSrc();
      continue;
    }
    if (auto s = dyn_cast<pto::SubViewOp>(def)) {
      v = s.getSource();
      continue;
    }
    break;
  }
  return v;
}

bool isRowPlusOneTile(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb)
    return false;
  return tb.getCompactModeI32() ==
         static_cast<int32_t>(pto::CompactMode::RowPlusOne);
}

std::optional<pto::AddressSpace> tileAddressSpace(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb)
    return std::nullopt;
  auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace());
  if (!attr)
    return std::nullopt;
  return attr.getAddressSpace();
}

// Resolve the static absolute physical byte range of a tile value after
// PTOResolveBufferSelect materialized planner addresses. Only allocation-rooted
// values with a foldable explicit address resolve; anything else leaves
// out.resolved == false (callers must then fail conservatively).
ByteRange resolveAllocationByteRange(Value v) {
  ByteRange out;
  v = skipViews(v);
  auto *def = v.getDefiningOp();
  if (!def)
    return out;
  auto alloc = dyn_cast<pto::AllocTileOp>(def);
  if (!alloc)
    return out;

  auto addr = alloc.getAddr();
  if (!addr)
    return out;
  auto baseOpt = getPTOConstantIntLike(addr);
  if (!baseOpt || *baseOpt < 0)
    return out;

  auto ty = dyn_cast<pto::TileBufType>(alloc.getResult().getType());
  if (!ty)
    return out;
  auto space = tileAddressSpace(ty);
  if (!space)
    return out;
  // Shared access-envelope sizing (design doc 5.4): alias checks use
  // getTileBufAccessEndByteSize (the last accessible byte) rather than the
  // allocation reservation, so RowPlusOne trailing gaps are not treated as
  // accessed. The base address folds constant casts / addi chains via the
  // shared resolver, matching the verifier's index folding.
  auto endBytesOpt = getTileBufAccessEndByteSize(ty);
  if (!endBytesOpt || *endBytesOpt <= 0)
    return out;
  int64_t end = 0;
  if (__builtin_add_overflow(*baseOpt, *endBytesOpt, &end))
    return out; // wrapped range would corrupt the alias analysis
  out.space = space;
  out.base = *baseOpt;
  out.end = end;
  out.resolved = true;
  return out;
}

bool rangesInteract(const ByteRange &a, const ByteRange &b) {
  if (!a.resolved || !b.resolved)
    return false;
  if (!a.space || !b.space || *a.space != *b.space)
    return false;
  // half-open ranges [base, end)
  return a.base < b.end && b.base < a.end;
}

bool isPartialValidTile(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb)
    return false;
  auto shape = tb.getShape();
  auto valid = tb.getValidShape();
  return shape.size() == 2 && valid.size() == 2 &&
         (valid[0] != shape[0] || valid[1] != shape[1]);
}

// Direct internal call component of a function: all functions reachable from
// it through direct func.call edges (both directions) to internal
// (body-bearing) definitions. Following only caller->callee edges from a
// partial producer misses aliasing TSTORE ops in callers that call into the
// producer's function (post-planning design doc 5.3.1 item 3/6), so we build
// the undirected reachability closure: an edge a->b puts both a and b in the
// same component regardless of entry side.
void collectCallComponent(func::FuncOp entry, ModuleOp module,
                          llvm::SmallVectorImpl<func::FuncOp> &out) {
  // Gather every internal function and every direct internal call edge.
  llvm::SmallPtrSet<func::FuncOp, 16> funcs;
  llvm::DenseMap<func::FuncOp, SmallVector<func::FuncOp, 4>> callees;
  llvm::DenseMap<func::FuncOp, SmallVector<func::FuncOp, 4>> callers;
  module.walk([&](func::FuncOp f) { funcs.insert(f); });
  module.walk([&](func::CallOp call) {
    func::FuncOp parentFunc = call->getParentOfType<func::FuncOp>();
    if (!parentFunc)
      return;
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!callee || callee.isDeclaration() || !funcs.count(callee))
      return;
    callees[parentFunc].push_back(callee);
    callers[callee].push_back(parentFunc);
  });

  llvm::SmallPtrSet<func::FuncOp, 8> visited;
  llvm::SmallVector<func::FuncOp, 16> worklist{entry};
  visited.insert(entry);
  while (!worklist.empty()) {
    func::FuncOp cur = worklist.pop_back_val();
    out.push_back(cur);
    auto pushEdge = [&](func::FuncOp other) {
      if (other && visited.insert(other).second)
        worklist.push_back(other);
    };
    for (func::FuncOp c : callees[cur])
      pushEdge(c);
    for (func::FuncOp c : callers[cur])
      pushEdge(c);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// validateTExtractNd2xNzInputProvenance
//===----------------------------------------------------------------------===//
// Design doc 5.1 item 11: reject runtime-bound tile provenance for the
// ND-to-2xNz form before planning. Allowed roots: AllocTileOp, or a
// MultiTileGetOp of an AllocMultiTileOp for non-RowPlusOne operands.
// DeclareTileOp / TAssignOp / TPopOp / TPopFromAicOp / TPopFromAivOp and their
// views, block arguments, and unknown roots are rejected.
mlir::LogicalResult
mlir::pto::validateTExtractNd2xNzInputProvenance(mlir::Operation *module) {
  auto mod = dyn_cast<ModuleOp>(module);
  if (!mod)
    return success();

  bool failedModule = false;
  mod.walk([&](pto::TExtractOp op) {
    if (!op.isNdTo2xNzForm())
      return;
    auto checkOperand = [&](Value v, StringRef name) {
      if (failedModule)
        return;
      Value root = skipViews(v);
      Operation *rootOp = root.getDefiningOp();
      bool rowPlusOne = isRowPlusOneTile(root.getType());
      if (!rootOp) {
        op.emitOpError() << "ND-to-2xNz TEXTRACT " << name
                         << " is a block argument or function operand; "
                            "runtime-bound tile provenance is not supported "
                            "(use alloc_tile with planner-owned or statically "
                            "known level3 address)";
        failedModule = true;
        return;
      }
      if (isa<pto::AllocTileOp>(rootOp))
        return;
      if (auto get = dyn_cast<pto::MultiTileGetOp>(rootOp)) {
        if (!rowPlusOne)
          return;
        rootOp->emitOpError()
            << "supplies RowPlusOne ND-to-2xNz " << name
            << " from multi-buffer provenance; a single pto.alloc_tile is "
               "required";
        failedModule = true;
        return;
      }
      if (isa<pto::DeclareTileOp, pto::TAssignOp, pto::TPopOp,
              pto::TPopFromAicOp, pto::TPopFromAivOp>(rootOp)) {
        rootOp->emitOpError()
            << "is runtime-bound tile provenance for ND-to-2xNz " << name
            << " (root op " << rootOp->getName()
            << "); only alloc_tile (or non-RowPlusOne alloc_multi_tile slots) "
               "are supported";
        failedModule = true;
        return;
      }
      rootOp->emitOpError() << "is unsupported tile provenance for "
                            << "ND-to-2xNz " << name << " (root op "
                            << rootOp->getName()
                            << "); use alloc_tile with planner-owned or "
                               "statically known level3 address";
      failedModule = true;
    };
    checkOperand(op.getSrc(), "src");
    checkOperand(op.getDsts()[0], "dst0");
    checkOperand(op.getDsts()[1], "dst1");
  });
  return failedModule ? failure() : success();
}

//===----------------------------------------------------------------------===//
// validateTExtractNd2xNzPostPlanningSafety
//===----------------------------------------------------------------------===//
// Design doc 5.3.1: after PTOResolveBufferSelect and before
// PTOInlineBackendHelpersPass, reject any generic TSTORE whose source physical
// range (same address space) aliases a partial-valid ND-to-2xNz destination
// inside the same direct-call component. When any partial producer exists, the
// call surface must also be closed: call_indirect / external / unresolved
// direct callees / other opaque CallOpInterface ops reject the whole unit.
mlir::LogicalResult
mlir::pto::validateTExtractNd2xNzPostPlanningSafety(mlir::Operation *module) {
  auto mod = dyn_cast<ModuleOp>(module);
  if (!mod)
    return success();

  bool failedModule = false;

  // Pairwise no-alias recheck (design doc 5.3.1 item 2): src/dst0/dst1 must
  // resolve to non-negative static absolute ranges and stay pairwise disjoint
  // in the same address space. Runs unconditionally for every dual-output op,
  // including explicit-address level3 shapes where PlanMemory is skipped.
  mod.walk([&](pto::TExtractOp op) {
    if (failedModule || !op.isNdTo2xNzForm())
      return;
    SmallVector<ByteRange, 3> ranges;
    SmallVector<StringRef, 3> names{"src", "dst0", "dst1"};
    SmallVector<Value, 3> operands{op.getSrc(), op.getDsts()[0],
                                   op.getDsts()[1]};
    for (unsigned i = 0; i < 3; ++i) {
      ByteRange range = resolveAllocationByteRange(operands[i]);
      if (!range.resolved) {
        op.emitOpError() << "cannot resolve static absolute physical range of "
                         << names[i]
                         << " for pairwise no-alias recheck";
        failedModule = true;
        return;
      }
      ranges.push_back(range);
    }
    for (unsigned i = 0; i < 3 && !failedModule; ++i) {
      for (unsigned j = i + 1; j < 3; ++j) {
        if (rangesInteract(ranges[i], ranges[j])) {
          op.emitOpError()
              << "ND-to-2xNz " << names[i] << "/" << names[j]
              << " physical ranges alias in the same address space "
                 "(no-alias contract)";
          failedModule = true;
        }
      }
    }
  });
  if (failedModule)
    return failure();

  struct PartialDest {
    func::FuncOp func;
    ByteRange range;
  };
  llvm::SmallVector<PartialDest, 8> partials;
  mod.walk([&](pto::TExtractOp op) {
    if (!op.isNdTo2xNzForm())
      return;
    auto func = op->getParentOfType<func::FuncOp>();
    for (Value dst : op.getDsts()) {
      if (!isPartialValidTile(dst.getType()))
        continue;
      ByteRange range = resolveAllocationByteRange(dst);
      if (!range.resolved) {
        op.emitOpError() << "cannot resolve static absolute physical range of "
                            "partial-valid ND-to-2xNZ destination";
        failedModule = true;
        continue;
      }
      partials.push_back({func, range});
    }
  });
  if (failedModule || partials.empty())
    return failedModule ? failure() : success();

  // Call-surface closure: with any partial producer, every direct func.call
  // must resolve to an internal definition and no opaque call-like op may be
  // present anywhere in the compile unit.
  mod.walk([&](func::CallOp call) {
    if (failedModule)
      return;
    auto callee = mod.lookupSymbol<func::FuncOp>(call.getCallee());
    if (callee && !callee.isDeclaration())
      return;
    call->emitOpError()
        << "call surface not closed: indirect/external/unresolved direct "
           "callee in a compile unit with a partial-valid ND-to-2xNZ producer";
    failedModule = true;
  });
  mod.walk([&](Operation *op) {
    if (failedModule)
      return;
    if (isa<func::CallOp>(op))
      return;
    if (isa<func::CallIndirectOp>(op)) {
      op->emitOpError()
          << "call surface not closed: func.call_indirect in a compile unit "
             "with a partial-valid ND-to-2xNZ producer";
      failedModule = true;
      return;
    }
    if (isa<CallOpInterface>(op)) {
      op->emitOpError()
          << "call surface not closed: opaque call-like op " << op->getName()
          << " in a compile unit with a partial-valid ND-to-2xNZ producer";
      failedModule = true;
    }
  });
  if (failedModule)
    return failure();

  // Component-wide alias check over direct-call components containing any
  // partial producer.
  llvm::SmallPtrSet<func::FuncOp, 8> checkedFuncs;
  for (auto &pd : partials) {
    if (!pd.func || failedModule)
      continue;
    llvm::SmallVector<func::FuncOp, 8> component;
    collectCallComponent(pd.func, mod, component);
    for (func::FuncOp f : component) {
      f.walk([&](pto::TStoreOp store) {
        if (failedModule)
          return;
        ByteRange srcRange = resolveAllocationByteRange(store.getSrc());
        if (!srcRange.resolved) {
          store->emitOpError()
              << "cannot resolve static absolute physical range of TSTORE "
                 "source while a partial-valid ND-to-2xNZ destination is "
                 "present in the same call component";
          failedModule = true;
          return;
        }
        if (rangesInteract(srcRange, pd.range)) {
          store->emitOpError()
              << "pto.tstore source physical range aliases a partial-valid "
                 "ND-to-2xNZ destination in the same address space and call "
                 "component; undefined NZ padding cannot be stored";
          failedModule = true;
        }
      });
    }
    checkedFuncs.insert(component.begin(), component.end());
  }
  (void)checkedFuncs;
  return failedModule ? failure() : success();
}

//===----------------------------------------------------------------------===//
// validateTExtractNd2xNzPrePartition
//===----------------------------------------------------------------------===//
// Design doc 5.3.2: the mixed-backend driver splits the outer module into
// child compile units and clones cross-child callees into declarations. This
// plain precheck runs before collectChildJobs()/child cloning:
//   - fixed-depth structure guard: while any ND-to-2xNz form exists, no
//     nested ModuleOp/function scope is allowed (design doc 5.3.2 item 1);
//   - while any partial-valid producer exists, every direct func.call must
//     stay within one immediate child and peer imports
//     (pto.import_reserved_buffer) are rejected outright (items 4/5).
mlir::LogicalResult
mlir::pto::validateTExtractNd2xNzPrePartition(mlir::Operation *module) {
  auto mod = dyn_cast<ModuleOp>(module);
  if (!mod) {
    return success();
  }

  bool hasNd2xNz = false;
  bool hasPartialProducer = false;
  mod.walk([&](pto::TExtractOp op) {
    if (!op.isNdTo2xNzForm()) {
      return;
    }
    hasNd2xNz = true;
    for (Value dst : op.getDsts()) {
      if (isPartialValidTile(dst.getType())) {
        hasPartialProducer = true;
      }
    }
  });
  if (!hasNd2xNz) {
    return success();
  }

  // Fixed-depth ownership guard (design doc 5.3.2): the mixed-backend driver
  // splits the outer module into exactly one level of backend child ModuleOps
  // (root Module -> child Module -> func.func). Every descendant ModuleOp
  // must be an immediate child of the root (its parent is the root, not a
  // sibling child or a function), and each child body must contain only
  // func.func ops directly (no further module scopes, no stray ops).
  bool hasNestedModule = false;
  mod.walk([&](ModuleOp m) {
    if (m.getOperation() == mod.getOperation())
      return;
    if (m.getOperation()->getParentOp() != mod.getOperation()) {
      hasNestedModule = true;
      return;
    }
    for (Operation &childOp : *m.getBody()) {
      if (!isa<func::FuncOp>(childOp)) {
        hasNestedModule = true;
        return;
      }
    }
  });
  if (hasNestedModule) {
    mod->emitOpError()
        << "backend-partitioned ND-to-2xNz validation requires every "
           "descendant module to be an immediate child of the root whose body "
           "contains only functions directly";
    return failure();
  }

  // Root body structure: the design allows either top-level funcs
  // (non-partitioned) or exactly immediate backend child modules
  // (partitioned); once partitioned, the root body must contain ONLY child
  // modules (any other top-level op is rejected).
  bool rootHasModuleChild = false;
  bool rootHasNonModuleTopOp = false;
  for (Operation &rootOp : *mod.getBody()) {
    if (isa<ModuleOp>(rootOp)) {
      rootHasModuleChild = true;
    } else {
      rootHasNonModuleTopOp = true;
    }
  }
  if (rootHasModuleChild && rootHasNonModuleTopOp) {
    mod->emitOpError()
        << "backend-partitioned ND-to-2xNz validation requires the root body "
           "to contain only immediate backend child modules when partitioned";
    return failure();
  }

  if (!hasPartialProducer) {
    return success();
  }

  // With any partial producer, the whole outer module must be partition-safe:
  // reject cross-child direct calls (even full-valid and disconnected) and any
  // peer import before child cloning (design doc 5.3.2 items 4/5).
  bool partitionUnsafe = false;
  mod.walk([&](func::CallOp call) {
    if (partitionUnsafe) {
      return;
    }
    auto caller = call->getParentOfType<func::FuncOp>();
    if (!caller) {
      return;
    }
    auto *callerParent = caller->getParentOp();
    // Resolve the callee against the symbol table the caller lives in: with
    // backend children, a direct call inside child A must resolve among A's
    // direct funcs (root lookupSymbol does not see into child modules).
    auto callerMod = caller->getParentOfType<ModuleOp>();
    auto callee =
        callerMod ? callerMod.lookupSymbol<func::FuncOp>(call.getCallee())
                  : mod.lookupSymbol<func::FuncOp>(call.getCallee());
    if (!callee) {
      call->emitOpError()
          << "call surface not closed: unresolved direct callee in a "
             "backend-partitioned module with a partial-valid ND-to-2xNZ "
             "producer";
      partitionUnsafe = true;
      return;
    }
    if (callee->getParentOp() != callerParent) {
      call->emitOpError()
          << "backend-partitioned module with partial-valid ND-to-2xNZ does "
             "not permit cross-child direct calls";
      partitionUnsafe = true;
    }
  });
  if (partitionUnsafe) {
    return failure();
  }

  mod.walk([&](pto::ImportReservedBufferOp importOp) {
    if (!partitionUnsafe) {
      importOp->emitOpError()
          << "backend-partitioned module with partial-valid ND-to-2xNZ does "
             "not permit peer imports (pto.import_reserved_buffer)";
      partitionUnsafe = true;
    }
  });
  if (partitionUnsafe) {
    return failure();
  }

  return success();
}
