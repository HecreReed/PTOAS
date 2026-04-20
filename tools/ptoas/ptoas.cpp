// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cctype>
#include <cstring>
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h" // [Fix] Required for OF_None
#include "ptobc/ptobc_decode.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringMap.h"
#include <string>

using namespace mlir;
using namespace pto;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

namespace {
struct EmitCFunctionBuckets {
  SmallVector<emitc::FuncOp> declarations;
  SmallVector<emitc::FuncOp> definitions;
  llvm::DenseMap<StringAttr, emitc::FuncOp> definitionsByName;
};
} // namespace

static EmitCFunctionBuckets collectEmitCFunctions(ModuleOp module) {
  EmitCFunctionBuckets buckets;
  for (auto func : module.getOps<emitc::FuncOp>()) {
    if (func.isDeclaration()) {
      buckets.declarations.push_back(func);
      continue;
    }
    buckets.definitions.push_back(func);
    buckets.definitionsByName[func.getSymNameAttr()] = func;
  }
  return buckets;
}

static LogicalResult buildEmitCCallGraph(
    const EmitCFunctionBuckets &buckets,
    llvm::DenseMap<Operation *, unsigned> &indegree,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &outgoing) {
  for (auto func : buckets.definitions)
    indegree[func.getOperation()] = 0;

  for (auto caller : buckets.definitions) {
    Operation *callerOp = caller.getOperation();
    llvm::SmallPtrSet<Operation *, 8> seenCallees;
    bool hasCycle = false;
    caller.walk([&](emitc::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto it = buckets.definitionsByName.find(calleeAttr.getLeafReference());
      if (it == buckets.definitionsByName.end())
        return;
      Operation *calleeOp = it->second;
      if (calleeOp == callerOp) {
        hasCycle = true;
        return;
      }
      if (!seenCallees.insert(calleeOp).second)
        return;
      outgoing[calleeOp].push_back(callerOp);
      ++indegree[callerOp];
    });
    if (hasCycle) {
      return caller.emitOpError()
             << "recursive function calls are not supported for EmitC C++ "
                "emission";
    }
  }
  return success();
}

static void seedEmitCReadyQueue(
    llvm::ArrayRef<emitc::FuncOp> definitions,
    const llvm::DenseMap<Operation *, unsigned> &indegree,
    SmallVectorImpl<Operation *> &ready) {
  for (auto func : definitions) {
    auto it = indegree.find(func.getOperation());
    if (it != indegree.end() && it->second == 0)
      ready.push_back(func.getOperation());
  }
}

static LogicalResult sortEmitCDefinitions(const EmitCFunctionBuckets &buckets,
                                          ModuleOp module,
                                          SmallVectorImpl<emitc::FuncOp> &out) {
  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> outgoing;
  if (failed(buildEmitCCallGraph(buckets, indegree, outgoing)))
    return failure();

  SmallVector<Operation *> ready;
  seedEmitCReadyQueue(buckets.definitions, indegree, ready);
  for (size_t readyIndex = 0; readyIndex < ready.size(); ++readyIndex) {
    Operation *next = ready[readyIndex];
    out.push_back(cast<emitc::FuncOp>(next));

    for (Operation *user : outgoing[next]) {
      unsigned &userIndegree = indegree[user];
      if (--userIndegree == 0)
        ready.push_back(user);
    }
  }

  if (out.size() != buckets.definitions.size()) {
    return module.emitError()
           << "cyclic function call graph is not supported for EmitC C++ "
              "emission";
  }
  return success();
}

static Operation *findFirstEmitCFunctionAnchor(Block &body) {
  for (Operation &op : body.getOperations()) {
    if (isa<emitc::FuncOp>(op))
      return &op;
  }
  return nullptr;
}

static void advanceEmitCAnchor(Operation *&anchor) {
  while (anchor) {
    anchor = anchor->getNextNode();
    if (!anchor || isa<emitc::FuncOp>(anchor))
      return;
  }
}

static LogicalResult reorderEmitCFunctions(ModuleOp module) {
  EmitCFunctionBuckets buckets = collectEmitCFunctions(module);
  SmallVector<emitc::FuncOp> sortedDefinitions;
  if (failed(sortEmitCDefinitions(buckets, module, sortedDefinitions)))
    return failure();
  if (buckets.declarations.empty() && buckets.definitions.size() <= 1)
    return success();

  SmallVector<emitc::FuncOp> desiredOrder;
  desiredOrder.append(buckets.declarations.begin(), buckets.declarations.end());
  desiredOrder.append(sortedDefinitions.begin(), sortedDefinitions.end());

  Block &body = module.getBodyRegion().front();
  Operation *anchor = findFirstEmitCFunctionAnchor(body);
  if (!anchor)
    return success();

  for (auto func : desiredOrder) {
    if (func.getOperation() == anchor) {
      advanceEmitCAnchor(anchor);
      continue;
    }
    if (anchor)
      func->moveBefore(anchor);
    else
      func->moveBefore(&body, body.end());
  }

  return success();
}

// --------------------------------------------------------------------------
// Command Line Options
// --------------------------------------------------------------------------
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o",
                                                 llvm::cl::desc("Output filename"),
                                                 llvm::cl::value_desc("filename"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<bool> enableInsertSync("enable-insert-sync",
                                            llvm::cl::desc("Enable automatic synchronization insertion pass"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> ptoTargetArch(
    "pto-arch",
    llvm::cl::desc("Target Ascend architecture for codegen: a3 or a5 (default: a3)"),
    llvm::cl::value_desc("a3|a5"),
    llvm::cl::init("a3"));

static llvm::cl::opt<std::string> ptoBuildLevel(
    "pto-level",
    llvm::cl::desc("Build level for pass pipeline: level1, level2, or level3 (default: level2)"),
    llvm::cl::value_desc("level1|level2|level3"),
    llvm::cl::init("level2"));

enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

static PTOBuildLevel defaultBuildLevel() {
  return PTOBuildLevel::Level2;
}

static bool parseBuildLevel(llvm::StringRef levelStr, PTOBuildLevel &out) {
  std::string s = levelStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "level1") {
    out = PTOBuildLevel::Level1;
    return true;
  }
  if (s == "level2") {
    out = PTOBuildLevel::Level2;
    return true;
  }
  if (s == "level3") {
    out = PTOBuildLevel::Level3;
    return true;
  }
  return false;
}

static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";

static bool parseAutoSyncTailHint(llvm::StringRef hintStr, std::string &normalized) {
  std::string s = hintStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "barrier-all" || s == "barrier_all" || s == "default") {
    normalized = kAutoSyncTailPolicyBarrierAll.str();
    return true;
  }
  if (s == "mte3-to-s-event0" || s == "mte3_to_s_event0" ||
      s == "setwait-mte3-to-s-event0" ||
      s == "setwait_mte3_to_s_event0") {
    normalized = kAutoSyncTailPolicyMte3ToSEvent0.str();
    return true;
  }
  return false;
}

// --------------------------------------------------------------------------
// Post-process C++ output: rewrite marker calls into Tile member calls.
//
// We emit marker calls in EmitC IR because EmitC currently does not provide a
// first-class op for member-function invocation. After translation, we rewrite:
//   PTOAS__TILE_SET_VALUE(dst, offset, val) -> dst.SetValue(offset, val)
//   PTOAS__TILE_GET_VALUE(src, offset)      -> src.GetValue(offset)
//   PTOAS__TILE_DATA(obj)                   -> obj.data()
//   PTOAS__TILE_SET_VALIDSHAPE(obj, r, c)   -> obj.SetValidShape(r, c)
//   PTOAS__PTR_LOAD(ptr, offset)            -> ptr[offset]
//   PTOAS__PTR_STORE(ptr, offset, val)      -> ptr[offset] = val
//   PTOAS__EVENTID_ARRAY_LOAD(arr, idx)     -> arr[idx]
//   PTOAS__EVENTID_ARRAY_STORE(arr, idx, v) -> arr[idx] = v
// --------------------------------------------------------------------------
struct ParsedMarkerCall {
  size_t markerPos = std::string::npos;
  size_t rparenPos = std::string::npos;
  llvm::SmallVector<llvm::StringRef, 4> args;
};

struct MarkerRewriteSpec {
  llvm::StringRef marker;
  llvm::StringRef memberName;
  unsigned expectedNumArgs = 0;
};

struct MarkerSubscriptRewriteSpec {
  llvm::StringRef marker;
  unsigned expectedNumArgs = 0;
  bool isStore = false;
};

static bool parseMarkerArgs(llvm::StringRef argsRef,
                            llvm::SmallVectorImpl<llvm::StringRef> &args) {
  size_t partBegin = 0;
  int parenDepth = 0;
  for (size_t i = 0; i < argsRef.size(); ++i) {
    char c = argsRef[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c == ')') {
      if (parenDepth > 0)
        --parenDepth;
      continue;
    }
    if (c == ',' && parenDepth == 0) {
      args.push_back(argsRef.slice(partBegin, i).trim());
      partBegin = i + 1;
    }
  }
  if (partBegin > argsRef.size())
    return false;
  args.push_back(argsRef.drop_front(partBegin).trim());
  return true;
}

static std::optional<ParsedMarkerCall>
findNextMarkerCall(const std::string &cpp, llvm::StringRef marker,
                   size_t searchPos) {
  ParsedMarkerCall call;
  call.markerPos = cpp.find(marker.str(), searchPos);
  if (call.markerPos == std::string::npos)
    return std::nullopt;

  size_t lparenPos = call.markerPos + marker.size();
  if (lparenPos >= cpp.size() || cpp[lparenPos] != '(')
    return ParsedMarkerCall{call.markerPos, std::string::npos, {}};

  size_t argsBegin = lparenPos + 1;
  int parenDepth = 0;
  for (size_t i = argsBegin; i < cpp.size(); ++i) {
    char c = cpp[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c != ')')
      continue;
    if (parenDepth == 0) {
      call.rparenPos = i;
      break;
    }
    --parenDepth;
  }
  if (call.rparenPos == std::string::npos)
    return call;

  llvm::StringRef argsRef(cpp.data() + argsBegin, call.rparenPos - argsBegin);
  if (!parseMarkerArgs(argsRef, call.args))
    call.args.clear();
  return call;
}

static bool rewriteMarkerCalls(
    std::string &cpp, llvm::StringRef marker,
    llvm::function_ref<std::optional<std::string>(const ParsedMarkerCall &)>
        buildReplacement) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, marker, searchPos); call;
       call = findNextMarkerCall(cpp, marker, searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + marker.size();
      continue;
    }

    std::optional<std::string> replacement = buildReplacement(*call);
    if (!replacement) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    cpp.replace(call->markerPos, (call->rparenPos - call->markerPos) + 1,
                *replacement);
    changed = true;
    searchPos = call->markerPos + replacement->size();
  }
  return changed;
}

static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;

        std::string replacement;
        replacement.reserve(marker.size() + 16);
        replacement.append(call.args[0].str());
        replacement.push_back('.');
        replacement.append(memberName.str());
        replacement.push_back('(');
        if (expectedNumArgs >= 2)
          replacement.append(call.args[1].str());
        if (expectedNumArgs == 3) {
          replacement.append(", ");
          replacement.append(call.args[2].str());
        }
        replacement.push_back(')');
        return replacement;
      });
}

static void rewriteMarkerCallsToMembers(
    std::string &cpp, llvm::ArrayRef<MarkerRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToMember(cpp, rewrite.marker,
                                           rewrite.memberName,
                                           rewrite.expectedNumArgs);
    }
  }
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  static constexpr MarkerRewriteSpec kTileMarkerRewrites[] = {
      {"PTOAS__TILE_SET_VALUE", "SetValue", 3},
      {"PTOAS__TILE_GET_VALUE", "GetValue", 2},
      {"PTOAS__TILE_DATA", "data", 1},
      {"PTOAS__TILE_SET_VALIDSHAPE", "SetValidShape", 3},
  };
  rewriteMarkerCallsToMembers(cpp, kTileMarkerRewrites);
}

static void rewriteAsyncEventMarkers(std::string &cpp) {
  static constexpr MarkerRewriteSpec kAsyncEventMarkerRewrites[] = {
      {"PTOAS__ASYNC_EVENT_WAIT", "Wait", 2},
      {"PTOAS__ASYNC_EVENT_TEST", "Test", 2},
  };
  rewriteMarkerCallsToMembers(cpp, kAsyncEventMarkerRewrites);
}

static bool rewriteMarkerCallToSubscript(std::string &cpp, llvm::StringRef marker,
                                         unsigned expectedNumArgs,
                                         bool isStore) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (isStore)
          return (call.args[0] + "[" + call.args[1] + "] = " + call.args[2])
              .str();
        return (call.args[0] + "[" + call.args[1] + "]").str();
      });
}

static void rewriteMarkerCallsToSubscripts(
    std::string &cpp, llvm::ArrayRef<MarkerSubscriptRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerSubscriptRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToSubscript(cpp, rewrite.marker,
                                              rewrite.expectedNumArgs,
                                              rewrite.isStore);
    }
  }
}

static void rewritePtrScalarMarkers(std::string &cpp) {
  static constexpr MarkerSubscriptRewriteSpec kPtrScalarMarkerRewrites[] = {
      {"PTOAS__PTR_LOAD", 2, false},
      {"PTOAS__PTR_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kPtrScalarMarkerRewrites);
}

static void rewriteEventIdArrayMarkers(std::string &cpp) {
  static constexpr MarkerSubscriptRewriteSpec kEventIdArrayMarkerRewrites[] = {
      {"PTOAS__EVENTID_ARRAY_LOAD", 2, false},
      {"PTOAS__EVENTID_ARRAY_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kEventIdArrayMarkerRewrites);
}

// --------------------------------------------------------------------------
// EmitC cleanup: drop empty emitc.expression ops.
//
// After FormExpressions + CSE, EmitC expressions can become empty when their
// root op is CSE'd with an equivalent dominating value outside the expression
// region. Such expressions crash mlir::emitc::translateToCpp because
// ExpressionOp::getRootOp() returns nullptr.
// --------------------------------------------------------------------------
static void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, 8> toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    if (expr.getRootOp())
      return;
    Block *body = expr.getBody();
    if (!body)
      return;
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    Value yielded = yield.getOperand(0);
    expr.getResult().replaceAllUsesWith(yielded);
    toErase.push_back(expr);
  });
  for (emitc::ExpressionOp expr : llvm::reverse(toErase))
    expr.erase();
}

static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(intTy, 0);
  if (isa<IndexType>(type))
    return builder.getIndexAttr(0);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(floatTy, 0.0);
  if (isa<emitc::OpaqueType, emitc::PointerType>(type))
    return emitc::OpaqueAttr::get(builder.getContext(), "");
  return Attribute{};
}

// FormExpressions may inline conditions into emitc.expression, but the C++
// emitter prints cf.br/cf.cond_br operands by variable name rather than by
// recursively emitting an expression. Materialize such operands so CFG-based
// lowering (e.g. scf.while -> cf.*) stays valid.
static void materializeControlFlowOperands(Operation *rootOp) {
  llvm::SmallVector<Operation *, 16> branches;
  rootOp->walk([&](Operation *op) {
    if (isa<cf::BranchOp, cf::CondBranchOp>(op))
      branches.push_back(op);
  });

  OpBuilder builder(rootOp->getContext());
  for (Operation *op : branches) {
    builder.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
      if (!expr)
        continue;

      Attribute initAttr =
          getDefaultEmitCVariableInitAttr(builder, value.getType());
      if (!initAttr)
        continue;

      Value tmp =
          builder.create<emitc::VariableOp>(op->getLoc(), value.getType(),
                                            initAttr)
              .getResult();
      builder.create<emitc::AssignOp>(op->getLoc(), tmp, value);
      operand.set(tmp);
    }
  }
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos);
       call; call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + 1;
      continue;
    }
    if (call->args.size() != 3) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (showTrace) {
      replacement.reserve(64);
      replacement.append("/* ADDPTR_TRACE: ");
      replacement.append(call->args[0].str());
      replacement.append(" = ");
      replacement.append(call->args[1].str());
      replacement.append(" + ");
      replacement.append(call->args[2].str());
      replacement.append(" */");
    }

    size_t replaceEnd = call->rparenPos;
    if (!showTrace) {
      size_t i = call->rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i])))
        ++i;
      if (i < cpp.size() && cpp[i] == ';')
        replaceEnd = i;
    }

    cpp.replace(call->markerPos, (replaceEnd - call->markerPos) + 1,
                replacement);
    changed = true;
    searchPos = call->markerPos + replacement.size();
  }
  return changed;
}

static bool isGeneratedGlobalTensorDecl(llvm::StringRef trimmed,
                                        llvm::StringRef &decl,
                                        llvm::StringRef &varName) {
  if (!trimmed.starts_with("GlobalTensor<") || !trimmed.ends_with(";") ||
      trimmed.contains('=') || trimmed.contains('(')) {
    return false;
  }

  decl = trimmed.drop_back().rtrim();
  size_t lastWs = decl.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;
  varName = decl.drop_front(lastWs + 1);
  if (!varName.starts_with("v") || varName.size() <= 1)
    return false;
  return llvm::all_of(varName.drop_front(1),
                      [](char c) { return std::isdigit(c); });
}

static void appendNullInitializedGlobalTensorDecl(std::string &out,
                                                  llvm::StringRef line,
                                                  llvm::StringRef decl) {
  size_t indentLen = line.find_first_not_of(" \t");
  if (indentLen == std::string::npos)
    indentLen = 0;
  llvm::StringRef indent = line.take_front(indentLen);
  out.append(indent.str());
  out.append(decl.str());
  out.append("(nullptr);");
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  // When `declareVariablesAtTop` is enabled, the C++ emitter hoists SSA value
  // declarations to the top of the function and emits assignments later. This
  // requires the C++ type to be default-constructible.
  //
  // `GlobalTensor<...>` from pto-isa does NOT have a default constructor, so a
  // hoisted declaration like:
  //   GlobalTensor<...> v42;
  // fails to compile. Initialize those hoisted temporaries with a null pointer
  // so they are constructible:
  //   GlobalTensor<...> v42(nullptr);
  //
  // We keep the assignment later; the null-initialized value is never used.
  std::string out;
  out.reserve(cpp.size() + 64);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    llvm::StringRef decl;
    llvm::StringRef varName;
    if (isGeneratedGlobalTensorDecl(trimmed, decl, varName)) {
      appendNullInitializedGlobalTensorDecl(out, line, decl);
      rewritten = true;
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

namespace {
struct ConstantDeclCandidate {
  size_t declLine = 0;
  std::string indent;
  std::string type;
  bool hasInitializer = false;
  std::string initializer;
  size_t assignmentCount = 0;
  size_t assignmentLine = 0;
  std::string assignmentRhs;
};
} // namespace

static bool isGeneratedValueName(llvm::StringRef name) {
  if (!name.consume_front("v") || name.empty())
    return false;
  return llvm::all_of(name, [](char c) { return std::isdigit(c); });
}

static bool isConstFoldableScalarType(llvm::StringRef type) {
  type = type.trim();
  if (type.starts_with("const ") || type.starts_with("constexpr "))
    return false;
  return llvm::StringSwitch<bool>(type)
      .Cases("bool", "float", "double", "half", "bfloat16_t", true)
      .Cases("int8_t", "uint8_t", "int16_t", "uint16_t", true)
      .Cases("int32_t", "uint32_t", "int64_t", "uint64_t", true)
      .Default(false);
}

static bool isLiteralInitializer(llvm::StringRef rhs) {
  rhs = rhs.trim();
  if (rhs.empty())
    return false;
  if (rhs == "true" || rhs == "false" || rhs == "nullptr")
    return true;

  static const llvm::Regex kIntLiteral(
      R"(^[+-]?(0[xX][0-9A-Fa-f]+|[0-9]+)[uUlL]*$)");
  static const llvm::Regex kFloatLiteral(
      R"(^[+-]?(([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)[fF]?$)");
  static const llvm::Regex kHexFloatLiteral(
      R"(^[+-]?0[xX]([0-9A-Fa-f]+\.[0-9A-Fa-f]*|[0-9A-Fa-f]+|\.[0-9A-Fa-f]+)[pP][+-]?[0-9]+[fF]?$)");
  static const llvm::Regex kSpecialFloatLiteral(
      R"(^[+-]?(nan|inf)[fF]?$)");

  return kIntLiteral.match(rhs) || kFloatLiteral.match(rhs) ||
         kHexFloatLiteral.match(rhs) || kSpecialFloatLiteral.match(rhs);
}

static std::string normalizeConstInitializer(llvm::StringRef type,
                                             llvm::StringRef rhs) {
  type = type.trim();
  rhs = rhs.trim();
  if (type == "bool") {
    if (rhs == "0" || rhs == "false")
      return "false";
    if (rhs == "1" || rhs == "-1" || rhs == "true")
      return "true";
  }
  return rhs.str();
}

static bool shouldSkipConstantDeclarationLine(llvm::StringRef body) {
  return body.starts_with("return") || body.starts_with("goto ") ||
         body.starts_with("if ") || body.starts_with("if(") ||
         body.starts_with("switch ") || body.starts_with("switch(") ||
         body.starts_with("for ") || body.starts_with("for(") ||
         body.starts_with("while ") || body.starts_with("while(") ||
         body.starts_with("case ") || body == "default";
}

static std::pair<llvm::StringRef, llvm::StringRef>
splitDeclarationAssignment(llvm::StringRef body) {
  if (size_t eqPos = body.find('='); eqPos != llvm::StringRef::npos) {
    return {body.take_front(eqPos).rtrim(), body.drop_front(eqPos + 1).trim()};
  }
  return {body, llvm::StringRef()};
}

static bool parseTypedGeneratedValue(llvm::StringRef lhs, llvm::StringRef &type,
                                     llvm::StringRef &name) {
  size_t lastWs = lhs.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;

  type = lhs.take_front(lastWs).rtrim();
  name = lhs.drop_front(lastWs + 1).trim();
  return isGeneratedValueName(name) && isConstFoldableScalarType(type);
}

static bool parseConstantDeclarationLine(llvm::StringRef line,
                                         ConstantDeclCandidate &candidate,
                                         std::string &valueName) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  if (shouldSkipConstantDeclarationLine(body))
    return false;

  auto [lhs, rhs] = splitDeclarationAssignment(body);
  llvm::StringRef type;
  llvm::StringRef name;
  if (!parseTypedGeneratedValue(lhs, type, name))
    return false;

  size_t indentLen = line.find_first_not_of(" \t");
  if (indentLen == llvm::StringRef::npos)
    indentLen = 0;
  candidate.indent = line.take_front(indentLen).str();
  candidate.type = type.str();
  valueName = name.str();

  if (!rhs.empty()) {
    if (!isLiteralInitializer(rhs))
      return false;
    candidate.hasInitializer = true;
    candidate.initializer = normalizeConstInitializer(type, rhs);
  }

  return true;
}

static bool parseGeneratedValueAssignment(llvm::StringRef line,
                                          llvm::StringRef &valueName,
                                          llvm::StringRef &rhs) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  size_t eqPos = body.find('=');
  if (eqPos == llvm::StringRef::npos)
    return false;

  llvm::StringRef lhs = body.take_front(eqPos).rtrim();
  rhs = body.drop_front(eqPos + 1).trim();
  if (!isGeneratedValueName(lhs))
    return false;
  valueName = lhs;
  return true;
}

static llvm::SmallVector<std::string, 0>
splitLinesPreservingText(llvm::StringRef text) {
  llvm::SmallVector<std::string, 0> lines;
  for (llvm::StringRef ref(text); !ref.empty(); ref = ref.split('\n').second) {
    auto split = ref.split('\n');
    lines.push_back(split.first.str());
  }
  return lines;
}

static void collectConstantDeclCandidates(
    llvm::ArrayRef<std::string> lines, size_t beginLine, size_t endLine,
    llvm::StringMap<ConstantDeclCandidate> &candidates) {
  for (size_t i = beginLine; i <= endLine; ++i) {
    ConstantDeclCandidate candidate;
    std::string valueName;
    if (!parseConstantDeclarationLine(lines[i], candidate, valueName))
      continue;
    candidate.declLine = i;
    candidates[valueName] = std::move(candidate);
  }
}

static void collectConstantAssignments(
    llvm::ArrayRef<std::string> lines, size_t beginLine, size_t endLine,
    llvm::StringMap<ConstantDeclCandidate> &candidates) {
  for (size_t i = beginLine; i <= endLine; ++i) {
    llvm::StringRef assignedName;
    llvm::StringRef rhs;
    if (!parseGeneratedValueAssignment(lines[i], assignedName, rhs))
      continue;

    auto it = candidates.find(assignedName);
    if (it == candidates.end())
      continue;

    ConstantDeclCandidate &info = it->second;
    ++info.assignmentCount;
    info.assignmentLine = i;
    info.assignmentRhs = rhs.str();
  }
}

static std::optional<std::string>
buildFoldedConstDeclaration(llvm::StringRef valueName,
                            ConstantDeclCandidate &info,
                            llvm::SmallVectorImpl<bool> &eraseLine) {
  std::string initializer;
  if (info.hasInitializer) {
    if (info.assignmentCount != 0)
      return std::nullopt;
    initializer = info.initializer;
  } else {
    if (info.assignmentCount != 1)
      return std::nullopt;
    if (!isLiteralInitializer(info.assignmentRhs))
      return std::nullopt;
    initializer =
        normalizeConstInitializer(info.type, llvm::StringRef(info.assignmentRhs));
    eraseLine[info.assignmentLine] = true;
  }
  return info.indent + "const " + info.type + " " + valueName.str() + " = " +
         initializer + ";";
}

static void rewriteConstantSegment(llvm::SmallVectorImpl<std::string> &lines,
                                   llvm::SmallVectorImpl<bool> &eraseLine,
                                   size_t beginLine, size_t endLine) {
  llvm::StringMap<ConstantDeclCandidate> candidates;
  collectConstantDeclCandidates(lines, beginLine, endLine, candidates);
  collectConstantAssignments(lines, beginLine, endLine, candidates);

  for (auto &entry : candidates) {
    if (auto rewritten =
            buildFoldedConstDeclaration(entry.getKey(), entry.getValue(),
                                        eraseLine)) {
      lines[entry.getValue().declLine] = *rewritten;
    }
  }
}

static void rewriteScalarConstantDecls(std::string &cpp) {
  llvm::SmallVector<std::string, 0> lines = splitLinesPreservingText(cpp);

  llvm::SmallVector<bool, 0> eraseLine(lines.size(), false);
  int braceDepth = 0;
  size_t segmentStart = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    int depthBefore = braceDepth;
    for (char c : lines[i]) {
      if (c == '{')
        ++braceDepth;
      else if (c == '}')
        --braceDepth;
    }

    if (depthBefore == 0 && braceDepth > 0)
      segmentStart = i;
    if (depthBefore > 0 && braceDepth == 0)
      rewriteConstantSegment(lines, eraseLine, segmentStart, i);
  }

  std::string out;
  out.reserve(cpp.size());
  for (size_t i = 0; i < lines.size(); ++i) {
    if (eraseLine[i])
      continue;
    out.append(lines[i]);
    if (i + 1 != lines.size())
      out.push_back('\n');
  }
  cpp.swap(out);
}

static bool shouldDeclareVariablesAtTop(ModuleOp module) {
  auto hasMultiBlockFunc = [](auto func) { return func.getBlocks().size() > 1; };
  return llvm::any_of(module.getOps<func::FuncOp>(), hasMultiBlockFunc) ||
         llvm::any_of(module.getOps<emitc::FuncOp>(), hasMultiBlockFunc);
}

static DialectRegistry createDialectRegistry() {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  registry.insert<mlir::pto::PTODialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  return registry;
}

static bool isCliArchSpecified(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg == "--pto-arch" || arg.starts_with("--pto-arch="))
      return true;
  }
  return false;
}

static std::string normalizeArchValue(llvm::StringRef archValue) {
  std::string normalized = archValue.str();
  for (char &c : normalized)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return normalized;
}

static std::optional<std::string>
detectTextualModuleArch(llvm::StringRef text) {
  llvm::SmallVector<llvm::StringRef, 4> matches;
  llvm::Regex archRegex(
      R"ptoarch("?(pto\.target_arch)"?[[:space:]]*=[[:space:]]*"([[:alpha:][:digit:]_]+)")ptoarch");
  if (!archRegex.match(text, &matches) || matches.size() < 3)
    return std::nullopt;
  return normalizeArchValue(matches[2]);
}

static std::string resolveTargetArch(bool cliArchSpecified, bool isPTOBC,
                                     llvm::StringRef buffer) {
  std::string arch = normalizeArchValue(ptoTargetArch);
  if (cliArchSpecified) {
    if (arch != "a3" && arch != "a5")
      return {};
    return arch;
  }
  if (!isPTOBC) {
    if (auto detectedArch = detectTextualModuleArch(buffer))
      arch = *detectedArch;
  }
  return arch == "a3" || arch == "a5" ? arch : "a3";
}

static void loadRequiredDialects(MLIRContext &context) {
  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

static LogicalResult loadInputModule(std::unique_ptr<llvm::MemoryBuffer> input,
                                     MLIRContext &context,
                                     llvm::StringRef arch,
                                     OwningOpRef<ModuleOp> &module) {
  llvm::StringRef buf = input->getBuffer();
  const bool isPTOBC =
      buf.size() >= 6 && std::memcmp(buf.data(), "PTOBC\0", 6) == 0;
  if (isPTOBC) {
    llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(buf.data()),
                                  buf.size());
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    try {
      module = ptobc::decodePTOBCToModule(bytes, context);
    } catch (...) {
      return failure();
    }
#else
    module = ptobc::decodePTOBCToModule(bytes, context);
#endif
    return success(static_cast<bool>(module));
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  pto::ScopedPTOParserTargetArch scopedParserArch(
      &context,
      arch == "a5" ? pto::PTOParserTargetArch::A5 : pto::PTOParserTargetArch::A3);
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  return success(static_cast<bool>(module));
}

static void setModuleTargetArch(ModuleOp module, MLIRContext &context,
                                llvm::StringRef arch, bool cliArchSpecified) {
  if (cliArchSpecified || !module->hasAttr("pto.target_arch")) {
    module->setAttr("pto.target_arch", mlir::StringAttr::get(&context, arch));
  }
}

static LogicalResult normalizeAutoSyncTailHints(ModuleOp module,
                                                MLIRContext &context) {
  bool invalidAutoSyncTailHint = false;
  module.walk([&](mlir::func::FuncOp func) {
    auto hintAttr =
        func->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint");
    if (!hintAttr)
      return;

    std::string normalizedHint;
    if (!parseAutoSyncTailHint(hintAttr.getValue(), normalizedHint)) {
      func.emitError("invalid pto.auto_sync_tail_hint '")
          << hintAttr.getValue()
          << "'. Expected 'barrier-all' (or 'default') or "
             "'mte3-to-s-event0'.";
      invalidAutoSyncTailHint = true;
      return;
    }
    func->setAttr("pto.auto_sync_tail_hint",
                  mlir::StringAttr::get(&context, normalizedHint));
  });
  return success(!invalidAutoSyncTailHint);
}

static bool moduleHasTAssign(ModuleOp module) {
  bool hasTAssign = false;
  module.walk([&](pto::TAssignOp) { hasTAssign = true; });
  return hasTAssign;
}

static LogicalResult validateTAssignOptions(bool hasTAssign,
                                            PTOBuildLevel effectiveLevel,
                                            bool enableInsertSyncFlag) {
  if (hasTAssign && effectiveLevel != PTOBuildLevel::Level3) {
    llvm::errs() << "Error: pto.tassign is only supported when "
                    "--pto-level=level3.\n";
    return failure();
  }
  if (hasTAssign && enableInsertSyncFlag) {
    llvm::errs() << "Error: pto.tassign requires --enable-insert-sync to be "
                    "disabled.\n";
    return failure();
  }
  return success();
}

static LogicalResult validateAllocTileLevel(ModuleOp module,
                                            PTOBuildLevel effectiveLevel) {
  if (effectiveLevel == PTOBuildLevel::Level3) {
    bool missing = false;
    module.walk([&](pto::AllocTileOp op) {
      if (!op.getAddr()) {
        op.emitError("requires 'addr' operand when --pto-level=level3");
        missing = true;
      }
    });
    return success(!missing);
  }

  bool hasAddr = false;
  module.walk([&](pto::AllocTileOp op) {
    if (op.getAddr()) {
      op.emitError(
          "unexpected 'addr' operand: only supported when --pto-level=level3");
      hasAddr = true;
    }
  });
  return success(!hasAddr);
}

static void populatePassManager(PassManager &pm, llvm::StringRef arch,
                                PTOBuildLevel effectiveLevel) {
  pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOLowerFrontendPipeOpsPass());
  pm.addPass(pto::createPTOInferValidatePipeInitPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOA5NormalizeTMovPass());
  pm.addPass(pto::createPTOViewToMemrefPass());

  if (effectiveLevel != PTOBuildLevel::Level3) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::LOCAL_MEM_PLAN;
    planMemoryOption.enableGlobalReuse = false;
    planMemoryOption.enablePrintMemoryAllocatedSize = false;
    pm.addPass(pto::createPlanMemoryPass(planMemoryOption));
  }
  pm.addPass(pto::createPTOResolveReservedBuffersPass());
  if (enableInsertSync)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());

  pm.addPass(createCSEPass());
  pm.addPass(arch == "a3" ? pto::createEmitPTOManualPass(pto::PTOArch::A3)
                          : pto::createEmitPTOManualPass(pto::PTOArch::A5));
  pm.addPass(emitc::createFormExpressionsPass());
  pm.addPass(mlir::createCSEPass());
}

static LogicalResult translateModuleToCpp(ModuleOp module, bool emitAddPtrTraceFlag,
                                          std::string &cppOutput) {
  llvm::raw_string_ostream cppOS(cppOutput);
  bool declareVariablesAtTop = shouldDeclareVariablesAtTop(module);
  if (failed(emitc::translateToCpp(
          module, cppOS,
          /*declareVariablesAtTop=*/declareVariablesAtTop))) {
    return failure();
  }
  cppOS.flush();
  rewriteTileGetSetValueMarkers(cppOutput);
  rewriteAsyncEventMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteEventIdArrayMarkers(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTraceFlag);
  rewriteScalarConstantDecls(cppOutput);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  return success();
}

static int loadPreparedModule(bool cliArchSpecified, MLIRContext &context,
                              std::string &arch, bool &isPTOBC,
                              OwningOpRef<ModuleOp> &module) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Error: Could not open input file: "
                 << fileOrErr.getError().message() << "\n";
    return 1;
  }

  auto inputBuffer = std::move(*fileOrErr);
  llvm::StringRef buf = inputBuffer->getBuffer();
  isPTOBC = buf.size() >= 6 && std::memcmp(buf.data(), "PTOBC\0", 6) == 0;
  arch = resolveTargetArch(cliArchSpecified, isPTOBC, buf);
  if (arch.empty()) {
    llvm::errs() << "Error: invalid --pto-arch='" << ptoTargetArch
                 << "'. Expected 'a3' or 'a5'.\n";
    return 1;
  }

  if (failed(loadInputModule(std::move(inputBuffer), context, arch, module))) {
    llvm::errs() << "Error: Failed to "
                 << (isPTOBC ? "decode PTOBC.\n" : "parse MLIR.\n");
    return 1;
  }
  setModuleTargetArch(*module, context, arch, cliArchSpecified);
  return 0;
}

static int runConfiguredPassPipeline(ModuleOp module, MLIRContext &context,
                                     llvm::StringRef arch,
                                     PTOBuildLevel effectiveLevel) {
  if (failed(normalizeAutoSyncTailHints(module, context)))
    return 1;
  if (failed(validateTAssignOptions(moduleHasTAssign(module), effectiveLevel,
                                    enableInsertSync))) {
    return 1;
  }
  if (failed(validateAllocTileLevel(module, effectiveLevel)))
    return 1;

  PassManager pm(&context);
  populatePassManager(pm, arch, effectiveLevel);
  if (failed(pm.run(module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  dropEmptyEmitCExpressions(module);
  materializeControlFlowOperands(module);
  if (failed(reorderEmitCFunctions(module))) {
    llvm::errs() << "Error: Failed to order emitted functions for C++ emission.\n";
    return 1;
  }
  return 0;
}

static int emitModuleCppToFile(ModuleOp module) {
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return 1;
  }

  std::string cppOutput;
  if (failed(translateModuleToCpp(module, emitAddPtrTrace, cppOutput))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  outputFile.os() << cppOutput;
  outputFile.keep();
  return 0;
}

static int runPTOAS(int argc, char **argv) {
  DialectRegistry registry = createDialectRegistry();
  llvm::cl::SetVersionPrinter(printPTOASVersion);
  bool cliArchSpecified = isCliArchSpecified(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  MLIRContext context(registry);
  context.allowUnregisteredDialects(true);
  loadRequiredDialects(context);

  std::string arch;
  bool isPTOBC = false;
  OwningOpRef<ModuleOp> module;
  if (int rc = loadPreparedModule(cliArchSpecified, context, arch, isPTOBC,
                                  module)) {
    return rc;
  }

  PTOBuildLevel effectiveLevel = defaultBuildLevel();
  if (!parseBuildLevel(ptoBuildLevel, effectiveLevel)) {
    llvm::errs() << "Error: invalid --pto-level='" << ptoBuildLevel
                 << "'. Expected 'level1', 'level2', or 'level3'.\n";
    return 1;
  }
  if (int rc =
          runConfiguredPassPipeline(*module, context, arch, effectiveLevel)) {
    return rc;
  }
  return emitModuleCppToFile(*module);
}

int main(int argc, char **argv) { return runPTOAS(argc, argv); }
