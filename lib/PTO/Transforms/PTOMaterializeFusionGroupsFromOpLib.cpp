#include "PTO/IR/PTO.h"
#include "PTO/IR/CCEC.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <system_error>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZEFUSIONGROUPSFROMOPLIB
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

static constexpr llvm::StringLiteral kOpLibAttrOp = "pto.oplib.op";
static constexpr llvm::StringLiteral kOpLibAttrKind = "pto.oplib.kind";
static constexpr llvm::StringLiteral kOpLibAttrRank = "pto.oplib.rank";
static constexpr llvm::StringLiteral kOpLibAttrSeedDType = "pto.oplib.seed_dtype";

static constexpr llvm::StringLiteral kOpLibKindBinaryTemplate =
    "binary_elementwise_template";

static bool isSupportedFusionOp(Operation *op) {
  return isa<pto::TMulOp, pto::TDivOp, pto::TAddOp, pto::TSubOp,
             pto::TMaxOp, pto::TMinOp>(op);
}

static StringRef getFusionOpName(Operation *op) {
  if (isa<pto::TMulOp>(op))
    return "tmul";
  if (isa<pto::TDivOp>(op))
    return "tdiv";
  if (isa<pto::TAddOp>(op))
    return "tadd";
  if (isa<pto::TSubOp>(op))
    return "tsub";
  if (isa<pto::TMaxOp>(op))
    return "tmax";
  if (isa<pto::TMinOp>(op))
    return "tmin";
  return "";
}

static Value getBinaryDst(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return mul.getDst();
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return div.getDst();
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return add.getDst();
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return sub.getDst();
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return max.getDst();
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return min.getDst();
  return {};
}

static SmallVector<Value, 2> getBinarySrcs(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return {mul.getSrc0(), mul.getSrc1()};
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return {div.getSrc0(), div.getSrc1()};
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return {add.getSrc0(), add.getSrc1()};
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return {sub.getSrc0(), sub.getSrc1()};
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return {max.getSrc0(), max.getSrc1()};
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return {min.getSrc0(), min.getSrc1()};
  return {};
}

static bool hasFusionGroupAttrs(Operation *op) {
  return op->hasAttr(kFusionGroupIdAttr) && op->hasAttr(kFusionOrderAttr);
}

static bool isAllowedTemplateOp(Operation *op) {
  if (isa<func::FuncOp, func::ReturnOp, scf::YieldOp, scf::ForOp,
          memref::LoadOp, memref::StoreOp, memref::DimOp,
          arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::MaximumFOp, arith::MinimumFOp, ccec::VBinOp,
          arith::ConstantIndexOp, arith::ConstantIntOp, arith::ConstantOp>(op))
    return true;
  return false;
}

static bool isForbiddenTemplateOp(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
             func::CallOp, scf::IfOp, scf::WhileOp>(op);
}

static bool isExpectedArithOpForTemplate(Operation *op, StringRef opName) {
  if (opName == "tadd")
    return isa<arith::AddFOp>(op);
  if (opName == "tsub")
    return isa<arith::SubFOp>(op);
  if (opName == "tmul")
    return isa<arith::MulFOp>(op);
  if (opName == "tdiv")
    return isa<arith::DivFOp>(op);
  if (opName == "tmax")
    return isa<arith::MaximumFOp>(op);
  if (opName == "tmin")
    return isa<arith::MinimumFOp>(op);
  return false;
}

static bool isTemplateArithCoreOp(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

static bool isExpectedCCECOpForTemplate(Operation *op, StringRef opName) {
  auto vbin = dyn_cast<ccec::VBinOp>(op);
  if (!vbin)
    return false;
  StringRef expectedKind = ccec::getExpectedVBinKindForOpLibOp(opName);
  return !expectedKind.empty() && vbin.getKind() == expectedKind;
}

static MemRefType getAllocCompatibleMemRefType(MemRefType ty) {
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(ty, strides, offset)))
    return ty;
  if (offset == 0)
    return ty;
  return MemRefType::get(ty.getShape(), ty.getElementType(),
                         StridedLayoutAttr::get(ty.getContext(), 0, strides),
                         ty.getMemorySpace());
}

static bool isFloatDTypeSupported(Type ty) {
  return ty.isF16() || ty.isF32();
}

static std::string dtypeToString(Type ty) {
  if (ty.isF16())
    return "f16";
  if (ty.isF32())
    return "f32";
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  return os.str();
}

static std::string memRefTypeToString(MemRefType ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  return os.str();
}

static Value castToMemRefTypeIfNeeded(OpBuilder &builder, Location loc, Value v,
                                      MemRefType dstTy) {
  if (v.getType() == dstTy)
    return v;

  auto srcTy = dyn_cast<MemRefType>(v.getType());
  if (!srcTy)
    return {};
  if (srcTy.getRank() != dstTy.getRank())
    return {};
  if (srcTy.getElementType() != dstTy.getElementType())
    return {};
  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return {};

  return builder.create<memref::CastOp>(loc, dstTy, v);
}

static bool valueInList(ArrayRef<Value> vals, Value v) {
  return llvm::is_contained(vals, v);
}

static void appendUniqueValue(SmallVectorImpl<Value> &vals, Value v) {
  if (!valueInList(vals, v))
    vals.push_back(v);
}

static void eraseDeadValueDefChain(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || !def->use_empty())
    return;

  if (!isa<memref::AllocOp, pto::BindTileOp, memref::CastOp,
           memref::ReinterpretCastOp, memref::SubViewOp>(def))
    return;

  SmallVector<Value, 4> operands(def->getOperands().begin(), def->getOperands().end());
  def->erase();
  for (Value operand : operands)
    eraseDeadValueDefChain(operand);
}

struct GroupInfo {
  int64_t groupId = -1;
  SmallVector<Operation *, 8> ops;
};

struct GroupInterface {
  DenseSet<Operation *> groupSet;
  SmallVector<Value, 8> producedInOrder;
  DenseSet<Value> producedSet;
  SmallVector<Value, 8> externalInputs;
  SmallVector<Value, 8> externalOutputs;
  SmallVector<Value, 8> callArgs;
};

struct TemplateRegistry;

static GroupInterface buildGroupInterface(GroupInfo &group) {
  GroupInterface iface;
  iface.groupSet.insert(group.ops.begin(), group.ops.end());

  for (Operation *op : group.ops) {
    Value dst = getBinaryDst(op);
    appendUniqueValue(iface.producedInOrder, dst);
    iface.producedSet.insert(dst);
  }

  for (Operation *op : group.ops) {
    for (Value src : getBinarySrcs(op)) {
      if (!iface.producedSet.contains(src))
        appendUniqueValue(iface.externalInputs, src);
    }
  }

  for (Value dst : iface.producedInOrder) {
    bool usedOutside = false;
    for (Operation *user : dst.getUsers()) {
      if (!iface.groupSet.contains(user)) {
        usedOutside = true;
        break;
      }
    }
    if (usedOutside)
      appendUniqueValue(iface.externalOutputs, dst);
  }

  for (Value v : iface.externalInputs)
    appendUniqueValue(iface.callArgs, v);
  for (Value v : iface.externalOutputs)
    appendUniqueValue(iface.callArgs, v);

  return iface;
}

static std::string makeUniqueFusedName(SymbolTable &symbolTable, int64_t groupId,
                                       int &fusedCounter) {
  std::string fusedName = "__pto_fused_group_" + std::to_string(groupId) +
                          "_" + std::to_string(fusedCounter++);
  while (symbolTable.lookup(fusedName))
    fusedName += "_x";
  return fusedName;
}

static func::FuncOp createFusedFunc(ModuleOp module, Location loc,
                                    StringRef name, ArrayRef<Value> callArgs,
                                    int64_t groupId) {
  SmallVector<Type, 8> argTypes;
  for (Value v : callArgs)
    argTypes.push_back(v.getType());

  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToStart(module.getBody());
  auto fusedFunc = moduleBuilder.create<func::FuncOp>(
      loc, name, FunctionType::get(module.getContext(), argTypes, {}));
  fusedFunc.setPrivate();
  fusedFunc->setAttr("pto.fusion.group_id",
                     IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                      groupId));
  return fusedFunc;
}

static DenseMap<Value, Value> mapCallArgsToEntry(Block *entry,
                                                 ArrayRef<Value> callArgs) {
  DenseMap<Value, Value> valueMap;
  for (auto [orig, arg] : llvm::zip(callArgs, entry->getArguments()))
    valueMap[orig] = arg;
  return valueMap;
}

static LogicalResult materializeGroupOps(GroupInfo &group,
                                         ArrayRef<Value> externalOutputs,
                                         DenseMap<Value, Value> &valueMap,
                                         OpBuilder &bodyBuilder,
                                         TemplateRegistry &registry, bool debug);

struct TemplateRegistry {
  ModuleOp module;
  SymbolTable &symbolTable;
  bool debug;

  llvm::StringMap<func::FuncOp> seedByOp;
  llvm::StringMap<func::FuncOp> instanceByKey;

  TemplateRegistry(ModuleOp module, SymbolTable &symbolTable, bool debug)
      : module(module), symbolTable(symbolTable), debug(debug) {}

  LogicalResult emitFailure(Operation *anchor, const Twine &msg) {
    anchor->emitError(msg);
    return failure();
  }

  LogicalResult emitFailure(Location loc, const Twine &msg) {
    emitError(loc) << msg;
    return failure();
  }

  bool validateTemplateSignature(func::FuncOp fn, StringRef opName) {
    FunctionType fnTy = fn.getFunctionType();
    if (fnTy.getNumInputs() != 3 || fnTy.getNumResults() != 0)
      return false;
    for (Type inTy : fnTy.getInputs()) {
      auto memTy = dyn_cast<MemRefType>(inTy);
      if (!memTy)
        return false;
      if (memTy.getRank() != 2)
        return false;
      if (!memTy.getElementType().isF32())
        return false;
      for (int64_t dim : memTy.getShape()) {
        if (dim != ShapedType::kDynamic)
          return false;
      }
    }
    (void)opName;
    return true;
  }

  LogicalResult validateTemplateBody(func::FuncOp fn, StringRef opName) {
    if (fn.isExternal()) {
      fn.emitError("template function must have a body");
      return failure();
    }

    bool hasExpectedArithCore = false;
    bool hasUnexpectedArithCore = false;
    bool hasExpectedCCECCore = false;
    bool hasUnexpectedCCECCore = false;
    int64_t forCount = 0;
    bool hasNestedFor = false;
    bool bodyOk = true;
    fn.walk([&](Operation *op) {
      if (isForbiddenTemplateOp(op)) {
        op->emitError("forbidden op in OP-Lib template body");
        bodyOk = false;
        return WalkResult::interrupt();
      }
      if (!isAllowedTemplateOp(op)) {
        op->emitError("unsupported op in OP-Lib template body");
        bodyOk = false;
        return WalkResult::interrupt();
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        ++forCount;
        for (Operation &nested : forOp.getBody()->without_terminator()) {
          if (isa<scf::ForOp>(nested)) {
            hasNestedFor = true;
            break;
          }
        }
      }
      if (isTemplateArithCoreOp(op)) {
        if (isExpectedArithOpForTemplate(op, opName))
          hasExpectedArithCore = true;
        else
          hasUnexpectedArithCore = true;
      }
      if (isa<ccec::VBinOp>(op)) {
        if (isExpectedCCECOpForTemplate(op, opName))
          hasExpectedCCECCore = true;
        else
          hasUnexpectedCCECCore = true;
      }
      return WalkResult::advance();
    });
    if (!bodyOk)
      return failure();

    if (hasUnexpectedArithCore) {
      fn.emitError() << "template for op '" << opName
                     << "' contains arith floating ops that do not match the OP contract";
      return failure();
    }

    if (hasUnexpectedCCECCore) {
      fn.emitError() << "template for op '" << opName
                     << "' contains ccec ops that do not match the OP contract";
      return failure();
    }

    if (hasExpectedArithCore && hasExpectedCCECCore) {
      fn.emitError() << "template for op '" << opName
                     << "' must use either legacy arith loops or direct ccec ops, not both";
      return failure();
    }

    if (hasExpectedCCECCore) {
      if (forCount != 0) {
        fn.emitError() << "template for op '" << opName
                       << "' using ccec ops must not contain scf.for loops";
        return failure();
      }
      return success();
    }

    if (forCount < 2 || !hasNestedFor) {
      fn.emitError() << "template for op '" << opName
                     << "' must contain a 2-level nested scf.for loop";
      return failure();
    }

    if (!hasExpectedArithCore) {
      fn.emitError() << "template for op '" << opName
                     << "' must contain expected arith floating binary op";
      return failure();
    }
    return success();
  }

  LogicalResult loadFromDir(StringRef opLibDir, MLIRContext *ctx, Location loc) {
    if (opLibDir.empty())
      return emitFailure(loc, "--op-lib-dir is required when op fusion is enabled");

    llvm::SmallVector<std::string, 16> mlirFiles;
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(opLibDir, ec), end; it != end && !ec;
         it.increment(ec)) {
      if (!llvm::sys::fs::is_regular_file(it->path()))
        continue;
      if (llvm::sys::path::extension(it->path()) == ".mlir")
        mlirFiles.push_back(it->path());
    }
    if (ec)
      return emitFailure(loc, Twine("failed to iterate --op-lib-dir '") + opLibDir +
                                  "': " + ec.message());

    llvm::sort(mlirFiles);
    if (mlirFiles.empty())
      return emitFailure(loc, Twine("no .mlir files found in --op-lib-dir '") + opLibDir + "'");

    int imported = 0;
    for (const std::string &path : mlirFiles) {
      auto fileOrErr = llvm::MemoryBuffer::getFile(path);
      if (!fileOrErr)
        return emitFailure(loc, Twine("failed to read OP-Lib file '") + path +
                                    "': " + fileOrErr.getError().message());

      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      OwningOpRef<ModuleOp> libModule = parseSourceFile<ModuleOp>(sourceMgr, ctx);
      if (!libModule)
        return emitFailure(loc, Twine("failed to parse OP-Lib file '") + path + "'");

      for (func::FuncOp libFunc : libModule->getOps<func::FuncOp>()) {
        auto kindAttr = libFunc->getAttrOfType<StringAttr>(kOpLibAttrKind);
        if (!kindAttr || kindAttr.getValue() != kOpLibKindBinaryTemplate)
          continue;

        auto opAttr = libFunc->getAttrOfType<StringAttr>(kOpLibAttrOp);
        auto rankAttr = libFunc->getAttrOfType<IntegerAttr>(kOpLibAttrRank);
        auto seedDTypeAttr = libFunc->getAttrOfType<StringAttr>(kOpLibAttrSeedDType);
        if (!opAttr || !rankAttr || !seedDTypeAttr) {
          libFunc.emitError("template missing required attrs: pto.oplib.op / "
                            "pto.oplib.rank / pto.oplib.seed_dtype");
          return failure();
        }

        StringRef opName = opAttr.getValue();
        if (opName != "tmul" && opName != "tdiv" && opName != "tadd" &&
            opName != "tsub" && opName != "tmax" && opName != "tmin") {
          libFunc.emitError() << "unsupported template op attr: " << opName;
          return failure();
        }

        if (rankAttr.getInt() != 2) {
          libFunc.emitError("V1 requires pto.oplib.rank = 2");
          return failure();
        }

        if (seedDTypeAttr.getValue() != "f32") {
          libFunc.emitError("V1 requires pto.oplib.seed_dtype = \"f32\"");
          return failure();
        }

        if (seedByOp.count(opName)) {
          libFunc.emitError() << "duplicate template for op " << opName;
          return failure();
        }

        if (!validateTemplateSignature(libFunc, opName)) {
          libFunc.emitError("invalid template signature; expected "
                            "(memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()");
          return failure();
        }

        if (failed(validateTemplateBody(libFunc, opName)))
          return failure();

        OpBuilder modBuilder(module.getContext());
        modBuilder.setInsertionPointToStart(module.getBody());

        std::string sym = ("__pto_oplib_seed_" + opName).str();
        int suffix = 0;
        while (symbolTable.lookup(sym))
          sym = ("__pto_oplib_seed_" + opName + "_" + std::to_string(++suffix)).str();
        auto importedFunc =
            cast<func::FuncOp>(modBuilder.clone(*libFunc.getOperation()));
        importedFunc.setSymName(sym);
        importedFunc.setPrivate();
        seedByOp[opName] = importedFunc;
        ++imported;

        if (debug) {
          llvm::errs() << "[op-fusion] imported OP-Lib template: op=" << opName
                       << " file=" << path << " symbol=@" << sym << "\n";
        }
      }
    }

    if (imported == 0)
      return emitFailure(loc, "no valid OP-Lib templates imported from --op-lib-dir");

    return success();
  }

  FailureOr<func::FuncOp> instantiate(StringRef opName, Type targetElemTy,
                                      ArrayRef<MemRefType> concreteArgTypes,
                                      Location loc) {
    if (!isFloatDTypeSupported(targetElemTy)) {
      (void)emitFailure(loc, Twine("unsupported dtype for OP-Lib template instantiation: ") +
                                 dtypeToString(targetElemTy));
      return failure();
    }
    if (concreteArgTypes.size() != 3) {
      (void)emitFailure(loc, "expected 3 concrete memref args for binary elementwise template");
      return failure();
    }

    auto seedIt = seedByOp.find(opName);
    if (seedIt == seedByOp.end()) {
      (void)emitFailure(loc, Twine("missing OP-Lib template for op '") + opName + "'");
      return failure();
    }

    std::string key = (opName + "|" + dtypeToString(targetElemTy)).str();
    for (MemRefType memTy : concreteArgTypes) {
      key += "|";
      key += memRefTypeToString(memTy);
    }
    auto instIt = instanceByKey.find(key);
    if (instIt != instanceByKey.end())
      return instIt->second;

    func::FuncOp seed = seedIt->second;
    auto seedElemTy =
        cast<MemRefType>(seed.getFunctionType().getInput(0)).getElementType();
    if (!seedElemTy.isF32()) {
      (void)emitFailure(seed.getLoc(),
                        "V1 template seed dtype must be f32 for auto-instantiation");
      return failure();
    }

    SmallVector<Type, 4> newInputs;
    newInputs.reserve(concreteArgTypes.size());
    for (MemRefType memTy : concreteArgTypes)
      newInputs.push_back(memTy);

    std::string sym = ("__pto_oplib_inst_" + opName + "__" + dtypeToString(targetElemTy)).str();
    int suffix = 0;
    while (symbolTable.lookup(sym))
      sym = ("__pto_oplib_inst_" + opName + "__" + dtypeToString(targetElemTy) + "_" +
             std::to_string(++suffix))
                .str();

    OpBuilder modBuilder(module.getContext());
    modBuilder.setInsertionPointToStart(module.getBody());
    auto inst = modBuilder.create<func::FuncOp>(
        loc, sym, FunctionType::get(module.getContext(), newInputs, {}));
    inst.setPrivate();
    inst->setAttr("pto.oplib.instance_of", StringAttr::get(module.getContext(), opName));
    inst->setAttr("pto.oplib.instance_dtype",
                  StringAttr::get(module.getContext(), dtypeToString(targetElemTy)));
    inst->setAttr("pto.oplib.instance_seed",
                  FlatSymbolRefAttr::get(module.getContext(), seed.getSymName()));

    instanceByKey[key] = inst;

    if (debug) {
      llvm::errs() << "[op-fusion] instantiated template: op=" << opName
                   << " dtype=" << dtypeToString(targetElemTy)
                   << " sig_key=" << key << " seed=@" << seed.getSymName()
                   << " -> @" << inst.getSymName()
                   << "\n";
    }

    return inst;
  }
};

static LogicalResult materializeGroupOps(GroupInfo &group,
                                         ArrayRef<Value> externalOutputs,
                                         DenseMap<Value, Value> &valueMap,
                                         OpBuilder &bodyBuilder,
                                         TemplateRegistry &registry, bool debug) {
  auto getMapped = [&](Value v) -> Value {
    auto it = valueMap.find(v);
    if (it == valueMap.end())
      return {};
    return it->second;
  };

  auto ensureMapped = [&](Value v) -> FailureOr<Value> {
    if (Value mapped = getMapped(v))
      return mapped;
    return failure();
  };

  for (Operation *op : group.ops) {
    StringRef opName = getFusionOpName(op);
    if (opName.empty())
      return emitError(op->getLoc(), "unsupported op in fusion group"), failure();

    SmallVector<Value, 2> srcs = getBinarySrcs(op);
    if (srcs.size() != 2)
      return emitError(op->getLoc(), "expected binary op src arity = 2"), failure();

    auto src0Or = ensureMapped(srcs[0]);
    auto src1Or = ensureMapped(srcs[1]);
    if (failed(src0Or) || failed(src1Or))
      return emitError(op->getLoc(), "failed to map fusion op sources"), failure();

    Value src0 = *src0Or;
    Value src1 = *src1Or;

    auto src0Ty = dyn_cast<MemRefType>(src0.getType());
    auto src1Ty = dyn_cast<MemRefType>(src1.getType());
    if (!src0Ty || !src1Ty)
      return emitError(op->getLoc(), "fusion source must be memref"), failure();

    Type elemTy = src0Ty.getElementType();
    if (!isFloatDTypeSupported(elemTy))
      return emitError(op->getLoc(), "V1 supports only f16/f32 fusion dtypes"),
             failure();
    if (src1Ty.getElementType() != elemTy)
      return emitError(op->getLoc(), "fusion src dtype mismatch"), failure();

    Value dst = getBinaryDst(op);
    Value mappedDst = getMapped(dst);
    bool isExternalOutput = valueInList(externalOutputs, dst);
    if (!mappedDst) {
      if (isExternalOutput)
        return emitError(op->getLoc(), "external fusion output is not mapped"),
               failure();

      auto origDstTy = dyn_cast<MemRefType>(dst.getType());
      if (!origDstTy)
        return emitError(op->getLoc(), "fusion dst must be memref"), failure();

      if (origDstTy.getElementType() != elemTy)
        return emitError(op->getLoc(), "fusion dst dtype mismatch"), failure();

      SmallVector<Value, 4> dynSizes;
      for (int64_t dim = 0, e = origDstTy.getRank(); dim < e; ++dim) {
        if (!origDstTy.isDynamicDim(dim))
          continue;
        Value dimIdx = bodyBuilder.create<arith::ConstantIndexOp>(op->getLoc(), dim);
        dynSizes.push_back(
            bodyBuilder.create<memref::DimOp>(op->getLoc(), src0, dimIdx));
      }
      MemRefType allocTy = getAllocCompatibleMemRefType(origDstTy);
      Value alloc = bodyBuilder.create<memref::AllocOp>(op->getLoc(), allocTy, dynSizes);
      mappedDst = allocTy == origDstTy
                      ? alloc
                      : bodyBuilder.create<memref::CastOp>(op->getLoc(), origDstTy, alloc);
      valueMap[dst] = mappedDst;
    }

    auto dstTy = dyn_cast<MemRefType>(mappedDst.getType());
    if (!dstTy)
      return emitError(op->getLoc(), "fusion dst must be memref"), failure();
    if (dstTy.getElementType() != elemTy)
      return emitError(op->getLoc(), "fusion mapped dst dtype mismatch"), failure();

    SmallVector<MemRefType, 3> concreteArgTypes{src0Ty, src1Ty, dstTy};
    FailureOr<func::FuncOp> instanceOr =
        registry.instantiate(opName, elemTy, concreteArgTypes, op->getLoc());
    if (failed(instanceOr))
      return failure();
    func::FuncOp instance = *instanceOr;

    auto instTy = instance.getFunctionType();
    auto instSrc0Ty = cast<MemRefType>(instTy.getInput(0));
    auto instSrc1Ty = cast<MemRefType>(instTy.getInput(1));
    auto instDstTy = cast<MemRefType>(instTy.getInput(2));

    Value callSrc0 =
        castToMemRefTypeIfNeeded(bodyBuilder, op->getLoc(), src0, instSrc0Ty);
    Value callSrc1 =
        castToMemRefTypeIfNeeded(bodyBuilder, op->getLoc(), src1, instSrc1Ty);
    Value callDst =
        castToMemRefTypeIfNeeded(bodyBuilder, op->getLoc(), mappedDst, instDstTy);
    if (!callSrc0 || !callSrc1 || !callDst)
      return emitError(op->getLoc(),
                       "failed to adapt operands to OP-Lib instance signature"),
             failure();

    if (debug) {
      llvm::errs() << "[op-fusion] materialize op=" << opName
                   << " request_dtype=" << dtypeToString(elemTy)
                   << " instance=@" << instance.getSymName() << "\n";
    }

    bodyBuilder.create<func::CallOp>(op->getLoc(), instance,
                                     ValueRange{callSrc0, callSrc1, callDst});
  }

  return success();
}

struct PTOMaterializeFusionGroupsFromOpLibPass
    : public pto::impl::PTOMaterializeFusionGroupsFromOpLibBase<
          PTOMaterializeFusionGroupsFromOpLibPass> {
  using pto::impl::PTOMaterializeFusionGroupsFromOpLibBase<
      PTOMaterializeFusionGroupsFromOpLibPass>::PTOMaterializeFusionGroupsFromOpLibBase;

  void collectGroupsInRegion(Region &region,
                             SmallVectorImpl<GroupInfo> &groups) {
    for (Block &block : region.getBlocks()) {
      GroupInfo current;
      int64_t expectedOrder = 0;

      auto flush = [&]() {
        if (current.ops.size() >= 2)
          groups.push_back(current);
        current = GroupInfo();
        expectedOrder = 0;
      };

      for (Operation &op : block.getOperations()) {
        Operation *curOp = &op;

        if (hasFusionGroupAttrs(curOp) && isSupportedFusionOp(curOp)) {
          auto gidAttr = curOp->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr);
          auto orderAttr = curOp->getAttrOfType<IntegerAttr>(kFusionOrderAttr);

          if (!gidAttr || !orderAttr) {
            flush();
          } else {
            int64_t gid = gidAttr.getInt();
            int64_t order = orderAttr.getInt();
            if (current.ops.empty()) {
              current.groupId = gid;
              current.ops.push_back(curOp);
              expectedOrder = order + 1;
            } else if (current.groupId == gid && order == expectedOrder) {
              current.ops.push_back(curOp);
              ++expectedOrder;
            } else {
              flush();
              current.groupId = gid;
              current.ops.push_back(curOp);
              expectedOrder = order + 1;
            }
          }
        } else {
          flush();
        }

        for (Region &nested : curOp->getRegions())
          collectGroupsInRegion(nested, groups);
      }

      flush();
    }
  }

  LogicalResult materializeOneGroup(ModuleOp module, SymbolTable &symbolTable,
                                    TemplateRegistry &registry, GroupInfo &group,
                                    int &fusedCounter) {
    if (group.ops.empty())
      return success();

    Operation *firstOp = group.ops.front();
    Location loc = firstOp->getLoc();

    GroupInterface iface = buildGroupInterface(group);
    if (iface.callArgs.empty())
      return emitError(loc, "fusion group has no external interface values"), failure();

    std::string fusedName =
        makeUniqueFusedName(symbolTable, group.groupId, fusedCounter);
    auto fusedFunc = createFusedFunc(module, loc, fusedName, iface.callArgs,
                                     group.groupId);

    Block *entry = fusedFunc.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);

    DenseMap<Value, Value> valueMap =
        mapCallArgsToEntry(entry, iface.callArgs);
    if (failed(materializeGroupOps(group, iface.externalOutputs, valueMap,
                                   bodyBuilder, registry, debug)))
      return failure();

    bodyBuilder.create<func::ReturnOp>(loc);

    OpBuilder callerBuilder(firstOp);
    callerBuilder.create<func::CallOp>(loc, fusedFunc, iface.callArgs);

    for (Operation *op : llvm::reverse(group.ops))
      op->erase();

    for (Value dst : iface.producedInOrder) {
      if (!valueInList(iface.externalOutputs, dst))
        eraseDeadValueDefChain(dst);
    }

    if (debug) {
      llvm::errs() << "[op-fusion] materialized group_id=" << group.groupId
                   << " into @" << fusedFunc.getSymName() << "\n";
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTable symbolTable(module);

    TemplateRegistry registry(module, symbolTable, debug);
    if (failed(registry.loadFromDir(opLibDir, ctx, module.getLoc()))) {
      signalPassFailure();
      return;
    }

    int fusedCounter = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_fused_group_"))
        continue;

      SmallVector<GroupInfo, 8> groups;
      collectGroupsInRegion(func.getRegion(), groups);
      if (groups.empty())
        continue;

      if (debug)
        llvm::errs() << "[op-fusion] found " << groups.size() << " group(s) in @"
                     << func.getSymName() << "\n";

      for (GroupInfo &group : groups) {
        if (failed(
                materializeOneGroup(module, symbolTable, registry, group, fusedCounter))) {
          signalPassFailure();
          return;
        }
      }
    }

  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOMaterializeFusionGroupsFromOpLibPass(
    const PTOMaterializeFusionGroupsFromOpLibOptions &options) {
  return std::make_unique<PTOMaterializeFusionGroupsFromOpLibPass>(options);
}
