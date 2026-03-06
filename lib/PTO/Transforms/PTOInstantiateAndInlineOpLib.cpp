#include "PTO/IR/CCEC.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINSTANTIATEANDINLINEOPLIB
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kOpLibAttrOp = "pto.oplib.op";
static constexpr llvm::StringLiteral kOpLibAttrKind = "pto.oplib.kind";
static constexpr llvm::StringLiteral kOpLibAttrInstanceOf = "pto.oplib.instance_of";
static constexpr llvm::StringLiteral kOpLibAttrInstanceDType = "pto.oplib.instance_dtype";
static constexpr llvm::StringLiteral kOpLibKindBinaryTemplate =
    "binary_elementwise_template";

static bool isTemplateFunc(func::FuncOp fn) {
  auto kind = fn->getAttrOfType<StringAttr>(kOpLibAttrKind);
  return kind && kind.getValue() == kOpLibKindBinaryTemplate;
}

static bool isInstanceFunc(func::FuncOp fn) {
  return fn->hasAttr(kOpLibAttrInstanceOf);
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

static std::string typeListKey(ArrayRef<Type> tys) {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (Type ty : tys) {
    ty.print(os);
    os << ";";
  }
  return os.str();
}

static Value maybeUnwrapCast(Value operand, Type expectedType) {
  auto cast = operand.getDefiningOp<memref::CastOp>();
  if (!cast)
    return operand;
  if (cast.getResult().getType() != expectedType)
    return operand;

  auto srcTy = dyn_cast<MemRefType>(cast.getSource().getType());
  auto expectedMemTy = dyn_cast<MemRefType>(expectedType);
  if (!srcTy || !expectedMemTy)
    return operand;
  if (srcTy.getRank() != expectedMemTy.getRank())
    return operand;
  if (srcTy.getElementType() != expectedMemTy.getElementType())
    return operand;
  if (srcTy.getMemorySpace() != expectedMemTy.getMemorySpace())
    return operand;
  return cast.getSource();
}

static void eraseDeadCasts(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<memref::CastOp, 8> dead;
    func.walk([&](memref::CastOp cast) {
      if (cast->use_empty())
        dead.push_back(cast);
    });
    if (dead.empty())
      break;
    for (memref::CastOp cast : llvm::reverse(dead))
      cast.erase();
    changed = true;
  }
}

static LogicalResult inlineCall(func::CallOp call, func::FuncOp callee) {
  if (call.getNumResults() != 0)
    return call.emitOpError("OP-Lib inline expects call without results");
  if (callee.isExternal())
    return call.emitOpError("callee must have a body before inlining");

  Block &entry = callee.getBody().front();
  if (entry.getNumArguments() != call.getNumOperands())
    return call.emitOpError("callee argument count mismatch during inlining");

  OpBuilder builder(call);
  IRMapping mapping;
  for (auto [arg, operand] : llvm::zip(entry.getArguments(), call.getOperands()))
    mapping.map(arg, operand);

  for (Operation &op : entry.without_terminator()) {
    Operation *newOp = builder.clone(op, mapping);
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), newOp->getResults()))
      mapping.map(oldRes, newRes);
  }

  call.erase();
  return success();
}

static FailureOr<func::FuncOp>
createConcreteInstanceFromSeed(ModuleOp module, func::FuncOp seed,
                               ArrayRef<Type> concreteInputs, StringRef opName,
                               bool debug) {
  if (seed.isExternal())
    return failure();

  MLIRContext *ctx = module.getContext();
  auto seedTy = seed.getFunctionType();
  if (seedTy.getNumInputs() != concreteInputs.size())
    return failure();

  std::string sym = ("__pto_oplib_inst_" + opName + "__" +
                     dtypeToString(cast<MemRefType>(concreteInputs.front()).getElementType()))
                        .str();
  int suffix = 0;
  while (module.lookupSymbol<func::FuncOp>(sym))
    sym = ("__pto_oplib_inst_" + opName + "__" +
           dtypeToString(cast<MemRefType>(concreteInputs.front()).getElementType()) + "_" +
           std::to_string(++suffix))
              .str();

  OpBuilder modBuilder(ctx);
  modBuilder.setInsertionPointToStart(module.getBody());
  auto inst = modBuilder.create<func::FuncOp>(
      seed.getLoc(), sym, FunctionType::get(ctx, concreteInputs, {}));
  inst.setPrivate();
  inst->setAttr(kOpLibAttrInstanceOf, StringAttr::get(ctx, opName));
  inst->setAttr(kOpLibAttrInstanceDType,
                StringAttr::get(
                    ctx, dtypeToString(cast<MemRefType>(concreteInputs.front()).getElementType())));

  Block *entry = inst.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);

  Block &seedEntry = seed.getBody().front();
  IRMapping mapping;
  for (unsigned i = 0; i < seedTy.getNumInputs(); ++i) {
    Value arg = entry->getArgument(i);
    Type dstTy = seedTy.getInput(i);
    Value adapted = arg;
    if (arg.getType() != dstTy) {
      auto castTy = dyn_cast<MemRefType>(dstTy);
      if (!castTy)
        return failure();
      adapted = bodyBuilder.create<memref::CastOp>(seed.getLoc(), castTy, arg);
    }
    mapping.map(seedEntry.getArgument(i), adapted);
  }

  for (Operation &op : seedEntry.without_terminator()) {
    Operation *newOp = bodyBuilder.clone(op, mapping);
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), newOp->getResults()))
      mapping.map(oldRes, newRes);
  }
  bodyBuilder.create<func::ReturnOp>(seed.getLoc());

  if (debug) {
    llvm::errs() << "[op-fusion] instantiate-oplib: created @" << inst.getSymName()
                 << " from @" << seed.getSymName() << "\n";
  }

  return inst;
}

static LogicalResult materializeExternalInstanceBody(func::FuncOp instance,
                                                     func::FuncOp seed, bool debug) {
  if (!instance.isExternal())
    return success();
  if (seed.isExternal())
    return instance.emitOpError("seed template must have body");

  auto instTy = instance.getFunctionType();
  auto seedTy = seed.getFunctionType();
  if (instTy.getNumInputs() != seedTy.getNumInputs())
    return instance.emitOpError("instance/seed signature arity mismatch");

  Block *entry = instance.addEntryBlock();
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
  Block &seedEntry = seed.getBody().front();
  IRMapping mapping;

  for (unsigned i = 0; i < instTy.getNumInputs(); ++i) {
    Value arg = entry->getArgument(i);
    Type dstTy = seedTy.getInput(i);
    Value adapted = arg;
    if (arg.getType() != dstTy) {
      auto castTy = dyn_cast<MemRefType>(dstTy);
      if (!castTy)
        return instance.emitOpError("seed input must be memref");
      adapted = bodyBuilder.create<memref::CastOp>(instance.getLoc(), castTy, arg);
    }
    mapping.map(seedEntry.getArgument(i), adapted);
  }

  for (Operation &op : seedEntry.without_terminator()) {
    Operation *newOp = bodyBuilder.clone(op, mapping);
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), newOp->getResults()))
      mapping.map(oldRes, newRes);
  }
  bodyBuilder.create<func::ReturnOp>(instance.getLoc());

  if (debug) {
    llvm::errs() << "[op-fusion] instantiate-oplib: materialized external instance @"
                 << instance.getSymName() << " from @" << seed.getSymName() << "\n";
  }
  return success();
}

struct PTOInstantiateAndInlineOpLibPass
    : public pto::impl::PTOInstantiateAndInlineOpLibBase<
          PTOInstantiateAndInlineOpLibPass> {
  using pto::impl::PTOInstantiateAndInlineOpLibBase<
      PTOInstantiateAndInlineOpLibPass>::PTOInstantiateAndInlineOpLibBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::StringMap<func::FuncOp> seedByOp;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!isTemplateFunc(func))
        continue;
      auto opAttr = func->getAttrOfType<StringAttr>(kOpLibAttrOp);
      if (!opAttr)
        continue;
      if (!seedByOp.count(opAttr.getValue()))
        seedByOp[opAttr.getValue()] = func;
    }

    llvm::StringMap<func::FuncOp> instanceByKey;
    int inlinedCalls = 0;
    int touchedFuncs = 0;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func.getSymName().starts_with("__pto_fused_group_"))
        continue;
      if (func.empty())
        continue;

      SmallVector<func::CallOp, 8> calls;
      func.walk([&](func::CallOp call) { calls.push_back(call); });

      bool changedThisFunc = false;
      for (func::CallOp oldCall : calls) {
        if (!oldCall || !oldCall->getBlock())
          continue;
        auto calleeAttr = oldCall.getCalleeAttr();
        if (!calleeAttr)
          continue;

        func::FuncOp callee = module.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
        if (!callee)
          continue;

        func::CallOp call = oldCall;

        if (isTemplateFunc(callee)) {
          auto opAttr = callee->getAttrOfType<StringAttr>(kOpLibAttrOp);
          if (!opAttr) {
            callee.emitError("template missing pto.oplib.op");
            signalPassFailure();
            return;
          }
          StringRef opName = opAttr.getValue();

          SmallVector<Value, 4> concreteOperands;
          concreteOperands.reserve(call.getNumOperands());
          for (auto [operand, expectedTy] :
               llvm::zip(call.getOperands(), callee.getFunctionType().getInputs()))
            concreteOperands.push_back(maybeUnwrapCast(operand, expectedTy));

          SmallVector<Type, 4> concreteTypes;
          concreteTypes.reserve(concreteOperands.size());
          for (Value v : concreteOperands)
            concreteTypes.push_back(v.getType());

          std::string key = (opName + "|" + typeListKey(concreteTypes)).str();
          func::FuncOp concreteInstance;
          auto it = instanceByKey.find(key);
          if (it != instanceByKey.end()) {
            concreteInstance = it->second;
          } else {
            auto createdOr =
                createConcreteInstanceFromSeed(module, callee, concreteTypes, opName, debug);
            if (failed(createdOr)) {
              call.emitError("failed to create concrete OP-Lib instance");
              signalPassFailure();
              return;
            }
            concreteInstance = *createdOr;
            instanceByKey[key] = concreteInstance;
          }

          OpBuilder builder(call);
          auto newCall =
              builder.create<func::CallOp>(call.getLoc(), concreteInstance, concreteOperands);
          call.erase();
          call = newCall;
          callee = concreteInstance;
        } else if (isInstanceFunc(callee)) {
          if (callee.isExternal()) {
            auto opAttr = callee->getAttrOfType<StringAttr>(kOpLibAttrInstanceOf);
            if (!opAttr) {
              callee.emitError("instance missing pto.oplib.instance_of");
              signalPassFailure();
              return;
            }
            auto seedIt = seedByOp.find(opAttr.getValue());
            if (seedIt == seedByOp.end()) {
              callee.emitError()
                  << "missing OP-Lib seed for instance_of=" << opAttr.getValue();
              signalPassFailure();
              return;
            }
            if (failed(materializeExternalInstanceBody(callee, seedIt->second, debug))) {
              signalPassFailure();
              return;
            }
          }
        } else {
          continue;
        }

        if (failed(inlineCall(call, callee))) {
          signalPassFailure();
          return;
        }
        ++inlinedCalls;
        changedThisFunc = true;
        if (debug) {
          llvm::errs() << "[op-fusion] inline-oplib: inlined @" << callee.getSymName()
                       << " into @" << func.getSymName() << "\n";
        }
      }

      if (changedThisFunc) {
        eraseDeadCasts(func);
        ++touchedFuncs;
      }
    }

    if (debug) {
      llvm::errs() << "[op-fusion] instantiate+inline touched " << touchedFuncs
                   << " fused function(s), inlined " << inlinedCalls << " call(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOInstantiateAndInlineOpLibPass(
    const PTOInstantiateAndInlineOpLibOptions &options) {
  return std::make_unique<PTOInstantiateAndInlineOpLibPass>(options);
}
