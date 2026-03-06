//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

def PTOConvertToDPS : Pass<"pto-convert-to-dps", "func::FuncOp"> {
  let summary = "Convert PTO value-semantic ops to DPS ops";
  let description = [{
    Rewrites High-Level PTO ops (load/store/matmul) to their DPS variants 
    by allocating empty tensors as destinations.
  }];
 
  let constructor = "mlir::pto::createPTOConvertToDPSPass()";
 
  let dependentDialects = [
    "tensor::TensorDialect",
    "pto::PTODialect"
  ];
}

// Define PTOInsertSync Pass
def PTOInsertSync : Pass<"pto-insert-sync", "func::FuncOp"> {
  let summary = "Insert synchronization flags for PTO dialect";
  let description = [{
    Analyzes data dependencies between Cube, Vector and MTE pipelines
    and inserts explicit synchronization instructions.
  }];
 
  // Declare the constructor function name
  let constructor = "mlir::pto::createPTOInsertSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect" 
  ];
}

def ConvertToPTOOp : Pass<"convert-to-pto-op"> {
  let summary = "Convert Ops from other dialects to PTO Ops";
  let constructor = "mlir::pto::createConvertToPTOOpPass()";
  let dependentDialects = ["arith::ArithDialect", "pto::PTODialect",
                           "tensor::TensorDialect",
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "func::FuncOp"> {
  let summary = "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules and attach the optional `layout` attribute.
    Dynamic stride/shape are left unset.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect"];
}


def InferPTOMemScope : Pass<"pto-infer-mem-scope"> {
  let summary = "Infer memory scope for PTO Ops";
  let constructor = "mlir::pto::createInferPTOMemScopePass()";
  let dependentDialects = ["pto::PTODialect", "memref::MemRefDialect", ];
}

def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
  let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";
  let dependentDialects = ["pto::PTODialect", ];
  let options = [
    Option<"memMode", "mem-plan-mode", "pto::MemPlanMode",
           "pto::MemPlanMode::LOCAL_MEM_PLAN",
           "plan mem mode (default is LOCAL_MEM_PLAN)",
           [{::llvm::cl::values(
               clEnumValN(pto::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for memref.alloc"),
               clEnumValN(
                   pto::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for memref_ext.alloc_workspace"))}]>,
    Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
           /*default=*/"false",
           "Enable global workspace reuse ,default : false">,
    Option<"enablePrintMemoryAllocatedSize", "enable-print-memory-allocated-size", "bool",
           /*default=*/"false",
           "Enable print memory allocated size, default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
  ];
}
def PTOLoweringSyncToPipe : Pass<"pto-lowering-sync-to-pipe", "func::FuncOp"> {
  let summary = "Lower high-level sync ops to low-level pipe ops";
  let description = [{
    Converts high-level `pto.record_event` and `pto.wait_event` operations
    into low-level `pto.set_flag` and `pto.wait_flag` operations based on
    hardware pipe mapping logic.
  }];

  let constructor = "mlir::pto::createLoweringSyncToPipePass()";
  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOCreateFusionGroups : Pass<"pto-create-fusion-groups", "func::FuncOp"> {
  let summary = "Create fusion groups for fusible PTO binary elementwise chains";
  let description = [{
    Finds contiguous fusible chains and annotates ops with:
      - pto.fusion.group_id
      - pto.fusion.order
  }];
  let constructor = "mlir::pto::createPTOCreateFusionGroupsPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOMaterializeFusionGroupsFromOpLib : Pass<"pto-materialize-fusion-groups-from-oplib", "ModuleOp"> {
  let summary = "Materialize fusion groups using external OP-Lib templates";
  let description = [{
    Reads OP-Lib template functions from --op-lib-dir, validates templates,
    and rewrites fusion groups into synthesized fused functions that call
    OP-Lib seed/instance symbols. Actual instance body generation and inlining
    are handled by a dedicated follow-up pass.
  }];
  let constructor = "mlir::pto::createPTOMaterializeFusionGroupsFromOpLibPass()";
  let dependentDialects = [
    "mlir::ccec::CCECDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::pto::PTODialect"
  ];
  let options = [
    Option<"opLibDir", "op-lib-dir", "std::string",
           /*default=*/"\"\"",
           "Directory containing OP-Lib template .mlir files">,
    Option<"debug", "debug", "bool",
           /*default=*/"false",
           "Enable verbose debug logging for OP-Lib materialization">
  ];
}

def PTOInstantiateAndInlineOpLib : Pass<"pto-instantiate-inline-oplib", "ModuleOp"> {
  let summary = "Instantiate OP-Lib instances and inline OP-Lib calls in fused groups";
  let description = [{
    Resolves OP-Lib instance declarations generated by materialization, materializes
    instance bodies from OP-Lib seeds, then inlines OP-Lib calls into synthesized
    fused functions.
  }];
  let constructor = "mlir::pto::createPTOInstantiateAndInlineOpLibPass()";
  let dependentDialects = [
    "mlir::ccec::CCECDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"debug", "debug", "bool",
           /*default=*/"false",
           "Enable verbose debug logging for OP-Lib instantiation/inlining">
  ];
}

def PTOLowerCCECToLoops : Pass<"pto-lower-ccec-to-loops", "ModuleOp"> {
  let summary = "Lower minimal CCEC vector ops to memref/scf/arith loops";
  let description = [{
    Rewrites the initial CCEC element-wise vector subset into explicit nested
    loops so the existing PTOAS optimization and codegen pipeline can continue
    to operate on memref/scf/arith IR.
  }];
  let constructor = "mlir::pto::createPTOLowerCCECToLoopsPass()";
  let dependentDialects = [
    "mlir::ccec::CCECDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def PTOLowLevelLoopFusion : Pass<"pto-low-level-loop-fusion", "ModuleOp"> {
  let summary = "Fuse adjacent low-level loop nests in synthesized fused functions";
  let description = [{
    Conservatively fuses adjacent 2-level scf.for nests with identical bounds.
    If fusion is not legal, keeps the original loops unchanged.
  }];
  let constructor = "mlir::pto::createPTOLowLevelLoopFusionPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [
    Option<"debug", "debug", "bool",
           /*default=*/"false",
           "Enable verbose debug logging for low-level loop fusion">
  ];
}

#endif // MLIR_DIALECT_PTO_PASSES
