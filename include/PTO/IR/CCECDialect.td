//===- CCECDialect.td - CCEC dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCEC_IR_CCECDIALECT
#define MLIR_DIALECT_CCEC_IR_CCECDIALECT

include "mlir/IR/OpBase.td"

def CCEC_Dialect : Dialect {
  let summary = "Minimal CCEC IR dialect for element-wise vector semantics";
  let description = [{
    CCEC dialect models target-facing vector semantics for PTOAS without tying
    templates to concrete TADD/TMUL wrapper APIs. The initial subset only
    covers element-wise binary vector ops.
  }];

  let name = "ccec";
  let cppNamespace = "::mlir::ccec";
}

#endif // MLIR_DIALECT_CCEC_IR_CCECDIALECT
