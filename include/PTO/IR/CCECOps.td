//===- CCECOps.td - CCEC dialect ops definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCEC_IR_CCECOPS
#define MLIR_DIALECT_CCEC_IR_CCECOPS

include "PTO/IR/CCECDialect.td"
include "mlir/IR/OpBase.td"

class CCEC_Op<string mnemonic, list<Trait> traits = []>
    : Op<CCEC_Dialect, mnemonic, traits>;

def CCEC_VBinOp : CCEC_Op<"vbin"> {
  let summary = "Element-wise binary vector op";
  let description = [{
    `ccec.vbin` expresses a target-facing binary vector semantic operation over
    two source tiles/buffers and one destination tile/buffer.

    The current implementation is intentionally small:
      * rank-2 memrefs only
      * floating element types (`f16` / `f32`)
      * supported kinds: add / sub / mul / div / max / min
  }];

  let arguments = (ins
    AnyMemRef:$src0,
    AnyMemRef:$src1,
    AnyMemRef:$dst,
    StrAttr:$kind
  );

  let assemblyFormat = [{
    `kind` `=` $kind attr-dict
    `ins` `(` $src0 `,` $src1 `:` type($src0) `,` type($src1) `)`
    `outs` `(` $dst `:` type($dst) `)`
  }];

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_CCEC_IR_CCECOPS
