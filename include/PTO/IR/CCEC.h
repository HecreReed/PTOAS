//===- CCEC.h - CCEC Dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCEC_IR_CCEC_H_
#define MLIR_DIALECT_CCEC_IR_CCEC_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/StringRef.h"

#include "PTO/IR/CCECDialect.h"

#define GET_OP_CLASSES
#include "PTO/IR/CCECOps.h.inc"

namespace mlir {
namespace ccec {

bool isSupportedVBinKind(StringRef kind);
StringRef getExpectedVBinKindForOpLibOp(StringRef opName);

} // namespace ccec
} // namespace mlir

#endif // MLIR_DIALECT_CCEC_IR_CCEC_H_
