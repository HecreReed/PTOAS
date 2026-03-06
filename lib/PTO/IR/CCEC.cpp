//===- CCEC.cpp - CCEC Dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/CCEC.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::ccec;

#include "PTO/IR/CCECDialect.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/CCECOps.cpp.inc"

bool mlir::ccec::isSupportedVBinKind(StringRef kind) {
  return kind == "add" || kind == "sub" || kind == "mul" || kind == "div" ||
         kind == "max" || kind == "min";
}

StringRef mlir::ccec::getExpectedVBinKindForOpLibOp(StringRef opName) {
  if (opName == "tadd")
    return "add";
  if (opName == "tsub")
    return "sub";
  if (opName == "tmul")
    return "mul";
  if (opName == "tdiv")
    return "div";
  if (opName == "tmax")
    return "max";
  if (opName == "tmin")
    return "min";
  return {};
}

void CCECDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PTO/IR/CCECOps.cpp.inc"
      >();
}

LogicalResult VBinOp::verify() {
  auto src0Ty = dyn_cast<MemRefType>(getSrc0().getType());
  auto src1Ty = dyn_cast<MemRefType>(getSrc1().getType());
  auto dstTy = dyn_cast<MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError("expects memref operands");

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("currently only supports rank-2 memrefs");

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError("expects src0/src1/dst to have the same element type");

  Type elemTy = dstTy.getElementType();
  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError("currently only supports f16/f32 element types");

  if (!isSupportedVBinKind(getKind()))
    return emitOpError() << "unsupported kind '" << getKind()
                         << "', expected one of add/sub/mul/div/max/min";
  return success();
}
