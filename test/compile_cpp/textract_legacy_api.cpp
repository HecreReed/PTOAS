// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"

#include "mlir/IR/Builders.h"
#include "llvm/Config/llvm-config.h"

#include <type_traits>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

static_assert(std::is_same_v<decltype(std::declval<TExtractOp>().getIndexRow()),
                             TypedValue<IndexType>>);
static_assert(std::is_same_v<decltype(std::declval<TExtractOp>().getIndexCol()),
                             TypedValue<IndexType>>);
static_assert(std::is_same_v<decltype(std::declval<TExtractOp>().getDst()),
                             TypedValue<Type>>);
static_assert(
    std::is_same_v<decltype(std::declval<TExtractOp>().getIndexRowMutable()),
                   OpOperand &>);
static_assert(
    std::is_same_v<decltype(std::declval<TExtractOp>().getIndexColMutable()),
                   OpOperand &>);
static_assert(
    std::is_same_v<decltype(std::declval<TExtractOp>().getDstMutable()),
                   OpOperand &>);

[[maybe_unused]] static void compileLegacyBuilders(
    OpBuilder &builder, Location loc, TypeRange resultTypes, Value src,
    Value indexRow, Value indexCol, Value dst, Value fp, Value preQuantScalar,
    AccToVecModeAttr accToVecMode, ReluPreModeAttr reluPreModeAttr) {
  (void)builder.create<TExtractOp>(loc, src, indexRow, indexCol, dst, fp,
                                   preQuantScalar, accToVecMode,
                                   reluPreModeAttr);
  (void)builder.create<TExtractOp>(loc, resultTypes, src, indexRow, indexCol,
                                   dst, fp, preQuantScalar, accToVecMode,
                                   reluPreModeAttr);
  (void)builder.create<TExtractOp>(loc, src, indexRow, indexCol, dst, fp,
                                   preQuantScalar, accToVecMode);
  (void)builder.create<TExtractOp>(loc, resultTypes, src, indexRow, indexCol,
                                   dst, fp, preQuantScalar, accToVecMode);

#if LLVM_VERSION_MAJOR >= 21
  ImplicitLocOpBuilder implicitBuilder(loc, builder);
  (void)TExtractOp::create(builder, loc, src, indexRow, indexCol, dst, fp,
                           preQuantScalar, accToVecMode, reluPreModeAttr);
  (void)TExtractOp::create(builder, loc, resultTypes, src, indexRow, indexCol,
                           dst, fp, preQuantScalar, accToVecMode,
                           reluPreModeAttr);
  (void)TExtractOp::create(implicitBuilder, src, indexRow, indexCol, dst, fp,
                           preQuantScalar, accToVecMode, reluPreModeAttr);
  (void)TExtractOp::create(implicitBuilder, resultTypes, src, indexRow,
                           indexCol, dst, fp, preQuantScalar, accToVecMode,
                           reluPreModeAttr);
  (void)TExtractOp::create(builder, loc, src, indexRow, indexCol, dst, fp,
                           preQuantScalar, accToVecMode);
  (void)TExtractOp::create(builder, loc, resultTypes, src, indexRow, indexCol,
                           dst, fp, preQuantScalar, accToVecMode);
  (void)TExtractOp::create(implicitBuilder, src, indexRow, indexCol, dst, fp,
                           preQuantScalar, accToVecMode);
  (void)TExtractOp::create(implicitBuilder, resultTypes, src, indexRow,
                           indexCol, dst, fp, preQuantScalar, accToVecMode);
#endif
}

int main() { return 0; }
