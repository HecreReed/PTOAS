// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PTOTOEMITCINTERNAL_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PTOTOEMITCINTERNAL_H

#pragma GCC diagnostic ignored "-Woverloaded-virtual"
// GCC warns on MLIR OpConversionPattern helper overloads hiding RewritePattern::rewrite.

#include "PTO/IR/PTO.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>
#include <string>

namespace mlir::pto {

Value peelUnrealized(Value v);

Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                              Location loc, Type type,
                              llvm::StringRef literal);

Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter, Location loc,
                           Type type, int64_t value);

Value emitCCast(ConversionPatternRewriter &rewriter, Location loc, Type dstType,
                Value src);

Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                         Location loc, Value v,
                                         unsigned bitWidth);

std::string getEmitCScalarTypeToken(Type elemTy);

pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr);

int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                              pto::BLayout blayout, int dimIdx);

std::optional<std::string> getEmitCTileTypeString(pto::TileBufType type);

bool isSetFFTsPointerLikeType(Type ty);

bool isEmitCGlobalTensorLikeType(Type ty);

std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    llvm::StringRef layoutEnum = "pto::Layout::ND");

std::string getElemTypeStringForGT(Type elemTy);

SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape);

void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> strides,
                                     SmallVectorImpl<int64_t> &shape5D,
                                     SmallVectorImpl<int64_t> &stride5D);

std::string joinIntTemplateParams(ArrayRef<int64_t> values);

Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                  Location loc, Value basePtr,
                                  MemRefType mrTy, Operation *anchor);

FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                 PTOArch targetArch);

Value castToGMBytePointer(ConversionPatternRewriter &rewriter, Location loc,
                          Value value);

Value materializeTensorViewDataPointer(ConversionPatternRewriter &rewriter,
                                       Location loc, Value value,
                                       Type originalType);

Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                  Location loc, Value addr,
                                  pto::AddressSpace as,
                                  llvm::StringRef elemTok);

Value applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
                              Location loc, Value basePtr, int64_t offset);

FailureOr<Value> buildAsyncScratchTileValue(ConversionPatternRewriter &rewriter,
                                            Location loc, Value originalScratch,
                                            Value emittedScratch);

bool needsA5NoSplitVectorGuard(Operation *op);

Value materializeTileDataValue(ConversionPatternRewriter &rewriter,
                               Location loc, Value tile,
                               pto::AddressSpace as,
                               llvm::StringRef elemTypeToken);

void populatePTOToEmitCArithPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter,
                                     MLIRContext *ctx);

void populatePTOToEmitCTilePatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx);

void populatePTOToEmitCTileExtraPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx);

void populatePTOToEmitCTileMaterializationPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx);

void populatePTOToEmitCSyncPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx, PTOArch targetArch);

void populatePTOToEmitCCommPatterns(RewritePatternSet &patterns,
                                    TypeConverter &typeConverter,
                                    MLIRContext *ctx, PTOArch targetArch);

void populatePTOToEmitCKernelOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx);

LogicalResult runPTOToEmitCSCFPreLowering(ModuleOp mop, MLIRContext *ctx);

void populatePTOToEmitCControlFlowPatterns(RewritePatternSet &patterns,
                                           TypeConverter &typeConverter,
                                           MLIRContext *ctx);

void populatePTOToEmitCSimpleOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx);

void populatePTOToEmitCRuntimeOpPatterns(RewritePatternSet &patterns,
                                         TypeConverter &typeConverter,
                                         MLIRContext *ctx, PTOArch targetArch);

void populatePTOToEmitCMemoryOpPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PTOTOEMITCINTERNAL_H
