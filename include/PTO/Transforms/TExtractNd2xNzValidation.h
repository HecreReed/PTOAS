// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Design doc: docs/designs/textract-nd-to-2xnz-design.md (merged in PR #1289).
// Driver-level validation helpers for the ND-to-2xNZ dual-output TEXTRACT form.
// These are plain driver functions, intentionally NOT MLIR passes:
//   - validateTExtractNd2xNzInputProvenance: runs after generic verification,
//     before any planning/sync pass manager. Rejects runtime-bound tile
//     provenance (DeclareTileOp/TAssignOp/pop-derived tiles and their views).
//   - validateTExtractNd2xNzPostPlanningSafety: runs after
//     PTOResolveBufferSelect and before PTOInlineBackendHelpersPass. Rejects
//     alias generic TSTORE of partial-valid destinations within a closed
//     direct-call component, with address-space-aware range comparison.

#ifndef PTO_TRANSFORMS_TEXTRACT_ND2XNZ_VALIDATION_H
#define PTO_TRANSFORMS_TEXTRACT_ND2XNZ_VALIDATION_H

#include "mlir/IR/Operation.h"

namespace mlir {
class ModuleOp;
namespace pto {

mlir::LogicalResult
validateTExtractNd2xNzInputProvenance(mlir::Operation *module);

mlir::LogicalResult
validateTExtractNd2xNzPostPlanningSafety(mlir::Operation *module);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_TEXTRACT_ND2XNZ_VALIDATION_H
