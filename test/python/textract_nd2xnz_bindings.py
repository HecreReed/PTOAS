#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Smoke tests for the ND-to-2xNZ dual-output pto.textract Python surface.

Design doc sections 4.4/10: the legacy TExtractOp constructor and the
pto.textract free function keep their single-output signature;
TExtractOp.build_nd_to_2xnz builds the dual-output form on the same op
class. .indexRow/.indexCol/.dst map back to the range slices.
"""

from ptoas.mlir.ir import (
    Context,
    F16Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    Location,
    Module,
)
from ptoas.mlir.dialects import arith, pto


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def tile_type(ctx, shape, valid_shape, blayout, slayout):
    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
    config = pto.TileBufConfigAttr.get(
        pto.BLayoutAttr.get(blayout, ctx),
        pto.SLayoutAttr.get(slayout, ctx),
        512,
        pto.PadValueAttr.get(pto.PadValue.Null, ctx),
        ctx,
    )
    return pto.TileBufType.get(
        shape,
        F16Type.get(ctx),
        vec,
        valid_shape=valid_shape,
        config=config,
        context=ctx,
    )


def build_module(ctx):
    module = Module.create()
    with InsertionPoint(module.body):
        # %src / %dst0 / %dst1 via pto.alloc_tile for smoke typing
        src = pto.alloc_tile(
            tile_type(ctx, [64, 128], [64, 128], pto.BLayout.RowMajor,
                      pto.SLayout.NoneBox)
        )
    return module, src


def main() -> None:
    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        module, src = build_module(ctx)
        with InsertionPoint(module.body):
            index_type = IndexType.get(ctx)
            row0 = arith.ConstantOp(index_type, IntegerAttr.get(index_type, 8)).result
            col0 = arith.ConstantOp(index_type, IntegerAttr.get(index_type, 16)).result
            row1 = arith.ConstantOp(index_type, IntegerAttr.get(index_type, 24)).result
            col1 = arith.ConstantOp(index_type, IntegerAttr.get(index_type, 48)).result
            dst0 = pto.alloc_tile(
                tile_type(ctx, [32, 64], [32, 64], pto.BLayout.ColMajor,
                          pto.SLayout.RowMajor)
            )
            dst1 = pto.alloc_tile(
                tile_type(ctx, [16, 32], [13, 29], pto.BLayout.ColMajor,
                          pto.SLayout.RowMajor)
            )
            # Legacy single-output constructor stays source compatible.
            legacy = pto.TExtractOp(src, row0, col0, dst0)
            assert_true(legacy.indexRow is not None and legacy.indexCol is not None,
                        "legacy .indexRow/.indexCol must exist")
            assert_true(legacy.dst is not None, "legacy .dst must exist")
            # The old generated-binder camelCase keywords must keep working.
            legacy_camel = pto.TExtractOp(
                src,
                indexRow=row0,
                indexCol=col0,
                dst=dst0,
                preQuantScalar=None,
                accToVecMode=None,
                reluPreMode=None,
            )
            assert_true(legacy_camel.dst is not None,
                        "camelCase keyword construction must work")
            # Free function keeps the old signature.
            legacy_fn = pto.textract(src, row0, col0, dst0)
            assert_true(legacy_fn.dst is not None, "textract() .dst must exist")
            # Dual-output factory returns a facade instance on the same class.
            dual = pto.TExtractOp.build_nd_to_2xnz(
                src, row0, col0, row1, col1, dst0, dst1
            )
            assert_true(isinstance(dual, pto.TExtractOp),
                        "build_nd_to_2xnz must return pto.TExtractOp")
            assert_true(len(dual.indices) == 4, "dual form must carry four indices")
            assert_true(len(dual.dsts) == 2, "dual form must carry two destinations")
        text = str(module)
        assert_true("pto.textract" in text, "op name must stay pto.textract")
        # Canonical dual-output text must round-trip with the custom parser.
        parsed = Module.parse(text, context=ctx)
        parsed_text = str(parsed)
        assert_true("pto.textract" in parsed_text, "parse must keep pto.textract")
        # The parser's opview must be the facade (legacy property surface).
        parsed_ops = list(parsed.body.operations)
        flawed = [op for op in parsed_ops
                  if op.operation.name == "pto.textract"
                  and not isinstance(op, pto.TExtractOp)]
        missing = [op for op in parsed_ops
                   if op.operation.name == "pto.textract"
                   and not hasattr(op, "indexRow")]
        assert_true(not flawed, "parsed pto.textract opview must be the facade")
        assert_true(not missing,
                    "parsed pto.textract opview must expose legacy properties")

    print("textract_nd2xnz bindings smoke OK")


if __name__ == "__main__":
    main()
