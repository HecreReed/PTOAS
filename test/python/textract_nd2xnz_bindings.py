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

from ptoas.mlir.ir import Context, InsertionPoint, Location, Module
from ptoas.mlir.dialects import pto


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def build_module(ctx):
    module = Module.create()
    with InsertionPoint(module.body):
        # %src / %dst0 / %dst1 via pto.alloc_tile for smoke typing
        src = pto.alloc_tile(
            type=pto.TileBufType.get(
                shape=[64, 128],
                element_type=pto.f16,
                memory_space=pto.AddressSpace.VEC,
                valid_shape=[64, 128],
                blayout=pto.BLayout.RowMajor,
                slayout=pto.SLayout.NoneBox,
            ),
        )
    return module, src


def main() -> None:
    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        module, src = build_module(ctx)
        with InsertionPoint(module.body):
            dst0 = pto.alloc_tile(
                type=pto.TileBufType.get(
                    shape=[32, 64],
                    element_type=pto.f16,
                    memory_space=pto.AddressSpace.VEC,
                    valid_shape=[32, 64],
                    blayout=pto.BLayout.ColMajor,
                    slayout=pto.SLayout.RowMajor,
                ),
            )
            dst1 = pto.alloc_tile(
                type=pto.TileBufType.get(
                    shape=[16, 32],
                    element_type=pto.f16,
                    memory_space=pto.AddressSpace.VEC,
                    valid_shape=[13, 29],
                    blayout=pto.BLayout.ColMajor,
                    slayout=pto.SLayout.RowMajor,
                ),
            )
            # Legacy single-output constructor stays source compatible.
            legacy = pto.TExtractOp(src, 8, 16, dst0)
            assert_true(legacy.indexRow is not None and legacy.indexCol is not None,
                        "legacy .indexRow/.indexCol must exist")
            assert_true(legacy.dst is not None, "legacy .dst must exist")
            # Free function keeps the old signature.
            legacy_fn = pto.textract(src, 8, 16, dst0)
            assert_true(legacy_fn.dst is not None, "textract() .dst must exist")
            # Dual-output factory on the same op class.
            dual = pto.TExtractOp.build_nd_to_2xnz(src, 8, 16, 24, 48, dst0, dst1)
            assert_true(len(dual.indices) == 4, "dual form must carry four indices")
            assert_true(len(dual.dsts) == 2, "dual form must carry two destinations")
        text = str(module)
        assert_true("pto.textract" in text, "op name must stay pto.textract")
        # Canonical dual-output text must round-trip with the custom parser.
        parsed = Module.parse(text, context=ctx)
        parsed_text = str(parsed)
        assert_true("pto.textract" in parsed_text, "parse must keep pto.textract")

    print("textract_nd2xnz bindings smoke OK")


if __name__ == "__main__":
    main()
