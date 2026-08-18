#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect

from ptoas.mlir.ir import Context, InsertionPoint, Location, Module
from ptoas.mlir.dialects import pto


MODULE_TEXT = r"""
module {
  func.func @textract_bindings(
      %src: !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row0: index, %col0: index, %row1: index, %col1: index,
      %dst0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64, v_row=32, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>,
      %dst1: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32, v_row=16, v_col=32, blayout=col_major, slayout=row_major, fractal=512, pad=0>) {
    return
  }
}
"""


PARSED_TEXT = r"""
module {
  func.func @parsed_textract(
      %src: !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %row: index, %col: index,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64, v_row=32, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>) {
    pto.textract ins(%src, %row, %col : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=128, v_row=64, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index, index)
                 outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=64, v_row=32, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    return
  }
}
"""


def assert_legacy_properties(op) -> None:
    assert op.indexRow is not None
    assert op.indexCol is not None
    assert op.dst is not None


def main() -> None:
    signature = inspect.signature(pto.textract)
    assert list(signature.parameters)[:4] == [
        "src",
        "index_row",
        "index_col",
        "dst",
    ]

    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        module = Module.parse(MODULE_TEXT)
        func = module.body.operations[0]
        block = func.regions[0].blocks[0]
        src, row0, col0, row1, col1, dst0, dst1 = block.arguments

        with InsertionPoint.at_block_terminator(block):
            positional = pto.TExtractOp(src, row0, col0, dst0)
            keyword = pto.TExtractOp(
                src=src, index_row=row0, index_col=col0, dst=dst0
            )
            free_positional = pto.textract(src, row0, col0, dst0)
            free_keyword = pto.textract(
                src=src, index_row=row0, index_col=col0, dst=dst0
            )
            dual = pto.TExtractOp.build_nd_to_2xnz(
                src, row0, col0, row1, col1, dst0, dst1
            )

        for op in (positional, keyword, free_positional, free_keyword):
            assert isinstance(op, pto.TExtractOp)
            assert_legacy_properties(op)
        assert len(dual.indices) == 4
        assert len(dual.dsts) == 2

        parsed = Module.parse(PARSED_TEXT)
        parsed_block = parsed.body.operations[0].regions[0].blocks[0]
        raw = next(
            candidate.operation
            for candidate in parsed_block.operations
            if candidate.operation.name == "pto.textract"
        )
        view = raw.opview
        assert isinstance(view, pto.TExtractOp)
        assert_legacy_properties(view)

    print("textract_bindings: PASS")


if __name__ == "__main__":
    main()
