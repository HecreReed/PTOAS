# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for the pto.textract ND-to-2xNZ dual-output form.

Design doc sections 9.1/9.2: a single op name pto.textract hosts three
template families on A5 (plain extract, fp form, and this one:
a5.textract / a5.textract_fp / a5.textract_nd2xnz). The selection layer
distinguishes them by operand arity plus layout/dtype constraints: the
dual-output form always carries seven operands
(src, row0, col0, row1, col1, dst0, dst1).
"""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _elem_bytes(dst):
    dtype = str(getattr(dst, "dtype", "f16"))
    width = {
        "f16": 16, "bf16": 16, "f32": 32, "i8": 8, "i16": 16,
        "i32": 32, "si8": 8, "ui8": 8, "si32": 32,
    }.get(dtype, 16)
    return width // 8


def _nd2xnz_constraint(src_kind, src_memory_space, dst0_kind, dst0_memory_space,
                       dst1_kind, dst1_memory_space, **_):
    return (
        src_kind == "tile"
        and dst0_kind == "tile"
        and dst1_kind == "tile"
        and src_memory_space == "vec"
        and dst0_memory_space == "vec"
        and dst1_memory_space == "vec"
    )


def _expand_window(src, row0, col0, dst):
    """Expand one ND window into one NZ destination (design doc 9.2).

    Window element mapping (design doc 3.2):
        window[r, c] = src[row0 + r, col0 + c]
    written to the NZ layout of dst. First-version coverage: c0-aligned
    full-valid window rows via vldas/vldus/vsstb with the destination block
    stride taken from the destination physical rows (plain NZ). Unaligned
    sub-c0 offsets, 1x1, FP4 and RowPlusOne stay gated until device goldens
    land.
    """
    m, n = dst.valid_shape
    c0 = 32 // _elem_bytes(dst)
    block_stride = dst.shape[0]  # storageRows (plain NZ); design doc 3.2
    src_ptr = src.as_ptr()
    base_elems = row0 * src.shape[1] + col0
    align = pto.vldas(src_ptr)
    for r in range(m):
        row_addr = base_elems + r * src.shape[1]
        value, align = pto.vldus(src_ptr + row_addr * _elem_bytes(dst), align)
        offset = r * c0
        pto.vsstb(value, dst.as_ptr() + offset, block_stride, 0, mask="PAT_ALL")


@tilelib.tile_template(
    op="pto.textract",
    target="a5",
    name="template_textract_nd2xnz",
    dtypes=(("f16", "i32", "i32", "i32", "i32", "f16", "f16"),),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[_nd2xnz_constraint],
    id=100,
    loop_depth=1,
    is_post_update=False,
    tags=("extract", "vec", "nd2xnz"),
)
def template_textract_nd2xnz(
    src: pto.Tile,
    index_row0: pto.i32,
    index_col0: pto.i32,
    index_row1: pto.i32,
    index_col1: pto.i32,
    dst0: pto.Tile,
    dst1: pto.Tile,
):
    """Dual-output ND-to-2xNZ TEXTRACT: expand two independent windows."""
    _expand_window(src, index_row0, index_col0, dst0)
    _expand_window(src, index_row1, index_col1, dst1)
