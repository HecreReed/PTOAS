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

The first version registers f16 only (design doc 9.1: unverified dtypes
must not be registered); the A5 verifier enforces the same set.
"""

import inspect

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ptodsl._ast_rewrite import rewrite_jit_function


def _elem_bytes(dst):
    dtype = str(getattr(dst, "dtype", "f16"))
    width = {
        "f16": 16, "bf16": 16, "f32": 32, "i8": 8, "i16": 16,
        "i32": 32, "si8": 8, "ui8": 8, "si32": 32,
    }.get(dtype, 16)
    return width // 8


def _nd2xnz_constraint(**context):
    # InsertTemplateAttributes/ExpandTileOp normalize the VEC address space
    # to the string "ub" (see InsertTemplateAttributes.cpp
    # stringifyMemorySpace), so candidates must declare "ub" here.
    required = {
        "src_kind": "tile",
        "dst0_kind": "tile",
        "dst1_kind": "tile",
        "src_memory_space": "ub",
        "dst0_memory_space": "ub",
        "dst1_memory_space": "ub",
    }
    return all(context.get(name) == value for name, value in required.items())


@rewrite_jit_function
def _expand_vector(src, dst, row0, col0):
    """Vector path for non-1x1 windows (design doc 9.2).

    Column blocks of c0 are walked with an exact trailing-block mask so
    validCols % c0 pads stay untouched; row padding beyond validRows is not
    written (undefined by the TEXTRACT contract, design 3.2.1). pto.addptr
    advances by element offset (not bytes).
    """
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    m, n = dst.valid_shape
    c0 = 32 // _elem_bytes(dst)
    block_stride = dst.shape[0]  # storageRows (plain NZ); design doc 3.2
    nblocks = (n + c0 - 1) // c0
    align = pto.vldas(src_ptr)
    base_elems = row0 * src.shape[1] + col0
    for cb in range(nblocks):
        cols_this = n - cb * c0
        if cols_this > c0:
            cols_this = c0
        # vsstb takes a !pto.mask SSA value; build it with make_mask so the
        # exact trailing-block width folds to a runtime predicate when n is
        # dynamic and to a constant pattern otherwise (design doc 9.2).
        # dst.dtype is already an MLIR Type on the traced tile; pass it
        # directly so make_mask can resolve the element width (a plain
        # 'f16' string is not a valid dtype descriptor).
        mask = pto.make_mask(dst.dtype, cols_this)
        for r in range(m):
            src_elem = base_elems + r * src.shape[1] + cb * c0
            value, align = pto.vldus(pto.addptr(src_ptr, src_elem), align)
            dst_elem = cb * block_stride * c0 + r * c0
            pto.vsstb(value, pto.addptr(dst_ptr, dst_elem),
                      block_stride, 0, mask=mask)


@rewrite_jit_function
def _expand_window(src, row0, col0, dst):
    """Expand one ND window into one NZ destination (design doc 9.2).

    Window element mapping (design doc 3.2):
        window[r, c] = src[row0 + r, col0 + c]
    written to the NZ layout of dst: NZ offset = floor(c/c0)*physRows*c0
    + r*c0 + (c%c0), c0 = 32/elemBytes.

    Access paths per design doc 9.2:
    - 1x1 windows use the scalar load/store path so the SyncMacroModel's A5
      1x1 scalar hidden-event model stays consistent with the lowering and
      no vector read footprint is touched;
    - other windows use vldas + vldus (unaligned); the verifier statically
      rejects sub-c0 windows whose vldus footprint would cross the source
      row end; the c0-aligned vlds optimization stays gated behind device
      goldens (design doc 9.1) and currently reuses vldas + vldus, which is
      functionally correct for aligned windows as well.

    The scalar/vector split must be structured if/else: runtime-conditional
    branches cannot use native Python `and` (the rewrite pass does not lower
    BoolOp) and must not `return` early (that would stop tracing before the
    vector path is emitted).
    """
    m, n = dst.valid_shape
    src_ptr = src.as_ptr()
    dst_ptr = dst.as_ptr()
    base_elems = row0 * src.shape[1] + col0
    if m == 1:
        if n == 1:
            # Scalar path (design doc 9.2): exactly one element. Consistent
            # with the 1x1 scalar hidden-event model (SyncMacroModel reserves
            # a bidirectional V<->S pair on event 0), the scalar S-pipe access
            # is bracketed by an explicit set/wait barrier so the S read of
            # src is ordered against the outer V pipe (design doc 6.3.1).
            pto.set_flag("V", "S", event_id=0)
            pto.wait_flag("V", "S", event_id=0)
            value = pto.load_scalar(src_ptr, base_elems)
            pto.store_scalar(dst_ptr, 0, value)
            pto.set_flag("S", "V", event_id=0)
            pto.wait_flag("S", "V", event_id=0)
        else:
            _expand_vector(src, dst, row0, col0)
    else:
        _expand_vector(src, dst, row0, col0)


def template_textract_nd2xnz(*operands):
    """Dual-output ND-to-2xNZ TEXTRACT: expand two independent windows."""
    if len(operands) != 7:
        raise TypeError("template_textract_nd2xnz expects seven operands")
    src, index_row0, index_col0, index_row1, index_col1, dst0, dst1 = operands
    _expand_window(src, index_row0, index_col0, dst0)
    _expand_window(src, index_row1, index_col1, dst1)


template_textract_nd2xnz.__signature__ = inspect.Signature(
    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      annotation=annotation)
    for name, annotation in (
        ("src", pto.Tile),
        ("index_row0", pto.i32),
        ("index_col0", pto.i32),
        ("index_row1", pto.i32),
        ("index_col1", pto.i32),
        ("dst0", pto.Tile),
        ("dst1", pto.Tile),
    )
)


template_textract_nd2xnz = tilelib.tile_template(
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
    tags=("extract", "ub", "nd2xnz"),
)(template_textract_nd2xnz)
