# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmov - Acc to Vec/UB buffer movement.

This template implements the TMOV_A2V scenario for cube kernels:
  - Source: L0C Accumulator buffer (memory_space="acc")
  - Destination: UB (Unified Buffer) Vec location (memory_space="ub")
  - Uses acc_store_ub (mte_l0c_ub) intrinsic operation

This is part of the fixpipe path where accumulator results are
moved from the cube unit to the vector unit for further processing.

Constraint: This template is selected when src.memory_space == ACC
and dst.memory_space == UB.
"""

import tilelang_dsl as pto


def _tmov_a2v_constraint(src: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint: Acc to Vec/UB transfer scenario.

    Supported scenario:
      - src.memory_space == ACC
      - dst.memory_space == UB
    """
    src_ms = src.memory_space
    dst_ms = dst.memory_space

    # Check src is ACC
    if isinstance(src_ms, str):
        src_is_acc = src_ms == "acc"
    elif isinstance(src_ms, pto.MemorySpace):
        src_is_acc = src_ms == pto.MemorySpace.ACC
    else:
        src_is_acc = hasattr(src_ms, "value") and src_ms.value == "acc"

    # Check dst is UB (or "vec" which maps to UB)
    if isinstance(dst_ms, str):
        dst_is_ub = dst_ms == "ub" or dst_ms == "vec"
    elif isinstance(dst_ms, pto.MemorySpace):
        dst_is_ub = dst_ms == pto.MemorySpace.UB
    else:
        dst_is_ub = hasattr(dst_ms, "value") and (dst_ms.value == "ub" or dst_ms.value == "vec")

    return src_is_acc and dst_is_ub


@pto.ckernel(
    target="a5",
    op="pto.tmov",
    constraints=[_tmov_a2v_constraint],
    dtypes=[
        (pto.f32, pto.f32),
        (pto.i32, pto.i32),
    ],
)
def template_tmov_a2v(src: pto.Tile, dst: pto.Tile):
    """Move data from Acc buffer to Vec/UB buffer.

    Args:
        src: Source tile in Acc location
        dst: Destination tile in Vec/UB location

    The m, n dimensions and strides are derived from the tile shapes.
    This performs NZ2ND layout conversion.
    """
    m, n = dst.valid_shape
    src_stride = (m + 15) // 16 * 16  # Align to 16 blocks
    dst_stride = n  # Row-major stride
    # mte_l0c_ub takes 7 positional args: src, dst, m, n, src_stride, dst_stride, dst_mode
    # dst_mode: integer for sub_blockid mode (0 or 1)
    # layout: keyword arg for layout conversion ("nz2nd" for NZ to ND)
    pto.mte_l0c_ub(
        src.as_ptr(), dst.as_ptr(), m, n, src_stride, dst_stride,
        0,  # dst_mode: sub_blockid value 0
        layout="nz2nd",
    )
    return