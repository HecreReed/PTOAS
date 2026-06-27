# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmov - Mat to Left buffer movement.

This template implements the TMOV_M2L scenario for cube kernels:
  - Source: L1 Mat buffer (memory_space="mat")
  - Destination: L0A Left buffer (memory_space="left")
  - Uses left_load (mte_l1_l0a) intrinsic operation

The operation is part of the cube matmul data flow where:
  1. Data is loaded from GM to L1 Mat via TLOAD
  2. TMOV moves data from L1 Mat to L0A Left
  3. The Left buffer is then used as input for TMATMUL

Constraint: This template is selected when dst.memory_space == LEFT.
"""

import tilelang_dsl as pto


def _tmov_m2l_constraint(src: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint: Mat to Left transfer scenario.

    Supported scenario:
      - src.memory_space == MAT
      - dst.memory_space == LEFT
    """
    src_ms = src.memory_space
    dst_ms = dst.memory_space

    # Check src is MAT
    if isinstance(src_ms, str):
        src_is_mat = src_ms == "mat"
    elif isinstance(src_ms, pto.MemorySpace):
        src_is_mat = src_ms == pto.MemorySpace.MAT
    else:
        src_is_mat = hasattr(src_ms, "value") and src_ms.value == "mat"

    # Check dst is LEFT
    if isinstance(dst_ms, str):
        dst_is_left = dst_ms == "left"
    elif isinstance(dst_ms, pto.MemorySpace):
        dst_is_left = dst_ms == pto.MemorySpace.LEFT
    else:
        dst_is_left = hasattr(dst_ms, "value") and dst_ms.value == "left"

    return src_is_mat and dst_is_left


@pto.ckernel(
    target="a5",
    op="pto.tmov",
    constraints=[_tmov_m2l_constraint],
    dtypes=[
        (pto.f16, pto.f16),
        (pto.bf16, pto.bf16),
        (pto.f32, pto.f32),
        (pto.i8, pto.i8),
    ],
)
def template_tmov_m2l(src: pto.Tile, dst: pto.Tile):
    """Move data from Mat buffer to Left buffer.

    Args:
        src: Source tile in L1 Mat location
        dst: Destination tile in L0A Left location

    The m, k dimensions are derived from the tile shapes.
    """
    m, k = dst.valid_shape
    pto.mte_l1_l0a(src.as_ptr(), dst.as_ptr(), m, k)
    return