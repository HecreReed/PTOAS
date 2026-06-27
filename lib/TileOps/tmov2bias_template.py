# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tmov - Mat to Bias buffer movement.

This template implements the TMOV_M2B scenario for cube kernels:
  - Source: L1 Mat buffer (memory_space="mat", 1xN row-major layout)
  - Destination: L0 Bias Table buffer (memory_space="bias")
  - Uses bias_load (mte_l1_bt) intrinsic operation

The Bias Table is a special 4KB buffer in L0 used for bias addition
in matmul operations. Requirements:
  - Row dimension must be 1
  - Column dimension * sizeof(dtype) must be aligned to 64 bits
  - Total size must not exceed 4KB (4096 bytes)

Constraint: This template is selected when dst.memory_space == BIAS.
"""

import tilelang_dsl as pto


def _tmov_m2b_constraint(src: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint: Mat to Bias transfer scenario.

    Supported scenario:
      - src.memory_space == MAT
      - dst.memory_space == BIAS
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

    # Check dst is BIAS
    if isinstance(dst_ms, str):
        dst_is_bias = dst_ms == "bias"
    elif isinstance(dst_ms, pto.MemorySpace):
        dst_is_bias = dst_ms == pto.MemorySpace.BIAS
    else:
        dst_is_bias = hasattr(dst_ms, "value") and dst_ms.value == "bias"

    return src_is_mat and dst_is_bias


@pto.ckernel(
    target="a5",
    op="pto.tmov",
    constraints=[_tmov_m2b_constraint],
    dtypes=[
        (pto.f32, pto.f32),
        (pto.f16, pto.f32),
        (pto.bf16, pto.f32),
        (pto.i32, pto.i32),
    ],
)
def template_tmov_m2b(src: pto.Tile, dst: pto.Tile):
    """Move data from Mat buffer to Bias Table buffer.

    Args:
        src: Source tile in L1 Mat location (1xN row-major)
        dst: Destination tile in Bias Table location

    The bias data is moved using burst transfer to the Bias Table.

    mte_l1_bt semantics:
      - len_burst = N (number of bias load units, where N = column count)
      - Each load unit corresponds to one bias channel/column
      - For 1x16 f32 bias: len_burst = 16
    """
    # Bias has shape 1xN, we derive N from the valid shape
    _, n = dst.valid_shape
    # len_burst = N (number of bias channels/columns)
    # See test/tilelang_st/npu/a5/src/st/testcase/tmatmul_bias/tmatmul_bias.pto:79
    len_burst = n
    # nburst = (n_burst, src_gap, dst_gap) - single burst with no gaps
    n_burst = 1
    src_gap = 0
    dst_gap = 0
    pto.mte_l1_bt(src.as_ptr(), dst.as_ptr(), len_burst, nburst=(n_burst, src_gap, dst_gap))
    return