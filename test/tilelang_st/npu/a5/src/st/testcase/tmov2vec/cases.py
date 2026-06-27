# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmov2vec ST test cases.

Tests the TMOV2VEC (Acc->Vec) operation explicitly through a matmul->vec flow:
  - TLOAD: GM -> L1 Mat
  - TMOV2LEFT/TMOV2RIGHT: L1 Mat -> L0A/L0B
  - TMATMUL: compute Acc = Left x Right
  - TMOV2VEC: L0C Acc -> UB Vec (mte_l0c_ub) - the operation being tested

Note: Due to VPTO lowering limitations with castptr in section mode for tile-based
vec operations, the vec_tile content is validated via VPTO IR generation, not direct
GM output. The TMOV Acc->Vec operation generates mte_l0c_ub which is validated in IR.

This tests the fixpipe path for moving accumulator results to the vector unit.
"""

import numpy as np


CASES = [
    {
        "name": "f16_f32_16x16x16",
        "dtype": np.float16,
        "shape_a": (16, 16),
        "shape_b": (16, 16),
        "shape_c": (16, 16),
        "eps": 1e-2,
    },
]