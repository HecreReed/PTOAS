# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmov2right ST test cases.

Tests the TMOV2RIGHT (Mat->Right) operation explicitly through a matmul flow:
  - TLOAD: GM -> L1 Mat
  - TMOV2LEFT: L1 Mat -> L0A Left (mte_l1_l0a)
  - TMOV2RIGHT: L1 Mat -> L0B Right (mte_l1_l0b) - the operation being tested
  - TMATMUL: compute Acc = Left x Right
  - TSTORE: Acc -> GM

The correctness of TMOV2RIGHT is verified by comparing the matmul output
with the expected golden result.
"""

import numpy as np


CASES = [
    {
        "name": "f16_16x16x16",
        "dtype": np.float16,
        "shape_a": (16, 16),
        "shape_b": (16, 16),
        "shape_c": (16, 16),
        "eps": 1e-2,
    },
]