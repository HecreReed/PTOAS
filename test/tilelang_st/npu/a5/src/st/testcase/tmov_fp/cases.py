# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmov_fp ST test cases.

Tests the TMOV_FP (Acc+Scaling->Mat fixpipe quantization) path:
  - TMOV Mat->Scaling: L1 Mat -> Scaling buffer (mte_l1_fb)
  - TMOV_FP: Acc(f32) + scaling buffer -> Mat(f16) via fixpipe quantization
  - Readback: Mat(f16) x identity -> Acc(f32) -> GM for validation

Golden computation: (A @ B) * scale (per-column scale multiplication)
This matches the fixpipe quantization semantics: output = accumulator * scale
"""

import numpy as np


CASES = [
    {
        "name": "f16_16x16x16",
        "dtype_a": np.float16,
        "dtype_b": np.float16,
        "dtype_scale": np.float32,
        "dtype_id": np.float16,
        "dtype_c": np.float32,
        "shape_a": (16, 16),
        "shape_b": (16, 16),
        "shape_scale": (16, 16),  # Fixpipe requires 16x16 scaling tile
        "shape_id": (16, 16),
        "shape_c": (16, 16),
        "eps": 1e-3,
    },
]