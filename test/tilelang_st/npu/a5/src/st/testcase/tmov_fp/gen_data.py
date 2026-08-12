# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmov_fp ST test cases.

Tests the TMOV_FP (Acc+Scaling->Mat fixpipe quantization) path.
Golden: (A @ B).astype(f16) (fixpipe quantization with scale=1, like textract_fp).

Note: Fixpipe requires 16x16 scaling tile format. For scale=1 quantization,
each row of the scaling tile contains all 1s (identity scaling).
"""

import numpy as np

from cases import CASES
from st_common import setup_case_rng, save_case_data


for case in CASES:
    setup_case_rng(case)

    shape_a = case["shape_a"]
    shape_b = case["shape_b"]
    shape_scale = case["shape_scale"]  # (16, 16) for fixpipe format
    shape_id = case["shape_id"]
    dtype_a = case["dtype_a"]
    dtype_b = case["dtype_b"]
    dtype_scale = case["dtype_scale"]
    dtype_id = case["dtype_id"]

    lhs = np.random.uniform(-1.0, 1.0, size=shape_a).astype(dtype_a)
    rhs = np.random.uniform(-1.0, 1.0, size=shape_b).astype(dtype_b)

    # Fixpipe scaling buffer requires 16x16 format with all 1s for identity scaling
    # (similar to textract_fp test which validates the fixpipe path)
    scale = np.ones(shape_scale, dtype=dtype_scale)

    # Identity matrix for readback validation
    identity = np.eye(shape_id[0], shape_id[1], dtype=dtype_id)

    # Compute golden: fixpipe quantization output with scale=1
    # Acc = A @ B (f32)
    # Mat_f16 = fixpipe(Acc, scale=1) = Acc.astype(f16) (quantization)
    # Output = Mat_f16.astype(f32) (readback via identity matmul)
    acc = np.matmul(lhs.astype(np.float32), rhs.astype(np.float32))
    mat_f16 = acc.astype(np.float16)  # Fixpipe quantization result (f16) with scale=1
    # Readback: Mat_f16 @ Identity -> Output (f32)
    golden = mat_f16.astype(np.float32)

    save_case_data(case["name"], {"input1": lhs, "input2": rhs, "scale": scale, "identity": identity, "golden": golden})
    print(
        f"[INFO] gen_data: {case['name']} "
        f"lhs={shape_a} rhs={shape_b} scale={shape_scale} (all_ones) id={shape_id} out={case['shape_c']} dtype=f32"
    )