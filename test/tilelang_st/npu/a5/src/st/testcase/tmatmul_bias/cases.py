# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmatmul_bias ST test cases.

Each case maps to a pto-isa tmatmul bias test (LaunchTMATMULBIAS key 1-11).
Excludes tmatmul_mx bias cases.
"""

import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2


def ceil_align(num, align):
    return (num + align - 1) // align * align


CASES = [
    # ---- case 0: f16 x f16 -> f32, bias=f32, 16x16x16 ----
    {
        "name": "f16_16x16x16",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 16, "K": 16, "N": 16,
        "M_aligned": 16, "K_use": 16, "N_aligned": 16,
        "eps": 1e-2,
    },
    # ---- case 1: i8 x i8 -> i32, bias=i32, 8x7x6 ----
    {
        "name": "i8_bias_i32_8x7x6",
        "a_dtype": np.int8,
        "b_dtype": np.int8,
        "bias_dtype": np.int32,
        "c_dtype": np.int32,
        "M": 8, "K": 7, "N": 6,
        "M_aligned": 16, "K_use": 32, "N_aligned": 32,
        "eps": 1e-6,
    },
    # ---- case 2: f16 x f16 -> f32, bias=f16, 16x15x16 ----
    {
        "name": "f16_bias_f16_16x15x16",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 16, "K": 15, "N": 16,
        "M_aligned": 16, "K_use": 16, "N_aligned": 16,
        "eps": 1e-2,
    },
    # ---- case 3: f16 x f16 -> f32, bias=bf16, 112x127x80 (SPLIT_K) ----
    {
        "name": "f16_bias_bf16_112x127x80",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 112, "K": 127, "N": 80,
        "M_aligned": 112, "K_use": 128, "N_aligned": 80,
        "eps": 1e-2,
        "split_k": True, "base_k": 64,
    },
    # ---- case 4: bf16 x bf16 -> f32, bias=bf16, 80x112x63 (SPLIT_K) ----
    {
        "name": "bf16_bias_bf16_80x112x63",
        "a_dtype": bfloat16,
        "b_dtype": bfloat16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 80, "K": 112, "N": 63,
        "M_aligned": 80, "K_use": 128, "N_aligned": 64,
        "eps": 1e-2,
        "split_k": True, "base_k": 64,
    },
    # ---- case 5: f32 x f32 -> f32, bias=f32, 127x128x63 (SPLIT_K) ----
    {
        "name": "f32_bias_f32_127x128x63",
        "a_dtype": np.float32,
        "b_dtype": np.float32,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 127, "K": 128, "N": 63,
        "M_aligned": 128, "K_use": 128, "N_aligned": 64,
        "eps": 1e-5,
    },
]
