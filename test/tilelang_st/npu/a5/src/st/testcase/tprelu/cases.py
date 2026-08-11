#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tprelu ST test cases.

Each case now also carries explicit src/tmp/dst fields so A5 tmp placeholders
are not conflated with src shape.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f16_64x64",
        "dtype": np.float16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-3,
    },
    {
        "name": "f16_63x63",
        "dtype": np.float16,
        "shape": (64, 64),
        "valid_shape": (63, 63),
        "eps": 1e-3,
    },
    {
        "name": "f16_1x16384",
        "dtype": np.float16,
        "shape": (1, 16384),
        "valid_shape": (1, 16384),
        "eps": 1e-3,
    },
    {
        "name": "f16_2048x16",
        "dtype": np.float16,
        "shape": (2048, 16),
        "valid_shape": (2048, 16),
        "eps": 1e-3,
    },
    {
        "name": "f32_64x64",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-6,
    },
    {
        "name": "f32_63x63",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (63, 63),
        "eps": 1e-6,
    },
    {
        "name": "f32_1x16384",
        "dtype": np.float32,
        "shape": (1, 16384),
        "valid_shape": (1, 16384),
        "eps": 1e-6,
    },
    {
        "name": "f32_2048x8",
        "dtype": np.float32,
        "shape": (2048, 8),
        "valid_shape": (2048, 8),
        "eps": 1e-6,
    },
]


def _a5_tmp_placeholder_shape(dtype):
    return (1, max(1, 32 // np.dtype(dtype).itemsize))


def _augment_case(case):
    case = dict(case)
    case.setdefault("src_shape", case["shape"])
    case.setdefault("src_valid_shape", case["valid_shape"])
    case.setdefault("tmp_shape", _a5_tmp_placeholder_shape(case["dtype"]))
    case.setdefault("tmp_valid_shape", (1, 1))
    case.setdefault("dst_shape", case["shape"])
    case.setdefault("dst_valid_shape", case["valid_shape"])
    return case


CASES = [_augment_case(case) for case in CASES]
