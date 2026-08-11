# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for trem ST test cases.

Each case now also carries explicit src/tmp/dst fields so A5 tmp placeholders
are not conflated with src shape.

gen_data.py and compare.py both import this list to avoid redundant definitions.
"""

import numpy as np

CASES = [
    {
        "name": "f32_16x64",
        "dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "f32_32x32",
        "dtype": np.float32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
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
