#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import ml_dtypes

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2

M = 1
K = 128
N = 16


def convert_scale_b_format(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    result = padded.reshape((k_padded // c0_size_mx, c0_size_mx, n_padded // 16, 16)).transpose(2, 0, 3, 1)
    return result.reshape(result.shape[1] * result.shape[3], result.shape[0] * result.shape[2])


def main():
    np.random.seed(23)

    a = np.random.randint(-10, 10, [M, K]).astype(fp8_e4m3fn)
    b = np.random.randint(-10, 10, [K, N]).astype(fp8_e5m2)

    a.tofile("input1.bin")
    b.tofile("input2.bin")

    a_scale = np.random.randint(127, 130, [M, K // 32]).astype(np.uint8)
    b_scale = np.random.randint(127, 130, [K // 32, N]).astype(np.uint8)

    a_full = np.zeros([M, K], dtype=np.float64)
    b_full = np.zeros([K, N], dtype=np.float64)
    a_mx = 2 ** (a_scale.astype(np.float64) - 127)
    b_mx = 2 ** (b_scale.astype(np.float64) - 127)
    for i in range(K):
        a_full[:, i] = a[:, i] * a_mx[:, i // 32]
        b_full[i, :] = b[i, :] * b_mx[i // 32, :]

    golden = np.matmul(a_full, b_full).astype(np.float32)
    golden.tofile("golden.bin")

    a_scale.astype(np.uint8).tofile("scale1.bin")
    convert_scale_b_format(b_scale).astype(np.uint8).tofile("scale2.bin")


if __name__ == "__main__":
    main()
