#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import struct
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, rng, write_buffers, write_golden


M = 16
K = 32
N = 32
F16_MAX = np.float32(np.finfo(np.float16).max)


def clip_to_f16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -F16_MAX, F16_MAX).astype(np.float16)


def pack_vector_quant(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    packed = np.empty(flat.size, dtype=np.uint64)
    for i, value in enumerate(flat):
        packed[i] = struct.unpack("!I", struct.pack("!f", float(value)))[0]
    return packed


def main():
    meta = load_case_meta()
    generator = rng()

    input_names = meta.inputs
    output_names = meta.outputs
    if len(input_names) < 4:
        raise ValueError(f"expected at least 4 inputs, got {input_names}")
    if len(output_names) < 3:
        raise ValueError(f"expected at least 3 outputs, got {output_names}")

    lhs_name, rhs_name, fp0_name, fp1_name = input_names[:4]
    out0_name, out1_name, out2_name = output_names[:3]

    lhs = generator.integers(-4, 5, size=(M, K), dtype=np.int16).astype(np.int8)
    rhs = generator.integers(-3, 4, size=(K, N), dtype=np.int16).astype(np.int8)

    cols = np.arange(N, dtype=np.float32)
    fp0_f32 = (np.float32(0.5) + cols * np.float32(0.03125)).reshape(1, N)
    fp1_f32 = (np.float32(1.0) + (cols % np.float32(7.0)) * np.float32(0.0625)).reshape(1, N)

    fp0 = pack_vector_quant(fp0_f32).astype(meta.np_types[fp0_name], copy=False).reshape(1, N)
    fp1 = pack_vector_quant(fp1_f32).astype(meta.np_types[fp1_name], copy=False).reshape(1, N)

    acc = lhs.astype(np.int32) @ rhs.astype(np.int32)

    out0 = clip_to_f16(acc.astype(np.float32) * fp0_f32)
    out1 = clip_to_f16(acc.astype(np.float32) * fp1_f32)
    out2 = clip_to_f16(acc.astype(np.float32) * fp0_f32)

    buffers = default_buffers(meta)
    buffers[lhs_name] = lhs.reshape(-1)
    buffers[rhs_name] = rhs.reshape(-1)
    buffers[fp0_name] = fp0.reshape(-1)
    buffers[fp1_name] = fp1.reshape(-1)
    for name in output_names:
        buffers[name] = np.zeros(meta.elem_counts[name], dtype=meta.np_types[name])
    write_buffers(meta, buffers)

    golden = {
        out0_name: out0.reshape(-1),
        out1_name: out1.reshape(-1),
        out2_name: out2.reshape(-1),
    }
    write_golden(meta, golden)


if __name__ == "__main__":
    main()
