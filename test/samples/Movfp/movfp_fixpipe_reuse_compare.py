#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import finalize_compare, load_case_meta


M = 16
K = 32
N = 32
F16_MAX = np.float32(np.finfo(np.float16).max)


def read_exact(path: str, dtype, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if values.size != count:
        raise ValueError(f"{path}: expected {count} elements, got {values.size}")
    return values


def unpack_deqf16_scales(payload: np.ndarray) -> np.ndarray:
    """Decode the m1 field consumed by A5 VDEQF16 vector quantization."""
    words = np.asarray(payload, dtype=np.uint64)
    m1_bits = ((words >> np.uint64(13)) & np.uint64(0x7FFFF)).astype(np.uint32)
    float_bits = (
        ((m1_bits >> np.uint32(18)) & np.uint32(1)) << np.uint32(31)
        | ((m1_bits >> np.uint32(10)) & np.uint32(0xFF)) << np.uint32(23)
        | (m1_bits & np.uint32(0x3FF)) << np.uint32(13)
    )
    return float_bits.view(np.float32)


def clip_to_f16(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -F16_MAX, F16_MAX).astype(np.float16)


def report_mismatch(name: str, expected: np.ndarray, output: np.ndarray):
    expected_f32 = expected.astype(np.float32, copy=False)
    output_f32 = output.astype(np.float32, copy=False)
    invalid = ~np.isfinite(output_f32)
    if np.any(invalid):
        index = int(np.flatnonzero(invalid)[0])
    else:
        diff = np.abs(expected_f32 - output_f32)
        index = int(np.argmax(diff))
    mismatch_count = int(np.count_nonzero(expected != output))
    print(
        f"[ERROR] Mismatch: recomputed_{name} vs {name}.bin, idx={index} "
        f"(expected={expected_f32[index]}, out={output_f32[index]}, "
        f"mismatches={mismatch_count}, nan={int(np.isnan(output_f32).sum())}, "
        f"inf={int(np.isinf(output_f32).sum())})"
    )


def check_golden_artifact(name: str, expected: np.ndarray):
    path = Path(f"golden_{name}.bin")
    if not path.is_file():
        print(f"[WARN] Golden artifact missing: {path}")
        return
    golden = np.fromfile(path, dtype=np.float16)
    if golden.shape == expected.shape and np.array_equal(golden, expected):
        return
    nan_count = int(np.isnan(golden).sum())
    inf_count = int(np.isinf(golden).sum())
    print(
        f"[WARN] Ignoring inconsistent {path}: shape={golden.shape}, "
        f"expected_shape={expected.shape}, nan={nan_count}, inf={inf_count}"
    )


def main():
    meta = load_case_meta()
    if len(meta.inputs) < 4:
        raise ValueError(f"expected at least 4 inputs, got {meta.inputs}")
    if len(meta.outputs) < 3:
        raise ValueError(f"expected at least 3 outputs, got {meta.outputs}")

    lhs_name, rhs_name, fp0_name, fp1_name = meta.inputs[:4]
    out0_name, out1_name, out2_name = meta.outputs[:3]
    lhs = read_exact(f"{lhs_name}.bin", meta.np_types[lhs_name], M * K).reshape(M, K)
    rhs = read_exact(f"{rhs_name}.bin", meta.np_types[rhs_name], K * N).reshape(K, N)
    fp0 = read_exact(f"{fp0_name}.bin", meta.np_types[fp0_name], N)
    fp1 = read_exact(f"{fp1_name}.bin", meta.np_types[fp1_name], N)

    acc = lhs.astype(np.int32) @ rhs.astype(np.int32)
    scale0 = unpack_deqf16_scales(fp0).reshape(1, N)
    scale1 = unpack_deqf16_scales(fp1).reshape(1, N)
    expected = {
        out0_name: clip_to_f16(acc.astype(np.float32) * scale0).reshape(-1),
        out1_name: clip_to_f16(acc.astype(np.float32) * scale1).reshape(-1),
        out2_name: clip_to_f16(acc.astype(np.float32) * scale0).reshape(-1),
    }

    ok = True
    for name in meta.outputs:
        if name not in expected:
            print(f"[ERROR] Unexpected output: {name}")
            ok = False
            continue
        check_golden_artifact(name, expected[name])
        output_path = Path(f"{name}.bin")
        if not output_path.is_file():
            print(f"[ERROR] Output missing: {output_path}")
            ok = False
            continue
        output = np.fromfile(output_path, dtype=np.float16)
        if output.shape != expected[name].shape:
            print(
                f"[ERROR] Shape mismatch: recomputed_{name} {expected[name].shape} "
                f"vs {output_path} {output.shape}"
            )
            ok = False
            continue
        if not np.all(np.isfinite(output)) or not np.array_equal(expected[name], output):
            report_mismatch(name, expected[name], output)
            ok = False
    finalize_compare(ok)


if __name__ == "__main__":
    main()
