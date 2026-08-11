#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np
import ml_dtypes

import cases
from cases import CASES
from compare import normalize_dtype
from st_common import save_case_data, setup_case_rng, validate_cases


def is_sub_float(dtype):
    if isinstance(dtype, str):
        return False
    return np.issubdtype(dtype, np.floating) or dtype == ml_dtypes.bfloat16


def is_sub_int(dtype):
    if isinstance(dtype, str):
        return False
    return np.issubdtype(dtype, np.integer)


def bits_to_f32(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def f32_to_bits(value):
    return np.array([value], dtype=np.float32).view(np.uint32)[0]


def f32_to_bf16_bits(value):
    bits = int(f32_to_bits(value))
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return np.uint16((rounded >> 16) & 0xFFFF)


def decode_f8e4m3fn(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if byte in (0x7F, 0xFF):
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 8.0) * (2.0 ** -6))
    return np.float32(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))


def decode_f8e5m2(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 2) & 0x1F
    mant = byte & 0x03
    if exp == 0x1F:
        if mant == 0:
            return np.float32(sign * np.inf)
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 4.0) * (2.0 ** -14))
    return np.float32(sign * (1.0 + mant / 4.0) * (2.0 ** (exp - 15)))


def fp32_constructor(sign, exp, mant):
    return ((sign & 1) << 31) | ((exp & 0xFF) << 23) | (mant & 0x7FFFFF)


def decode_hif8(byte):
    if byte == 0x00:
        return bits_to_f32(0x00000000)
    if byte == 0x80:
        return np.float32(np.nan)
    if byte == 0x6F:
        return bits_to_f32(0x7F800000)
    if byte == 0xEF:
        return bits_to_f32(0xFF800000)

    input_sign = (byte >> 7) & 0x01
    bit6 = (byte >> 6) & 0x01
    bit5 = (byte >> 5) & 0x01
    bit4 = (byte >> 4) & 0x01
    bit3 = (byte >> 3) & 0x01
    if bit6 == 0 and bit5 == 0 and bit4 == 0 and bit3 == 0:
        return bits_to_f32(fp32_constructor(input_sign, (byte & 0x7) - 23 + 127, 0))

    if bit6 == 0:
        if bit5 == 0:
            exp_width = 0 if bit4 == 0 else 1
            d_width = 4 if bit4 == 0 else 3
        else:
            exp_width = 2
            d_width = 2
    else:
        exp_width = 3 if bit5 == 0 else 4
        d_width = 2
    man_width = 8 - d_width - exp_width - 1

    exp_mask = (1 << exp_width) - 1
    exp = 0
    if exp_width != 0:
        exp = ((byte >> man_width) & exp_mask) | (1 << (exp_width - 1))
        exp_msb = (byte >> (man_width + exp_width - 1)) & 0x1
        if exp_msb != 0:
            exp = -exp
    exp += 127

    man_mask = (1 << man_width) - 1
    mant = (byte & man_mask) << (23 - man_width)
    return bits_to_f32(fp32_constructor(input_sign, exp, mant))


FP4E1M2_TO_BF16 = np.array(
    [
        0x0000, 0x3E80, 0x3F00, 0x3F40, 0x3F80, 0x3FA0, 0x3FC0, 0x3FE0,
        0x8000, 0xBE80, 0xBF00, 0xBF40, 0xBF80, 0xBFA0, 0xBFC0, 0xBFE0,
    ],
    dtype=np.uint16,
)

FP4E2M1_TO_BF16 = np.array(
    [
        0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
        0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
    ],
    dtype=np.uint16,
)


def f8e4_quantize_pairs():
    exact = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x3C, 0xBC, 0x7E, 0xFE]
    pairs = [(decode_f8e4m3fn(byte), byte) for byte in exact]
    # The f32->fp8 TileLang template uses #sat=1. In V300
    # instruction-controlled saturation, infinities/overflow clamp to max
    # finite and NaN is saturated to zero.
    pairs.extend([
        (np.float32(np.inf), 0x7E),
        (np.float32(-np.inf), 0xFE),
        (np.float32(np.nan), 0x00),
        (np.float32(1000.0), 0x7E),
        (np.float32(-1000.0), 0xFE),
        (np.float32(1.0625), 0x38),
        (np.float32(1.1875), 0x3A),
        (np.float32(-1.0625), 0xB8),
        (np.float32(-1.1875), 0xBA),
    ])
    return pairs


def f8e5_quantize_pairs():
    exact = [0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84, 0x3C, 0xBC, 0x40, 0xC0, 0x7B, 0xFB]
    pairs = [(decode_f8e5m2(byte), byte) for byte in exact]
    # The f32->fp8 TileLang template uses #sat=1. In V300
    # instruction-controlled saturation, infinities/overflow clamp to max
    # finite and NaN is saturated to zero.
    pairs.extend([
        (np.float32(np.inf), 0x7B),
        (np.float32(-np.inf), 0xFB),
        (np.float32(np.nan), 0x00),
        (np.float32(1.0e10), 0x7B),
        (np.float32(-1.0e10), 0xFB),
        (np.float32(1.125), 0x3C),
        (np.float32(1.375), 0x3E),
        (np.float32(-1.125), 0xBC),
        (np.float32(-1.375), 0xBE),
    ])
    return pairs


def hif8_quantize_pairs():
    exact = [
        0x00, 0x80, 0x6F, 0xEF, 0x01, 0x81, 0x07, 0x87,
        0x08, 0x88, 0x10, 0x90, 0x18, 0x98, 0x20, 0xA0,
        0x40, 0xC0, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
    ]
    return [(decode_hif8(byte), byte) for byte in exact]


def f4e1_quantize_pairs():
    pairs = [(bits, nibble) for nibble, bits in enumerate(FP4E1M2_TO_BF16)]
    pairs.extend([
        (f32_to_bf16_bits(0.625), 0x2),
        (f32_to_bf16_bits(0.75), 0x3),
        (f32_to_bf16_bits(0.875), 0x4),
        (f32_to_bf16_bits(-0.625), 0xA),
        (f32_to_bf16_bits(-0.75), 0xB),
        (f32_to_bf16_bits(-0.875), 0xC),
    ])
    return pairs


def f4e2_quantize_pairs():
    pairs = [(bits, nibble) for nibble, bits in enumerate(FP4E2M1_TO_BF16)]
    pairs.extend([
        (f32_to_bf16_bits(0.75), 0x2),
        (f32_to_bf16_bits(1.25), 0x2),
        (f32_to_bf16_bits(1.75), 0x4),
        (f32_to_bf16_bits(-0.75), 0xA),
        (f32_to_bf16_bits(-1.25), 0xA),
        (f32_to_bf16_bits(-1.75), 0xC),
    ])
    return pairs


def _make_input_inner(src_dtype, shape):
    total = int(np.prod(shape))
    float_types = (np.float32, np.float16, ml_dtypes.bfloat16)
    int8_like_types = (np.int8, )

    # Generate input data
    if src_dtype in float_types:
        return (np.random.random([total]) * 200 - 100)
    elif src_dtype in int8_like_types:
        return np.random.randint(-128, 128, [total])
    elif src_dtype == np.uint8:
        return np.random.randint(0, 256, [total])
    elif src_dtype == np.int16:
        return np.random.randint(-1000, 1000, [total])
    elif src_dtype == np.uint16:
        return np.random.randint(0, 10000, [total])
    elif src_dtype in (np.int32, np.int64):
        return np.random.randint(-10000, 10000, [total])
    elif src_dtype == np.uint32:
        return np.random.randint(0, 10000, [total])
    else:
        return np.random.randint(-10000, 10000, [total])


def make_input(src_dtype, shape):
    return _make_input_inner(src_dtype, shape).astype(normalize_dtype(src_dtype)).reshape(shape)


def round_half_away_from_zero(values):
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def default_saturation_off(src_dtype, dst_dtype):
    """Mirror the current A5 default saturation policy for supported pairs."""
    return (
        (src_dtype is np.float16 and dst_dtype is np.uint8)
        or (src_dtype is np.float16 and dst_dtype is np.int8)
        or (src_dtype is np.float32 and dst_dtype is np.int16)
        or (src_dtype is np.float16 and dst_dtype is np.int16)
        or (src_dtype is np.int64 and dst_dtype is np.int32)
        or (src_dtype is np.int32 and dst_dtype is np.int16)
    )


def apply_round_mode(values, round_mode):
    rounding_funcs = {
        "RINT": np.rint,
        "ROUND": round_half_away_from_zero,
        "FLOOR": np.floor,
        "CEIL": np.ceil,
        "TRUNC": np.trunc,
    }
    return rounding_funcs.get(round_mode, np.rint)(values)


def convert(values: np.ndarray, src_dtype, dst_dtype, round_mode=None):
    is_float_src = is_sub_float(src_dtype)
    is_int_dst = is_sub_int(dst_dtype)
    is_f32_to_f32 = src_dtype == np.float32 and dst_dtype == np.float32
    needs_rounding = is_float_src and (is_int_dst or is_f32_to_f32)

    if needs_rounding:
        values = apply_round_mode(values, round_mode or "RINT")
    
    if is_int_dst:
        # Determine if this conversion has default saturation OFF (truncation) or ON (clamping)
        if default_saturation_off(src_dtype, dst_dtype):
            # OFF (truncation): bit extraction - wrap around using modulo
            return truncate_to_int(values, dst_dtype)
        else:
            # Saturation ON: clamp to range (widen to int64/float64 to preserve sign)
            return clamp_to_range_int(values, dst_dtype)
    elif is_sub_float(dst_dtype):
        return clamp_to_range_float(values, dst_dtype)
    else:
        return values.astype(dst_dtype)


def truncate_to_int(values: np.ndarray, dst_dtype):
    golden_list = []
    for val in values.flat:
        int_val = 0 if np.isnan(val) or np.isinf(val) else int(np.int64(val))

        if dst_dtype == np.int8:
            byte_val = int_val & 0xFF
            truncated_val = byte_val if byte_val < 128 else byte_val - 256
        elif dst_dtype == np.uint8:
            truncated_val = int_val & 0xFF
        elif dst_dtype == np.int16:
            word_val = int_val & 0xFFFF
            truncated_val = word_val if word_val < 32768 else word_val - 65536
        elif dst_dtype == np.int32:
            dword_val = int_val & 0xFFFFFFFF
            truncated_val = dword_val if dword_val < 2147483648 else dword_val - 4294967296
        else:
            truncated_val = int_val
        golden_list.append(truncated_val)
    return np.array(golden_list, dtype=dst_dtype).reshape(values.shape)


def clamp_to_range_int(values: np.ndarray, dst_dtype):
    info = ml_dtypes.iinfo(dst_dtype)
    is_int_type = is_sub_int(values.dtype)
    temp_dtype = np.int64 if is_int_type else np.float64
    widened = values.astype(temp_dtype, copy=False)
    return np.clip(widened, info.min, info.max).astype(dst_dtype)


def clamp_to_range_float(values: np.ndarray, dst_dtype):
    info = ml_dtypes.finfo(dst_dtype)
    return np.clip(values, info.min, info.max).astype(dst_dtype)


def apply_valid_shape(values: np.ndarray, valid_shape, dst_dtype):
    vr, vc = valid_shape
    masked = np.zeros_like(values, dtype=dst_dtype)
    masked[:vr, :vc] = values[:vr, :vc]
    return masked


def make_low_precision_quantize_golden(case):
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    shape = case["shape"]
    dst_shape = case.get("dst_shape", shape)
    total = int(np.prod(shape))
    dst_total = int(np.prod(dst_shape))

    if dst_dtype == cases.F8E4M3:
        pairs = f8e4_quantize_pairs()
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), total)
        golden = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), total)
        return values.astype(src_dtype).reshape(shape), golden.reshape(shape)

    if dst_dtype == cases.F8E5M2:
        pairs = f8e5_quantize_pairs()
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), total)
        golden = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), total)
        return values.astype(src_dtype).reshape(shape), golden.reshape(shape)

    if dst_dtype == cases.HIF8:
        pairs = hif8_quantize_pairs()
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), total)
        golden = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), total)
        return values.astype(src_dtype).reshape(shape), golden.reshape(shape)

    if dst_dtype == cases.F4E1M2X2:
        pairs = f4e1_quantize_pairs()
    elif dst_dtype == cases.F4E2M1X2:
        pairs = f4e2_quantize_pairs()
    else:
        raise ValueError(f"unsupported low-precision dst dtype: {dst_dtype}")

    bits = np.resize(np.array([bits for bits, _ in pairs], dtype=np.uint16), total)
    nibbles = np.resize(np.array([nibble for _, nibble in pairs], dtype=np.uint8), total)
    golden = np.array(
        [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, dst_total * 2, 2)],
        dtype=np.uint8,
    )
    return bits.view(ml_dtypes.bfloat16).reshape(shape), golden.reshape(dst_shape)


def generate_golden(case):
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    src_dtype_norm = normalize_dtype(src_dtype)
    dst_dtype_norm = normalize_dtype(dst_dtype)
    shape = case["shape"]
    round_mode = case.get("round_mode")

    if cases.is_low_precision_dtype(dst_dtype):
        input_arr, golden = make_low_precision_quantize_golden(case)
        dst_valid_shape = case.get("dst_valid_shape", case["valid_shape"])
        return input_arr, apply_valid_shape(golden, dst_valid_shape, np.uint8)

    input_arr = make_input(src_dtype, shape)
    converted = convert(input_arr, src_dtype_norm, dst_dtype_norm, round_mode)
    golden = apply_valid_shape(converted, case["valid_shape"], dst_dtype_norm)

    return input_arr, golden


if __name__ == "__main__":
    np.random.seed(19)

    validate_cases(CASES)

    for case in CASES:
        setup_case_rng(case)
        input_arr, golden = generate_golden(case)

        save_case_data(case["name"], {"input": input_arr, "golden": golden})
        src_dtype = case["src_dtype"]
        dst_dtype = case["dst_dtype"]
        src_name = src_dtype.__name__ if isinstance(src_dtype, type) else src_dtype
        dst_name = dst_dtype.__name__ if isinstance(dst_dtype, type) else dst_dtype
        print(
            f"[INFO] gen_data: {case['name']} shape={case['shape']} "
            f"src_dtype={src_name} dst_dtype={dst_name} "
            f"round_mode={case.get('round_mode')}"
        )
