#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 board validation for pto.fdiv_hp (issue #1117).

Compares pto.fdiv_hp results with an INDEPENDENT golden: the exact rational
quotient rounded to f32 with round-to-nearest-even (fractions.Fraction, no
native fdiv involvement; the generator itself is cross-checked against the
host FPU in ptodsl/tests/test_fdiv_hp_algorithm.py). Comparisons are done on
the raw uint32 bit patterns - never allclose.

Coverage:
  - issue #1117 mismatch inputs (torch seed 20260803 vectors, fixed)
  - positive/negative normals, 1/7, 7/3, min/max normal
  - subnormals (min/max and intermediates)
  - rounding midpoints / ties-to-even
  - underflow to subnormal/zero, overflow to +/-Inf
  - +/-0, +/-Inf, NaN: pto.fdiv_hp must fall back to the native arith.divf
    result bit-for-bit (checked against a plain-divf kernel on the same data)

Run (A5 box with torch_npu):
    pytest ptodsl/tests/e2e/test_fdiv_hp.py --target=a5 -v
"""

from __future__ import annotations

from fractions import Fraction
import struct

import numpy as np
import pytest

from ptodsl import pto


# ---------------------------------------------------------------------------
# Independent golden oracle: exact rational division + RNE rounding to f32.
# ---------------------------------------------------------------------------

def _frac_to_f32(q):
    """RNE-round positive Fraction q to f32 exponent+mantissa bits."""
    if q == 0:
        return 0
    e0 = q.numerator.bit_length() - q.denominator.bit_length()
    if e0 >= 0:
        if q.numerator < (q.denominator << e0):
            e0 -= 1
    else:
        if (q.numerator << (-e0)) < q.denominator:
            e0 -= 1
    if e0 <= 23:
        scaled = q * Fraction(1 << (23 - e0), 1)
    else:
        scaled = q * Fraction(1, 1 << (e0 - 23))
    m, d = scaled.numerator, scaled.denominator
    intpart, rem = divmod(m, d)
    twice_rem = 2 * rem
    if twice_rem > d:
        intpart += 1
    elif twice_rem == d and (intpart & 1):
        intpart += 1
    sig = intpart
    if sig >= (1 << 24):
        sig >>= 1
        e0 += 1
    if e0 >= -126:
        field = e0 + 127
        if field >= 255:
            return 0x7F800000
        return (field << 23) | (sig & 0x7FFFFF)
    # Subnormal/underflow: round the exact 23-bit fraction directly (avoid
    # double rounding via the 24-bit sig).
    fracq = q * Fraction(1 << 149, 1)
    base, rem = divmod(fracq.numerator, fracq.denominator)
    twice_rem = 2 * rem
    if twice_rem > fracq.denominator:
        base += 1
    elif twice_rem == fracq.denominator and (base & 1):
        base += 1
    if base >= (1 << 23):
        return 1 << 23
    return base


def golden_bits(a_bits: int, b_bits: int) -> int:
    """Correctly rounded fp32 division result as a uint32 bit pattern."""
    special = (((a_bits >> 23) & 0xFF) == 255 or
               ((b_bits >> 23) & 0xFF) == 255 or
               (a_bits & 0x7FFFFFFF) == 0 or (b_bits & 0x7FFFFFFF) == 0)
    if special:
        return None  # device-native special-value semantics; see special test
    sign = 0x80000000 if ((a_bits ^ b_bits) & 0x80000000) else 0

    def to_frac(bits):
        e = (bits >> 23) & 0xFF
        f = bits & 0x7FFFFF
        if e == 0:
            return Fraction(f, 1 << 149)
        if e >= 150:
            return Fraction(((1 << 23) | f) << (e - 150), 1)
        return Fraction((1 << 23) | f, 1 << (150 - e))

    q = to_frac(a_bits) / to_frac(b_bits)
    return sign | _frac_to_f32(q)


# ---------------------------------------------------------------------------
# Fixed coverage corpus (bit patterns; also the issue #1117 vectors below).
# ---------------------------------------------------------------------------

def _coverage_pairs():
    pairs = [
        (0x3F800000, 0x40E00000),   # 1 / 7
        (0x40E00000, 0x40400000),   # 7 / 3
        (0x3F800000, 0x40400000),   # 1 / 3
        (0x40490FDB, 0x40000000),   # pi / 2
        (0x3F800000, 0x3F800000),   # 1 / 1
        (0x40000000, 0x3F800000),   # 2 / 1
        (0x3F000000, 0x3F800000),   # 0.5 / 1
        (0x00800000, 0x3F800000),   # min normal / 1
        (0x3F800000, 0x00800000),   # 1 / min normal
        (0x7F7FFFFF, 0x3F800000),   # max normal / 1
        (0x3F800000, 0x7F7FFFFF),   # 1 / max normal
        (0x7F7FFFFF, 0x3F7FFFFF),   # overflow -> Inf
        (0xFF7FFFFF, 0x3EFFFFFF),   # -max / 0.5 -> -Inf
        (0x00000001, 0x3F800000),   # min subnormal / 1
        (0x00000001, 0x40000000),   # min subnormal / 2 -> tie to +0
        (0x00000001, 0x40400000),   # min subnormal / 3 -> +0 (sticky)
        (0x00000002, 0x3F800000),   # subnormal / 1
        (0x00000003, 0x3F800000),
        (0x007FFFFF, 0x3F800000),   # max subnormal / 1
        (0x007FFFFF, 0x40000000),
        (0x00800000, 0x40000000),   # min normal / 2 -> max subnormal
        (0x00800000, 0x3F000000),   # min normal / 0.5 -> min normal
        (0x00FFFFFF, 0x3F800000),
        (0x00010000, 0x3F800000),
        (0x3F800001, 0x3F800000),   # (1 + 2^-23) / 1, odd mantissa exact
        (0x3F800001, 0x40000000),   # (1 + 2^-23) / 2 -> tie to even
        (0x3F800002, 0x40000000),   # exact 1 + 2^-23
        (0x3F800000, 0x40000000),   # 1 / 2 exact
        (0x3FA00000, 0x3F000000),   # 1.25 / 0.5
        (0x41200000, 0x40000000),   # 10 / 2 = 5 exact
        (0x41200000, 0x40900000),   # 10 / 4.5
        (0x40490FDB, 0x3E22F983),   # pi / 0.01
        (0x7EFFFFFF, 0x3F7FFFFF),   # near max / (1 - eps) -> Inf edge
        (0x7EFFFFFF, 0x40000000),
        (0x1FFFFFFF, 0x40000001),
        (0x33FFFFFF, 0x40400000),   # near-underflow
        (0x34000000, 0x40400001),
        (0x3727C5AC, 0x3F800000),
        (0x3DCCCCCD, 0x3E4CCCCD),
        (0x3EAAAAAB, 0x3F800000),
        (0x3FC90FDB, 0x3F800000),
        # underflow sweep: 1 / 2^k for k = 127..152 and 3 / 2^k
    ]
    for k in range(127, 153):
        two_pow = Fraction(1, 1 << k)
        b = _frac_to_f32(two_pow)
        if b:
            pairs.append((0x3F800000, b))
    pairs.append((0x00000001, 0x00000002))   # subnormal / subnormal
    pairs.append((0x00000002, 0x00000004))
    pairs.append((0x007FFFFF, 0x00000001))   # max subnormal / min subnormal
    pairs.append((0x00400000, 0x00800000))
    out = []
    for x, y in pairs:
        out.append((x, y))
        out.append((x | 0x80000000, y))
        out.append((x, y | 0x80000000))
        out.append((x | 0x80000000, y | 0x80000000))
    # specials: +-0, +-Inf, NaN (compared against the native divf kernel)
    specials = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000,
                0xFFC00000, 0x7F800001]
    for s in specials:
        out.append((0x3F800000, s))
        out.append((s, 0x3F800000))
        out.append((s, s))
    return out


# ---------------------------------------------------------------------------
# PTODSL kernels (mirror of the issue #1117 repro).
# ---------------------------------------------------------------------------

_MAX_THREADS = 1024


def _fdiv_hp_kernel():
    @pto.simt(max_threads=_MAX_THREADS)
    def div_hp_simt(
        lhs: pto.ptr(pto.f32, "gm"),
        rhs: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm"),
    ):
        tid = pto.get_tid_x()
        pto.stg(pto.fdiv_hp(pto.ldg(lhs, tid), pto.ldg(rhs, tid)), dst, tid)

    @pto.jit(
        name="simt_fdiv_hp_board",
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
    )
    def div_hp_kernel(
        lhs: pto.ptr(pto.f32, "gm"),
        rhs: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm"),
    ):
        div_hp_simt[_MAX_THREADS, 1, 1](lhs, rhs, dst)

    return div_hp_kernel


def _divf_kernel():
    @pto.simt(max_threads=_MAX_THREADS)
    def div_simt(
        lhs: pto.ptr(pto.f32, "gm"),
        rhs: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm"),
    ):
        tid = pto.get_tid_x()
        pto.stg(pto.ldg(lhs, tid) / pto.ldg(rhs, tid), dst, tid)

    @pto.jit(
        name="simt_divf_board",
        kernel_kind="vector",
        target="a5",
        backend="vpto",
        mode="explicit",
    )
    def div_kernel(
        lhs: pto.ptr(pto.f32, "gm"),
        rhs: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm"),
    ):
        div_simt[_MAX_THREADS, 1, 1](lhs, rhs, dst)

    return div_kernel


def _run_kernel(torch, handle, a_np, b_np):
    n = a_np.shape[0]
    a = torch.from_numpy(a_np).to("npu:0")
    b = torch.from_numpy(b_np).to("npu:0")
    out = torch.empty_like(a)
    stream = torch.npu.current_stream()._as_parameter_
    compiled = handle.compile()
    compiled[1, stream](a.data_ptr(), b.data_ptr(), out.data_ptr())
    torch.npu.synchronize()
    return out.cpu().numpy().view(np.uint32)


def _issue1117_vectors(torch):
    """Fixed issue #1117 mismatch inputs (exact repro code path)."""
    n = 256
    torch.manual_seed(20260803)
    lhs_host = torch.rand(n, dtype=torch.float32) * 2
    rhs_host = torch.rand(n, dtype=torch.float32) * 128 + 0.25
    return lhs_host.numpy(), rhs_host.numpy()


def _bits_of(f32arr):
    return f32arr.astype(np.float32).view(np.uint32).astype(np.int64)


def _run_board(torch, target_arch):
    if target_arch != "a5":
        pytest.skip("pto.fdiv_hp is an A5-only op")
    # assemble the full corpus
    fixed = np.array([x for x, y in _coverage_pairs()], dtype=np.float32)
    pairs = _coverage_pairs()
    a_fixed = np.array([x for x, y in pairs], dtype=np.uint32).view(np.float32)
    b_fixed = np.array([y for x, y in pairs], dtype=np.uint32).view(np.float32)
    a_iss, b_iss = _issue1117_vectors(torch)
    a_all = np.concatenate([a_fixed, a_iss.astype(np.float32)])
    b_all = np.concatenate([b_fixed, b_iss.astype(np.float32)])

    a_bits = a_all.view(np.uint32).astype(np.int64)
    b_bits = b_all.view(np.uint32).astype(np.int64)

    fdiv_hp_out = _run_kernel(torch, _fdiv_hp_kernel(), a_all, b_all)
    divf_out = _run_kernel(torch, _divf_kernel(), a_all, b_all)

    mismatches = []
    special_mismatches = []
    for i in range(a_bits.shape[0]):
        got_bits = int(fdiv_hp_out[i])
        want = golden_bits(int(a_bits[i]), int(b_bits[i]))
        if want is None:
            # special inputs: pto.fdiv_hp must match the native divf result
            if got_bits != int(divf_out[i]):
                special_mismatches.append(
                    (i, hex(int(a_bits[i])), hex(int(b_bits[i])),
                     hex(got_bits), hex(int(divf_out[i]))))
        elif got_bits != want:
            mismatches.append(
                (i, hex(int(a_bits[i])), hex(int(b_bits[i])),
                 hex(got_bits), hex(want)))

    if mismatches or special_mismatches:
        for row in mismatches[:10]:
            print("FDIV_HP MISMATCH", row)
        for row in special_mismatches[:10]:
            print("SPECIAL FALLBACK MISMATCH", row)
        raise AssertionError(
            f"fdiv_hp bit mismatches: {len(mismatches)}, "
            f"special fallback mismatches: {len(special_mismatches)}")


@pytest.mark.require_npu
def test_fdiv_hp_bit_exact(torch, target_arch, backend):
    _run_board(torch, target_arch)
