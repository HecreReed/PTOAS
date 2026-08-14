#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Software model of the pto.fdiv_hp expansion (issue #1117).

fdiv_hp_model() is a bit-exact Python mirror of ExpandFDivHpPattern in
lib/PTO/Transforms/VPTOExpandWrapperOps.cpp: it reproduces the same i32
bit-cast / CLZ / base-2^7 long-division / guard-round-sticky sequence that the
MLIR expansion emits. oracle_fdiv_bits() is an independent golden generator:
it computes the quotient as an exact Fraction and rounds to f32 with
round-to-nearest-even, and is itself cross-checked against the host FPU
(double division of two f32-representable values is exact; the double->f32
rounding via struct.pack is the hardware's own RNE conversion).

The test asserts bit-exact equality between the model and the oracle over a
corpus that includes the issue #1117 mismatch inputs, the coverage list
specified for the A5 board validation (positive/negative normals, 1/7, 7/3,
min/max normal, subnormals, rounding midpoints/ties, underflow, overflow,
special values) and randomized + exhaustive significand sweeps.

Run: pytest ptodsl/tests/test_fdiv_hp_algorithm.py   (no NPU required)
"""

from fractions import Fraction
import random
import struct

U32 = 0xFFFFFFFF


def u32(x):
    return x & U32


def f32_bits(x):
    return struct.unpack(">I", struct.pack(">f", x))[0]


def bits_to_f32(b):
    return struct.unpack(">f", struct.pack(">I", b))[0]


def double_to_f32_bits(d):
    """Independent double->f32 RNE rounding using host conversion."""
    return struct.unpack(">I", struct.pack(">f", d))[0]


def native_div_bits(a, b):
    """IEEE f32 division via double arithmetic + host double->f32 RNE rounding.
    For two f32-representable operands the double quotient is exact (<= 48
    significant bits), so this mirrors a correctly-rounded f32 division."""
    x = bits_to_f32(a)
    y = bits_to_f32(b)
    import math
    ax, ay = abs(x), abs(y)
    inf = float("inf")
    nan = float("nan")
    if (x != x) or (y != y):
        return f32_bits(nan)
    if ax == inf or ay == inf:
        if ax == inf and ay == inf:
            return f32_bits(nan)
        if ax == inf:
            return f32_bits(inf if (x * y > 0) else -inf)
        return f32_bits(0.0 if (x * y > 0) else -0.0)
    if ay == 0.0:
        if ax == 0.0:
            return f32_bits(nan)
        return f32_bits(inf if (x * y > 0) else -inf)
    if ax == 0.0:
        # IEEE: result sign is XOR of operand signs (also for +/-0)
        neg = (math.copysign(1.0, x) * math.copysign(1.0, y)) < 0
        return f32_bits(-0.0 if neg else 0.0)
    d = x / y
    if d > 3.4028234663852886e38 or d < -3.4028234663852886e38:
        return f32_bits(inf if d > 0 else -inf)
    if d != d:
        return f32_bits(nan)
    return double_to_f32_bits(d)


def _frac_to_f32(q):
    """RNE-round positive Fraction q to f32 exponent+mantissa bits (sign bits
    are added by the caller). Handles normal, subnormal, underflow, overflow."""
    if q == 0:
        return 0
    e0 = q.numerator.bit_length() - q.denominator.bit_length()
    # bit_length diff can overestimate floor(log2(q)) by one
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
    # Subnormal / underflow: round the exact 23-bit fraction
    # frac = value * 2^149 directly with RNE. (Rounding the 24-bit sig first
    # and then shifting would double-round and corrupt 2^-149 ties, so the
    # full-precision fraction is used here.)
    fracq = q * Fraction(1 << 149, 1)
    base, rem = divmod(fracq.numerator, fracq.denominator)
    twice_rem = 2 * rem
    if twice_rem > fracq.denominator:
        base += 1
    elif twice_rem == fracq.denominator and (base & 1):
        base += 1
    if base >= (1 << 23):  # rounded up to the minimum normal 2^-126
        return 1 << 23
    return base


def clz32(x):
    # 16/24/28/30/31 checks with 16/8/4/2/1 shifts; mirrors the C++ expansion.
    n = 0
    for chk, upd in ((16, 16), (24, 8), (28, 4), (30, 2), (31, 1)):
        if u32(x >> chk) == 0:
            n += upd
            x = u32(x << upd)
    return n


def fdiv_hp_model(a_bits, b_bits):
    """Bit-exact mirror of ExpandFDivHpPattern."""
    lhs, rhs = u32(a_bits), u32(b_bits)
    ea = (lhs >> 23) & 0xFF
    eb = (rhs >> 23) & 0xFF
    fa = lhs & 0x7FFFFF
    fb = rhs & 0x7FFFFF
    special = (ea == 255) or (eb == 255) or ((ea | fa) == 0) or ((eb | fb) == 0)
    if special:
        return native_div_bits(lhs, rhs)
    signBits = u32((lhs ^ rhs) & 0x80000000)
    clzA = clz32(fa)
    subA = (ea == 0) and (fa != 0)
    ma = u32(fa << (clzA - 8)) if subA else (0x800000 | fa)
    eaExp = -(clzA + 141) if subA else (ea - 150)
    clzB = clz32(fb)
    subB = (eb == 0) and (fb != 0)
    mb = u32(fb << (clzB - 8)) if subB else (0x800000 | fb)
    ebExp = -(clzB + 141) if subB else (eb - 150)
    e0 = eaExp - ebExp
    if ma < mb:
        e = e0 - 1
        maAdj = ma << 1
    else:
        e = e0
        maAdj = ma
    # (maAdj << 26) in base 2^7. maAdj can be 25 bits after the ratio
    # normalization shift (Ma <<= 1), so the numerator takes 9 digits; digit i
    # holds N bits [7i, 7i+6] = maAdj bits [7i-26, 7i+6-26].
    digits = [
        0,                       # i = 8: maAdj bits [30..36] -> 0
        (maAdj >> 23) & 0x7F,    # i = 7: bits [23..29]
        (maAdj >> 16) & 0x7F,    # i = 6: bits [16..22]
        (maAdj >> 9) & 0x7F,     # i = 5: bits [9..15]
        (maAdj >> 2) & 0x7F,     # i = 4: bits [2..8]
        (maAdj & 3) << 5,        # i = 3: bits [0..1]
        0, 0, 0,                 # i = 2..0: none
    ]
    r = 0
    raw = 0
    for d in digits:
        r = u32((r << 7) | d)
        q = r // mb
        r = u32(r - q * mb)
        raw = u32((raw << 7) | q)
    rem = r
    sig = raw >> 3
    guard = (raw >> 2) & 1
    round_bit = (raw >> 1) & 1
    sticky = (raw & 1) | (1 if rem != 0 else 0)
    inc = guard & (round_bit | sticky | (sig & 1))
    sig += inc
    if sig == (1 << 24):
        sig >>= 1
        e += 1
    if e >= 128:
        return signBits | 0x7F800000
    if e >= -126:
        field = e + 127
        return signBits | (field << 23) | (sig & 0x7FFFFF)
    if e >= -151:
        shift = e + 152
        s = 26 - shift
        sigSub = raw >> (s + 3)
        guardSub = (raw >> (s + 2)) & 1
        roundSub = (raw >> (s + 1)) & 1
        maskSub = (1 << s) - 1
        stickySub = ((1 if (raw & maskSub) != 0 else 0) |
                     (1 if rem != 0 else 0) | ((raw >> s) & 1))
        incSub = guardSub & (roundSub | stickySub | (sigSub & 1))
        sigSub += incSub
        if sigSub == (1 << 23):
            return signBits | (1 << 23)
        return signBits | sigSub
    return signBits


def oracle_fdiv_bits(a_bits, b_bits):
    """Independent oracle: exact rational division + RNE to f32."""
    special = (((a_bits >> 23) & 0xFF) == 255 or
               ((b_bits >> 23) & 0xFF) == 255 or
               (a_bits & 0x7FFFFFFF) == 0 or (b_bits & 0x7FFFFFFF) == 0)
    if special:
        return native_div_bits(a_bits, b_bits)
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


def rand_f32_bits(rng):
    e = rng.randint(1, 254)
    f = rng.randint(0, (1 << 23) - 1)
    s = rng.randint(0, 1)
    return (s << 31) | (e << 23) | f


def rand_subnormal_bits(rng):
    f = rng.randint(1, (1 << 23) - 1)
    s = rng.randint(0, 1)
    return (s << 31) | f


def main():
    rng = random.Random(1117)
    cases = []

    # Deterministic coverage (user-required cases).
    fixed = [
        (0x3F800000, 0x40E00000),   # 1 / 7
        (0x40E00000, 0x40400000),   # 7 / 3
        (0x3F800000, 0x40400000),   # 1 / 3
        (0x40490FDB, 0x40000000),   # pi / 2
        (0x3FC90FDB, 0x3F800000),
        (0x3EAAAAAB, 0x3F800000),   # 1/3 ugly repr
        (0x3DCCCCCD, 0x3E4CCCCD),
        (0x00800000, 0x3F800000),   # min normal / 1
        (0x3F800000, 0x00800000),   # 1 / min normal -> huge normal
        (0x7F7FFFFF, 0x3F800000),   # max normal / 1
        (0x3F800000, 0x7F7FFFFF),
        (0x7F7FFFFF, 0x3F7FFFFF),   # max/(1-eps) -> Inf
        (0xFF7FFFFF, 0x3EFFFFFF),   # -max/0.5 -> -Inf
        (0x7F7FFFFF, 0x40000000),
        (0x00000001, 0x3F800000),   # min subnormal / 1
        (0x00000001, 0x40000000),   # min subnormal / 2 -> tie to 0
        (0x00000001, 0x40400000),   # min subnormal / 3 -> sticky 0
        (0x00000002, 0x3F800000),
        (0x00000003, 0x3F800000),
        (0x007FFFFF, 0x3F800000),   # max subnormal / 1
        (0x00400000, 0x3F800000),
        (0x00800000, 0x40000000),   # min normal / 2 -> max subnormal
        (0x00800000, 0x3F000000),   # min normal / 0.5 -> min normal
        (0x007FFFFF, 0x40000000),
        (0x007FFFFF, 0x40400000),
        (0x00FFFFFE, 0x3F800000),
        (0x00FFFFFF, 0x40000000),
        (0x00FFFFFF, 0x3F800000),
        (0x00010000, 0x3F800000),
        (0x00000001, 0x3E800000),   # min subnormal / 0.25 -> subnormal result
        (0x00000001, 0x3DCCCCCD),
        (0x3F800001, 0x3F800000),   # (1+2^-23)/1 exact odd mantissa
        (0x3F800001, 0x40000000),   # (1+2^-23)/2 -> tie, odd -> up
        (0x3F800000, 0x40000000),   # 1/2 exact
        (0x3F800002, 0x40000000),   # (1+2^-22)/2 -> exact 1+2^-23
        (0x3FA00000, 0x3F000000),   # 1.25/0.5 = 2.5
        (0x40400000, 0x40080000),
        (0x41200000, 0x40000000),   # 10/2 = 5
        (0x41200000, 0x40900000),
        (0x49742400, 0x40000000),
        (0x33FFFFFF, 0x40400000),   # near-underflow subnormal-ish
        (0x34000000, 0x40400001),
        (0x3727C5AC, 0x3F800000),
        (0x7EFFFFFF, 0x3F7FFFFF),   # near-max/0.999... -> Inf edge
        (0x7EFFFFFF, 0x40000000),
        (0x1FFFFFFF, 0x40000001),
        (0x40490FDB, 0x3E22F983),   # pi / 0.01
    ]
    for x, y in fixed:
        cases.append((x, y))
        cx = x | 0x80000000
        cy = y | 0x80000000
        cases.append((cx, y))
        cases.append((x, cy))
        cases.append((cx, cy))

    # Sweep subnormal result paths: 1 / 2^k for k = 127..151 and 3/2^k.
    for k in range(126, 152):
        b = _frac_to_f32(Fraction(1, 1 << k))
        b |= 0x3F800000 & 0  # keep b as-is (already f32 bits)
        cases.append((0x3F800000, b))
        b3 = _frac_to_f32(Fraction(3, 1 << k))
        if k >= 127:
            cases.append((0x40400000, b3))
    # Sweep overflow edge: max_normal / 2^-k
    for k in range(0, 5):
        b = _frac_to_f32(Fraction(1, 1 << k))
        cases.append((0x7F7FFFFF, b))
    # Midpoint/ties: a and b chosen so exact quotient ends at odd/even ties.
    tie_cases = [
        (0x3F800001, 0x40000000),
        (0x40000001, 0x40000000),   # 3.0000002/2
        (0x40400001, 0x40800000),   # 3.0000002/4 = 0.75... 
        (0x3FA00001, 0x3F000000),
        (0x40A00001, 0x40C00000),
        (0x41400001, 0x3F800000),
        (0x41D55555, 0x41400000),
    ]
    for x, y in tie_cases:
        cases.append((x, y))

    # Random normals (wide exponent spread to exercise all branches).
    for _ in range(25000):
        cases.append((rand_f32_bits(rng), rand_f32_bits(rng)))

    # Random subnormal dividend/divisor mixes.
    for _ in range(6000):
        a = rand_subnormal_bits(rng) if rng.random() < 0.5 else rand_f32_bits(rng)
        b = rand_subnormal_bits(rng) if rng.random() < 0.5 else rand_f32_bits(rng)
        cases.append((a, b))

    # Exhaustive significand sweep at selected exponent pairs (validates the
    # long-division digit windows and GRS over all 64-digit neighborhoods).
    for ea in (0x7E, 0x80, 0x00, 0x81):
        for eb in (0x7F, 0x80, 0x40, 0x00):
            for fa in range(0x600000, 0x600000 + 256):
                for fb in range(0x400000, 0x400000 + 32):
                    cases.append(((ea << 23) | fa, (eb << 23) | fb))

    # --- model vs oracle ---
    mismatches = 0
    checked = 0
    for a, b in cases:
        a, b = u32(a), u32(b)
        if ((a >> 23) & 0xFF) == 255 or ((b >> 23) & 0xFF) == 255:
            continue
        if (a & 0x7FFFFFFF) == 0 or (b & 0x7FFFFFFF) == 0:
            continue
        checked += 1
        try:
            got = fdiv_hp_model(a, b)
        except Exception as exc:
            print("CRASH INPUT", hex(a), hex(b), "exc", exc)
            raise
        want = oracle_fdiv_bits(a, b)
        if got != want:
            mismatches += 1
            if mismatches <= 15:
                print("MODEL MISMATCH", hex(a), "/", hex(b),
                      "model", hex(got), "oracle", hex(want),
                      "model~", bits_to_f32(got), "oracle~", bits_to_f32(want))
    print("model vs oracle: checked", checked, "mismatches", mismatches)

    # --- oracle vs host-FPU cross-check (validates the oracle itself) ---
    oracle_bad = 0
    cc = 0
    for a, b in cases[:60000]:
        a, b = u32(a), u32(b)
        if ((a >> 23) & 0xFF) == 255 or ((b >> 23) & 0xFF) == 255:
            continue
        if (a & 0x7FFFFFFF) == 0 or (b & 0x7FFFFFFF) == 0:
            continue
        cc += 1
        got = oracle_fdiv_bits(a, b)
        want = native_div_bits(a, b)
        if got != want:
            oracle_bad += 1
            if oracle_bad <= 5:
                print("ORACLE MISMATCH", hex(a), hex(b), hex(got), hex(want))
    print("oracle vs host FPU: checked", cc, "mismatches", oracle_bad)

    # Also verify division-by-zero / zero behavior of the model matches native.
    special_ok = True
    for a, b, want in [
        (0x3F800000, 0x00000000, None),   # 1/0 -> +Inf
        (0xBF800000, 0x00000000, None),   # -1/0 -> -Inf
        (0x00000000, 0x3F800000, 0x00000000),  # 0/1 -> +0
        (0x80000000, 0x3F800000, 0x80000000),  # -0/1 -> -0
        (0x7F800000, 0x3F800000, 0x7F800000),  # Inf/1 -> +Inf
        (0xFF800000, 0x3F800000, 0xFF800000),  # -Inf/1 -> -Inf
        (0x3F800000, 0x7F800000, 0x00000000),  # 1/Inf -> +0
        (0x3F800000, 0xFF800000, 0x80000000),  # 1/-Inf -> -0
        (0x7F800000, 0x7F800000, None),        # Inf/Inf -> NaN
        (0x7FC00000, 0x3F800000, None),        # NaN/1 -> NaN
    ]:
        got = fdiv_hp_model(a, b)
        if want is None:
            ok = ((got >> 23) & 0xFF) == 255
            if not ok:
                print("SPECIAL BAD", hex(a), hex(b), hex(got), "want raw-NaN")
            special_ok = special_ok and ok
        elif got != want:
            special_ok = False
            print("SPECIAL BAD", hex(a), hex(b), hex(got), "want", hex(want))
    print("special fallback: ok =", special_ok)

    if mismatches or oracle_bad or not special_ok:
        raise SystemExit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
