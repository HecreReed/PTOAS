#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL compile test for pto.fdiv_hp (issue #1117).

Verifies that pto.fdiv_hp() frontend maps to the dedicated pto.fdiv_hp op with
the expected f32, f32 -> f32 result type, and that plain Python / is NOT
changed (it still lowers to arith.divf).
"""

from ptodsl import pto


def _compile_fdiv_hp():
    @pto.simt(max_threads=8)
    def div_hp_simt(
        lhs: pto.ptr(pto.f32, "gm"),
        rhs: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm"),
    ):
        tid = pto.get_tid_x()
        pto.stg(pto.fdiv_hp(pto.ldg(lhs, tid), pto.ldg(rhs, tid)), dst, tid)

    @pto.jit(
        name="simt_fdiv_hp_probe",
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
        div_hp_simt[8, 1, 1](lhs, rhs, dst)

    return div_hp_kernel.compile().mlir_text()


def _compile_plain_div():
    """Plain Python / must still lower to arith.divf (unchanged)."""

    @pto.simt(max_threads=8)
    def div_simt(
        lhs: pto.ptr(pto.f32, "gm"),
        rhs: pto.ptr(pto.f32, "gm"),
        dst: pto.ptr(pto.f32, "gm"),
    ):
        tid = pto.get_tid_x()
        pto.stg(pto.ldg(lhs, tid) / pto.ldg(rhs, tid), dst, tid)

    @pto.jit(
        name="simt_div_plain_probe",
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
        div_simt[8, 1, 1](lhs, rhs, dst)

    return div_kernel.compile().mlir_text()


def main():
    text = _compile_fdiv_hp()
    if "pto.fdiv_hp" not in text:
        raise AssertionError("pto.fdiv_hp op missing from generated IR")
    if ": f32, f32 -> f32" not in text:
        raise AssertionError("pto.fdiv_hp must produce scalar f32 -> f32")

    plain = _compile_plain_div()
    if "pto.fdiv_hp" in plain:
        raise AssertionError("plain Python / must not emit pto.fdiv_hp")
    if "arith.divf" not in plain:
        raise AssertionError("plain Python / must still emit arith.divf")
    print("fdiv_hp compile test OK")


if __name__ == "__main__":
    main()
