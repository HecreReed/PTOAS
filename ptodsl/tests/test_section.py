#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Focused tracing coverage for explicit physical section hints."""

from ptodsl import pto
from ptodsl._context import make_context
from ptodsl._tracing.active import current_session
from ptoas.mlir.ir import Module


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def explicit_sections_probe():
    event_id = pto.const(1, dtype=pto.i32)
    with pto.section("cube"):
        pto.set_flag("MTE2", "S", event_id=0)
        pto.wait_flag("S", "MTE2", event_id=event_id)
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=0)
        pto.set_flag("S", "MTE2", event_id=event_id)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def duplicate_section_probe():
    with pto.section("cube"):
        pass
    with pto.section("cube"):
        pass


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def nested_section_probe():
    with pto.section("cube"):
        with pto.section("vector"):
            pass


@pto.jit(
    target="a5",
    mode="explicit",
    kernel_kind="cube",
    ast_rewrite=False,
)
def kernel_kind_section_probe():
    with pto.section("cube"):
        pass


@pto.jit(target="a5", ast_rewrite=False)
def auto_mode_section_probe():
    with pto.section("vector"):
        pass


@pto.jit(
    target="a5",
    mode="explicit",
    kernel_kind=None,
    ast_rewrite=False,
)
def unspecified_kernel_kind_section_probe():
    with pto.section("vector"):
        pto.wait_flag("MTE2", "S", event_id=0)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def recovered_section_probe():
    try:
        with pto.section("cube"):
            raise RuntimeError("abort authored section")
    except RuntimeError:
        pass
    with pto.section("cube"):
        pto.set_flag("MTE2", "S", event_id=0)


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def loop_subkernel_then_section_probe():
    c0 = pto.const(0, dtype=pto.i32)
    c1 = pto.const(1, dtype=pto.i32)
    with pto.for_(c0, c1, step=c1):
        session = current_session()
        with session.enter_subkernel_body("cube", "loop_cube", "a5"):
            pass
    with pto.section("cube"):
        pass


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def section_then_subkernel_probe():
    with pto.section("cube"):
        pass
    session = current_session()
    with session.enter_subkernel_body("cube", "late_cube", "a5"):
        pass


@pto.jit(target="a5", mode="explicit", ast_rewrite=False)
def section_inside_subkernel_probe():
    session = current_session()
    with session.enter_subkernel_body("cube", "outer_cube", "a5"):
        with pto.section("vector"):
            pass


def _expect_raises(exc_type, callback, message):
    try:
        callback()
    except exc_type as exc:
        assert message in str(exc), str(exc)
        return
    raise AssertionError(f"expected {exc_type.__name__} containing {message!r}")


def main() -> None:
    text = explicit_sections_probe.compile().mlir_text()
    assert text.count("pto.section.cube {") == 1
    assert text.count("pto.section.vector {") == 1
    assert "pto.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]" in text
    assert "pto.wait_flag_dyn[<PIPE_S>, <PIPE_MTE2>" in text
    assert "pto.wait_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]" in text
    assert "pto.set_flag_dyn[<PIPE_S>, <PIPE_MTE2>" in text
    with make_context() as context:
        module = Module.parse(text, context)
        module.operation.verify()

    recovered_text = recovered_section_probe.compile().mlir_text()
    assert recovered_text.count("pto.section.cube {") == 1
    assert "pto.set_flag[<PIPE_MTE2>, <PIPE_S>, <EVENT_ID0>]" in recovered_text

    _expect_raises(TypeError, lambda: pto.section(1), "expects 'cube' or 'vector'")
    _expect_raises(ValueError, lambda: pto.section("scalar"), "expects 'cube' or 'vector'")
    _expect_raises(ValueError, lambda: pto.section("CUBE"), "expects 'cube' or 'vector'")
    _expect_raises(
        RuntimeError,
        lambda: duplicate_section_probe.compile(),
        "each physical section kind may appear at most once in a function",
    )
    _expect_raises(
        RuntimeError,
        lambda: nested_section_probe.compile(),
        "nested pto.section() scopes are not allowed",
    )
    _expect_raises(
        RuntimeError,
        lambda: loop_subkernel_then_section_probe.compile(),
        "each physical section kind may appear at most once in a function",
    )
    _expect_raises(
        RuntimeError,
        lambda: section_then_subkernel_probe.compile(),
        "each physical section kind may appear at most once in a function",
    )
    _expect_raises(
        RuntimeError,
        lambda: section_inside_subkernel_probe.compile(),
        "pto.section() is not allowed inside a cube or simd subkernel body",
    )
    _expect_raises(
        RuntimeError,
        lambda: kernel_kind_section_probe.compile(),
        "cannot be combined with explicit @pto.jit(kernel_kind=...)",
    )
    _expect_raises(
        RuntimeError,
        lambda: auto_mode_section_probe.compile(),
        "only available in @pto.jit(mode=\"explicit\")",
    )
    _expect_raises(
        RuntimeError,
        lambda: pto.section("cube").__enter__(),
        "may only be used while tracing",
    )

    unspecified_text = unspecified_kernel_kind_section_probe.compile().mlir_text()
    assert "pto.section.vector {" in unspecified_text


if __name__ == "__main__":
    main()
