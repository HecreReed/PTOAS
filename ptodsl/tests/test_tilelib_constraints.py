# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Constraint-driven selection test for the reduction op pto.tcolmax.

tcolmax is legal only for row-major / none-box operands with a 1-row output
(dst_valid_shape[0] == 1). Selection must accept the legal case and reject the others.
"""

import unittest

from ptodsl.tilelib import ScalarType, TileSpec, VectorSpec, ViewSpec, select
from ptodsl.tilelib.constraints import build_context
from ptodsl.tilelib.registry import NoMatchingTemplate

F32 = ScalarType("f32")


def _specs(*, dst_valid=(1, 64), dst_blayout="row_major", dst_slayout="none_box"):
    src = TileSpec(shape=(8, 64), dtype=F32, valid_shape=(8, 64))
    dst = TileSpec(shape=(8, 64), dtype=F32, valid_shape=dst_valid,
                   b_layout=dst_blayout, s_layout=dst_slayout)
    return {"src": src, "dst": dst}


class TileLibConstraintTest(unittest.TestCase):
    def test_legal_colmax_selected(self):
        chosen = select("pto.tcolmax", "a5", _specs())
        self.assertEqual(chosen.name, "template_tcolmax")

    def test_rejected_when_dst_not_single_row(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tcolmax", "a5", _specs(dst_valid=(8, 64)))

    def test_rejected_when_not_row_major(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tcolmax", "a5", _specs(dst_blayout="col_major"))

    def test_rejected_when_not_none_box(self):
        with self.assertRaises(NoMatchingTemplate):
            select("pto.tcolmax", "a5", _specs(dst_slayout="row_major"))

    def test_legal_colmax_renders_structured_mlir(self):
        chosen = select("pto.tcolmax", "a5", _specs())
        mlir = chosen.specialize(**_specs()).mlir_text()
        for op in ("pto.tile_valid_rows", "memref.subview", "scf.for", "iter_args",
                   "pto.vmax", "pto.vsts", "pto.tilelang.instance"):
            self.assertIn(op, mlir)
        self.assertNotIn("pto.castptr", mlir)

    def test_context_tracks_view_and_vector_operands(self):
        context = build_context(
            {
                "src": ViewSpec(
                    shape=(16, 64),
                    dtype=F32,
                    memory_space="gm",
                    strides=(64, 1),
                    layout="ND",
                ),
                "aux": VectorSpec(shape=(4,), dtype=ScalarType("i32")),
            },
            "a5",
            "pto.example",
        )

        self.assertEqual(context["operand_kinds"], ("view", "vector"))
        self.assertEqual(context["src_shape"], (16, 64))
        self.assertEqual(context["src_strides"], (64, 1))
        self.assertEqual(context["src_memory_space"], "gm")
        self.assertEqual(context["src_layout"], "ND")
        self.assertEqual(context["aux_shape"], (4,))
        self.assertEqual(context["aux_size"], 4)


if __name__ == "__main__":
    unittest.main()
