# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Phase-4 breadth test: each ported elementwise op selects + renders to the structured
abstraction, using the right vector op."""

import unittest

from ptodsl.tilelib import ScalarType, TileSpec, select

# op -> (expected template name, expected vector op in the rendered MLIR)
ELEMENTWISE = {
    "pto.tsub": ("template_tsub", "pto.vsub"),
    "pto.tmul": ("template_tmul", "pto.vmul"),
    "pto.tmax": ("template_tmax", "pto.vmax"),
    "pto.tmin": ("template_tmin", "pto.vmin"),
    "pto.tdiv": ("template_tdiv", "pto.vdiv"),
}

# Structured abstraction every elementwise template must preserve.
SHARED_OPS = ["pto.tile_buf_addr", "memref.subview", "scf.for", "iter_args",
              "pto.plt_b32", "pto.vlds", "pto.vsts", "pto.tilelang.instance"]


def _f32_specs():
    spec = TileSpec(shape=(8, 64), dtype=ScalarType("f32"))
    return {"src0": spec, "src1": spec, "dst": spec}


class TileLibElementwiseTest(unittest.TestCase):
    def test_each_op_selects_and_renders(self):
        for op, (name, vop) in ELEMENTWISE.items():
            with self.subTest(op=op):
                descriptor = select(op, "a5", _f32_specs(), candidate_id=name)
                self.assertEqual(descriptor.name, name)

                mlir = descriptor.specialize(**_f32_specs()).mlir_text()
                self.assertIn(vop, mlir)                 # the op's own vector instruction
                for shared in SHARED_OPS:
                    self.assertIn(shared, mlir)
                self.assertNotIn("pto.castptr", mlir)    # structured, not bare-pointer


if __name__ == "__main__":
    unittest.main()
