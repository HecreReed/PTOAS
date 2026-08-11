# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

PTO_IR = r"""

module {
  func.func @loop_outer_live(%arg0: !pto.ptr<f16>,
                             %arg1: !pto.ptr<f16>) {
    %pm_c0 = arith.constant 0 : index
    %pm_c1 = arith.constant 1 : index
    %pm_c16 = arith.constant 16 : index
    %pm_c256 = arith.constant 256 : index
    %arg0_view = pto.make_tensor_view %arg0, shape = [%pm_c16, %pm_c256], strides = [%pm_c256, %pm_c1] : !pto.tensor_view<?x?xf16>
    %arg1_view = pto.make_tensor_view %arg1, shape = [%pm_c16, %pm_c256], strides = [%pm_c256, %pm_c1] : !pto.tensor_view<?x?xf16>
    %arg0_part = pto.partition_view %arg0_view, offsets = [%pm_c0, %pm_c0], sizes = [%pm_c16, %pm_c256] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x256xf16>
    %arg1_part = pto.partition_view %arg1_view, offsets = [%pm_c0, %pm_c0], sizes = [%pm_c16, %pm_c256] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x256xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // A buffer that remains live across the loop (used after the loop).
    %outer = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0_part : !pto.partition_tensor_view<16x256xf16>)
             outs(%outer : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    // A loop-local buffer used inside the loop.
    %inner = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    scf.for %i = %c0 to %c4 step %c1 {
      pto.tload ins(%arg0_part : !pto.partition_tensor_view<16x256xf16>)
               outs(%inner : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      pto.tstore ins(%inner : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%arg1_part : !pto.partition_tensor_view<16x256xf16>)
    }

    // Use %outer after the loop to keep it live across the loop.
    pto.tstore ins(%outer : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_part : !pto.partition_tensor_view<16x256xf16>)

    return
  }
}

// Expect a loop, and two planned buffers at distinct offsets.
"""

if __name__ == "__main__":
    print(PTO_IR)
