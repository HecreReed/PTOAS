# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

PTO_IR = r"""

module {
  func.func @tile_liveness(%arg0: !pto.ptr<f16>,
                                      %arg1: !pto.ptr<f16>) {
    %pm_c0 = arith.constant 0 : index
    %pm_c1 = arith.constant 1 : index
    %pm_c16 = arith.constant 16 : index
    %pm_c256 = arith.constant 256 : index
    %arg0_view = pto.make_tensor_view %arg0, shape = [%pm_c16, %pm_c256], strides = [%pm_c256, %pm_c1] : !pto.tensor_view<?x?xf16>
    %arg1_view = pto.make_tensor_view %arg1, shape = [%pm_c16, %pm_c256], strides = [%pm_c256, %pm_c1] : !pto.tensor_view<?x?xf16>
    %arg0_part = pto.partition_view %arg0_view, offsets = [%pm_c0, %pm_c0], sizes = [%pm_c16, %pm_c256] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x256xf16>
    %arg1_part = pto.partition_view %arg1_view, offsets = [%pm_c0, %pm_c0], sizes = [%pm_c16, %pm_c256] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x256xf16>
    %c16 = arith.constant 16 : index

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%arg0_part : !pto.partition_tensor_view<16x256xf16>)
             outs(%b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_part : !pto.partition_tensor_view<16x256xf16>)

    // Using %a should keep %a live; %b must not reuse %a's offset.
    pto.tload ins(%arg0_part : !pto.partition_tensor_view<16x256xf16>)
             outs(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=256, v_row=16, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1_part : !pto.partition_tensor_view<16x256xf16>)
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
