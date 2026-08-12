#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Script to generate tcvt.pto"""

import cases

def gen_rmode_attr(rmode):
    return f"#pto<round_mode {rmode}>"

def gen_kernel(case, idx=0):
    src_name = cases.dtype_name(case["src_dtype"])
    dst_name = cases.dtype_name(case["dst_dtype"])
    src_dtype = cases.pto_dtype_name(case["src_dtype"])
    dst_dtype = cases.pto_dtype_name(case["dst_dtype"])
    rows, cols = case["shape"]
    v_rows, v_cols = case["valid_shape"]
    dst_rows, dst_cols = case.get("dst_shape", case["shape"])
    dst_v_rows, dst_v_cols = case.get("dst_valid_shape", case["valid_shape"])
    
    kernel_name = f"TCVT_{case['name']}"
    
    rmode = "RINT"
    rmode_command = "default RINT"
    if "round_mode" in case:
        rmode = case['round_mode']
        if rmode != "RINT":
            rmode_command = f"explicit {rmode}"
    
    src_stride = rows * cols
    dst_stride = dst_rows * dst_cols
    
    tile_valid = "" if v_rows == rows and v_cols == cols else f", valid={v_rows}x{v_cols}"
    dst_tile_valid = "" if dst_v_rows == dst_rows and dst_v_cols == dst_cols else f", valid={dst_v_rows}x{dst_v_cols}"
    tile_src = f"!pto.tile_buf<vec, {rows}x{cols}x{src_dtype}{tile_valid}>"
    tile_dst = f"!pto.tile_buf<vec, {dst_rows}x{dst_cols}x{dst_dtype}{dst_tile_valid}>"

    const_vals = sorted(set([0, 1, rows, cols, v_rows, v_cols, dst_rows, dst_cols, dst_v_rows, dst_v_cols, src_stride, dst_stride]))
    longest_const = len(str(const_vals[-1]))
    const_defs = [f"    %c{i:<{longest_const}} = arith.constant {i:<{longest_const}} : index" for i in const_vals]
    
    lines = [
        f"  // Case {idx}: {src_dtype} -> {dst_dtype}, {rmode_command}",
        f"  func.func @{kernel_name}(%src_ptr: !pto.ptr<{src_dtype}>, %dst_ptr: !pto.ptr<{dst_dtype}>) attributes {{pto.entry, pto.kernel}} {{",
    ]
    lines.extend(const_defs)
    lines.extend([
        "",
        f"    %src_view = pto.make_tensor_view %src_ptr,",
        f"      shape = [%c1, %c1, %c1, %c{rows}, %c{cols}],",
        f"      strides = [%c{src_stride}, %c{src_stride}, %c{src_stride}, %c{cols}, %c1]",
        f"      : !pto.tensor_view<1x1x1x{rows}x{cols}x{src_dtype}>",
        f"    %dst_view = pto.make_tensor_view %dst_ptr,",
        f"      shape = [%c1, %c1, %c1, %c{dst_rows}, %c{dst_cols}],",
        f"      strides = [%c{dst_stride}, %c{dst_stride}, %c{dst_stride}, %c{dst_cols}, %c1]",
        f"      : !pto.tensor_view<1x1x1x{dst_rows}x{dst_cols}x{dst_dtype}>",
        "",
        f"    %src_part = pto.partition_view %src_view,",
        f"      offsets = [%c0, %c0, %c0, %c0, %c0],",
        f"      sizes = [%c1, %c1, %c1, %c{v_rows}, %c{v_cols}]",
        f"      : !pto.tensor_view<1x1x1x{rows}x{cols}x{src_dtype}> -> !pto.partition_tensor_view<1x1x1x{v_rows}x{v_cols}x{src_dtype}>",
        f"    %dst_part = pto.partition_view %dst_view,",
        f"      offsets = [%c0, %c0, %c0, %c0, %c0],",
        f"      sizes = [%c1, %c1, %c1, %c{dst_v_rows}, %c{dst_v_cols}]",
        f"      : !pto.tensor_view<1x1x1x{dst_rows}x{dst_cols}x{dst_dtype}> -> !pto.partition_tensor_view<1x1x1x{dst_v_rows}x{dst_v_cols}x{dst_dtype}>",
        "",
        f"    %src = pto.alloc_tile",
        f"      : {tile_src}",
        f"    %dst = pto.alloc_tile",
        f"      : {tile_dst}",
        "",
        f"    pto.tload ins(%src_part : !pto.partition_tensor_view<1x1x1x{v_rows}x{v_cols}x{src_dtype}>)",
        f"              outs(%src : {tile_src})",
        "",
        f"    pto.tcvt ins(%src {{rmode = {gen_rmode_attr(rmode)}}} : {tile_src})" if rmode != "RINT" else f"    pto.tcvt ins(%src : {tile_src})",
        f"             outs(%dst : {tile_dst})",
        "",
        f"    pto.tstore ins(%dst : {tile_dst})",
        f"               outs(%dst_part : !pto.partition_tensor_view<1x1x1x{dst_v_rows}x{dst_v_cols}x{dst_dtype}>)",
        f"    return",
        f"  }}",
        ""
    ])
    return "\n".join(lines)

header = """// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// TileLang ST kernels for pto.tcvt.
// Compiled by ptoas --enable-insert-sync --enable-tile-op-expand --pto-backend=vpto
// to produce a fatobj object.
// Generated by gen_tcvt_pto.py from cases.py.

module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
"""

footer = "\n}\n"

if __name__ == "__main__":
    from pathlib import Path
    HERE = Path(__file__).parent

    with open(HERE / "tcvt.pto", "w") as f:
        f.write(header)
        f.write("\n".join(gen_kernel(case, idx) for idx, case in enumerate(cases.CASES)))
        f.write(footer)
    print(f"Generated {(HERE / 'tcvt.pto').as_posix()!r}")
