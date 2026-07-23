// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

// FP8/FP4 typedef fallbacks: only emit when the bisheng compiler itself
// doesn't already provide them. CANN 9.2.0+ bisheng unconditionally
// typedef's these names in its built-in <__clang_cce_types.h> header
// (pulled in automatically under -xcce), so emitting our own struct-based
// typedef here would collide ("typedef redefinition with different types",
// since the SDK side is e.g. `typedef __hif8 hifloat8_t;`).
#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && !__has_include(<__clang_cce_types.h>)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif

#include <stdint.h>

#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

// Gate on __has_include(<pto/common/type.hpp>) instead of the TMRGSORT_HPP
// macro: CANN 9.2.0+ moved pto::MrgSortExecutedNumList into
// pto/common/type.hpp, which does not #define TMRGSORT_HPP, so the old
// macro check leaked and produced a redefinition compile error whenever
// type.hpp was pulled in.
#if !defined(__CCE_AICORE__) && !__has_include(<pto/common/type.hpp>)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

#if defined(__CCE_AICORE__)
__global__ AICORE void subview_split4(__gm__ float* src, __gm__ float* out0,
                                     __gm__ float* out1, __gm__ float* out2,
                                     __gm__ float* out3);
#else
__global__ AICORE void subview_split4(__gm__ float* src, __gm__ float* out0,
                                     __gm__ float* out1, __gm__ float* out2,
                                     __gm__ float* out3);
#endif

void LaunchSubViewSplit4_kernel(float *src, float *out0, float *out1, float *out2,
                                float *out3, void *stream) {
    subview_split4<<<1, nullptr, stream>>>((__gm__ float*)src, (__gm__ float*)out0,
                                          (__gm__ float*)out1, (__gm__ float*)out2,
                                          (__gm__ float*)out3);
}
