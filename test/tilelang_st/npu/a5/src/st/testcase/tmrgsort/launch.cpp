// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

// Case: f32_single_1x256_b64 (transplanted from pto-isa case_single1)
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x256_b64(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x256_b64(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x256_b64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case: f32_single_1x320_b64 (transplanted from pto-isa case_single2)
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x320_b64(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x320_b64(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x320_b64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case: f32_single_1x512_b64 (transplanted from pto-isa case_single3)
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x512_b64(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x512_b64(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x512_b64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case: f32_single_1x640_b64 (transplanted from pto-isa case_single4)
extern "C" __global__ AICORE void TMRGSORT_f32_single_1x640_b64(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_single_1x640_b64(float *src, float *dst, void *stream) {
    TMRGSORT_f32_single_1x640_b64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// Case: f16_single_1x256_b64 (transplanted from pto-isa case_single5)
extern "C" __global__ AICORE void TMRGSORT_f16_single_1x256_b64(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_single_1x256_b64(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_single_1x256_b64<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}

// Case: f16_single_1x320_b64 (transplanted from pto-isa case_single6)
extern "C" __global__ AICORE void TMRGSORT_f16_single_1x320_b64(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_single_1x320_b64(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_single_1x320_b64<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}

// Case: f16_single_1x512_b64 (transplanted from pto-isa case_single7)
extern "C" __global__ AICORE void TMRGSORT_f16_single_1x512_b64(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_single_1x512_b64(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_single_1x512_b64<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}

// Case: f16_single_1x1024_b256 (transplanted from pto-isa case_single8)
extern "C" __global__ AICORE void TMRGSORT_f16_single_1x1024_b256(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_single_1x1024_b256(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_single_1x1024_b256<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}

// Multi-list cases
extern "C" __global__ AICORE void TMRGSORT_f32_2list_b64_basic(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTMRGSORT_f32_2list_b64_basic(float *src0, float *src1, float *dst, void *stream) {
    TMRGSORT_f32_2list_b64_basic<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TMRGSORT_f16_2list_b64_basic(__gm__ half *src0, __gm__ half *src1, __gm__ half *dst);

void LaunchTMRGSORT_f16_2list_b64_basic(uint16_t *src0, uint16_t *src1, uint16_t *dst, void *stream) {
    TMRGSORT_f16_2list_b64_basic<<<1, nullptr, stream>>>((__gm__ half *)src0, (__gm__ half *)src1, (__gm__ half *)dst);
}

extern "C" __global__ AICORE void TMRGSORT_f32_2list_exhausted(__gm__ float *src0, __gm__ float *src1, __gm__ float *dst);

void LaunchTMRGSORT_f32_2list_exhausted(float *src0, float *src1, float *dst, void *stream) {
    TMRGSORT_f32_2list_exhausted<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TMRGSORT_f32_3list_b64_basic(__gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *dst);

void LaunchTMRGSORT_f32_3list_b64_basic(float *src0, float *src1, float *src2, float *dst, void *stream) {
    TMRGSORT_f32_3list_b64_basic<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)src2, (__gm__ float *)dst);
}

extern "C" __global__ AICORE void TMRGSORT_f32_4list_b32_basic(__gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3, __gm__ float *dst);

void LaunchTMRGSORT_f32_4list_b32_basic(float *src0, float *src1, float *src2, float *src3, float *dst, void *stream) {
    TMRGSORT_f32_4list_b32_basic<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)src2, (__gm__ float *)src3, (__gm__ float *)dst);
}

// 4-list case: f16_4list_b64_basic
extern "C" __global__ AICORE void TMRGSORT_f16_4list_b64_basic(__gm__ half *src0, __gm__ half *src1, __gm__ half *src2, __gm__ half *src3, __gm__ half *dst);

void LaunchTMRGSORT_f16_4list_b64_basic(uint16_t *src0, uint16_t *src1, uint16_t *src2, uint16_t *src3, uint16_t *dst, void *stream) {
    TMRGSORT_f16_4list_b64_basic<<<1, nullptr, stream>>>((__gm__ half *)src0, (__gm__ half *)src1, (__gm__ half *)src2, (__gm__ half *)src3, (__gm__ half *)dst);
}

// 4-list case: f16_4list_basic (pto-isa case_multi2)
extern "C" __global__ AICORE void TMRGSORT_f16_4list_basic(__gm__ half *src0, __gm__ half *src1, __gm__ half *src2, __gm__ half *src3, __gm__ half *dst);

void LaunchTMRGSORT_f16_4list_basic(uint16_t *src0, uint16_t *src1, uint16_t *src2, uint16_t *src3, uint16_t *dst, void *stream) {
    TMRGSORT_f16_4list_basic<<<1, nullptr, stream>>>((__gm__ half *)src0, (__gm__ half *)src1, (__gm__ half *)src2, (__gm__ half *)src3, (__gm__ half *)dst);
}

// 3-list non-uniform: f32_3list_non_uniform
extern "C" __global__ AICORE void TMRGSORT_f32_3list_non_uniform(__gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *dst);

void LaunchTMRGSORT_f32_3list_non_uniform(float *src0, float *src1, float *src2, float *dst, void *stream) {
    TMRGSORT_f32_3list_non_uniform<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)src2, (__gm__ float *)dst);
}

// 3-list f16 exhausted: f16_3list_exhausted
extern "C" __global__ AICORE void TMRGSORT_f16_3list_exhausted(__gm__ half *src0, __gm__ half *src1, __gm__ half *src2, __gm__ half *dst);

void LaunchTMRGSORT_f16_3list_exhausted(uint16_t *src0, uint16_t *src1, uint16_t *src2, uint16_t *dst, void *stream) {
    TMRGSORT_f16_3list_exhausted<<<1, nullptr, stream>>>((__gm__ half *)src0, (__gm__ half *)src1, (__gm__ half *)src2, (__gm__ half *)dst);
}

// 4-list non-uniform: f32_4list_non_uniform
extern "C" __global__ AICORE void TMRGSORT_f32_4list_non_uniform(__gm__ float *src0, __gm__ float *src1, __gm__ float *src2, __gm__ float *src3, __gm__ float *dst);

void LaunchTMRGSORT_f32_4list_non_uniform(float *src0, float *src1, float *src2, float *src3, float *dst, void *stream) {
    TMRGSORT_f32_4list_non_uniform<<<1, nullptr, stream>>>((__gm__ float *)src0, (__gm__ float *)src1, (__gm__ float *)src2, (__gm__ float *)src3, (__gm__ float *)dst);
}

// TopK cases: f32_topk_2048_1024
extern "C" __global__ AICORE void TMRGSORT_f32_topk_2048_1024(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_topk_2048_1024(float *src, float *dst, void *stream) {
    TMRGSORT_f32_topk_2048_1024<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// TopK cases: f32_topk_2048_2048
extern "C" __global__ AICORE void TMRGSORT_f32_topk_2048_2048(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_topk_2048_2048(float *src, float *dst, void *stream) {
    TMRGSORT_f32_topk_2048_2048<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// TopK cases: f32_topk_1280_512
extern "C" __global__ AICORE void TMRGSORT_f32_topk_1280_512(__gm__ float *src, __gm__ float *dst);

void LaunchTMRGSORT_f32_topk_1280_512(float *src, float *dst, void *stream) {
    TMRGSORT_f32_topk_1280_512<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst);
}

// TopK cases: f16_topk_2048_1024
extern "C" __global__ AICORE void TMRGSORT_f16_topk_2048_1024(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_topk_2048_1024(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_topk_2048_1024<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}

// TopK cases: f16_topk_2048_2048
extern "C" __global__ AICORE void TMRGSORT_f16_topk_2048_2048(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_topk_2048_2048(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_topk_2048_2048<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}

// TopK cases: f16_topk_1280_512
extern "C" __global__ AICORE void TMRGSORT_f16_topk_1280_512(__gm__ half *src, __gm__ half *dst);

void LaunchTMRGSORT_f16_topk_1280_512(uint16_t *src, uint16_t *dst, void *stream) {
    TMRGSORT_f16_topk_1280_512<<<1, nullptr, stream>>>((__gm__ half *)src, (__gm__ half *)dst);
}
