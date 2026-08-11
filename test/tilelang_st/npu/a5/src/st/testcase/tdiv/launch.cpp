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


// Case: f32_16x64
extern "C" __global__ AICORE void TDIV_f32_16x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_32x32
extern "C" __global__ AICORE void TDIV_f32_32x32(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_32x32(float *a, float *b, float *c, void *stream) {
    TDIV_f32_32x32<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_64x64
extern "C" __global__ AICORE void TDIV_f32_64x64(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_64x64(float *a, float *b, float *c, void *stream) {
    TDIV_f32_64x64<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_16x256
extern "C" __global__ AICORE void TDIV_f16_16x256(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_16x256(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x256<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_16x64_hp_precision
extern "C" __global__ AICORE void TDIV_f32_16x64_hp_precision(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64_hp_precision(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64_hp_precision<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_16x64_hp_precision
extern "C" __global__ AICORE void TDIV_f16_16x64_hp_precision(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_16x64_hp_precision(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x64_hp_precision<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_16x64_hp_subnormal
extern "C" __global__ AICORE void TDIV_f32_16x64_hp_subnormal(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64_hp_subnormal(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64_hp_subnormal<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_16x64_hp_subnormal
extern "C" __global__ AICORE void TDIV_f16_16x64_hp_subnormal(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_16x64_hp_subnormal(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x64_hp_subnormal<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_16x64_hp_overflow
extern "C" __global__ AICORE void TDIV_f32_16x64_hp_overflow(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64_hp_overflow(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64_hp_overflow<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_16x64_hp_overflow
extern "C" __global__ AICORE void TDIV_f16_16x64_hp_overflow(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_16x64_hp_overflow(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x64_hp_overflow<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_32x32_hp
extern "C" __global__ AICORE void TDIV_f32_32x32_hp(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_32x32_hp(float *a, float *b, float *c, void *stream) {
    TDIV_f32_32x32_hp<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_64x64_hp
extern "C" __global__ AICORE void TDIV_f32_64x64_hp(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_64x64_hp(float *a, float *b, float *c, void *stream) {
    TDIV_f32_64x64_hp<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_16x256_hp
extern "C" __global__ AICORE void TDIV_f16_16x256_hp(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_16x256_hp(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x256_hp<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_16x64_hp_partial
extern "C" __global__ AICORE void TDIV_f32_16x64_hp_partial(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_16x64_hp_partial(float *a, float *b, float *c, void *stream) {
    TDIV_f32_16x64_hp_partial<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_16x64_hp_partial
extern "C" __global__ AICORE void TDIV_f16_16x64_hp_partial(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_16x64_hp_partial(void *a, void *b, void *c, void *stream) {
    TDIV_f16_16x64_hp_partial<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f32_2x16_hp
extern "C" __global__ AICORE void TDIV_f32_2x16_hp(__gm__ float *a, __gm__ float *b, __gm__ float *c);

void LaunchTDIV_f32_2x16_hp(float *a, float *b, float *c, void *stream) {
    TDIV_f32_2x16_hp<<<1, nullptr, stream>>>(a, b, c);
}

// Case: f16_2x32_hp
extern "C" __global__ AICORE void TDIV_f16_2x32_hp(__gm__ void *a, __gm__ void *b, __gm__ void *c);

void LaunchTDIV_f16_2x32_hp(void *a, void *b, void *c, void *stream) {
    TDIV_f16_2x32_hp<<<1, nullptr, stream>>>(a, b, c);
}
