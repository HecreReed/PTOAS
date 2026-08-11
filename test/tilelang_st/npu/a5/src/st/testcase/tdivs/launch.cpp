// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>
#include <cstring>

#ifndef AICORE
#define AICORE [aicore]
#endif

static constexpr float TDIVS_SCALAR_F32 = 3.0f;

// Helper to convert IEEE 754 hex bits to float (runtime initialization)
inline float bits_to_float(uint32_t bits) {
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ========== src / scalar direction ==========

// Case 0: f32 32x64
extern "C" __global__ AICORE void TDIVS_f32_32x64(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_32x64(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 1: f16 63x64
extern "C" __global__ AICORE void TDIVS_f16_63x64(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_63x64(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_63x64<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4200);
}

// Case 2: f32 7x448
extern "C" __global__ AICORE void TDIVS_f32_7x448(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_7x448(float *src, float *dst, void *stream) {
    TDIVS_f32_7x448<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 3: f32 256x16
extern "C" __global__ AICORE void TDIVS_f32_256x16(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_256x16(float *src, float *dst, void *stream) {
    TDIVS_f32_256x16<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// ========== scalar / src direction ==========

// Case 4: f32 32x64 scalar/src
extern "C" __global__ AICORE void TDIVS_f32_32x64_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_32x64_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 5: f16 63x64 scalar/src
extern "C" __global__ AICORE void TDIVS_f16_63x64_scalar_src(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_63x64_scalar_src(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_63x64_scalar_src<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4200);
}

// Case 6: f32 7x448 scalar/src
extern "C" __global__ AICORE void TDIVS_f32_7x448_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_7x448_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_7x448_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// Case 7: f32 256x16 scalar/src
extern "C" __global__ AICORE void TDIVS_f32_256x16_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_256x16_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_256x16_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, TDIVS_SCALAR_F32);
}

// ========== HIGH_PRECISION mode - src / scalar direction ==========

// Case 8: f32 32x64 HP (precision_sensitive) - scalar=3.0f
extern "C" __global__ AICORE void TDIVS_f32_32x64_hp(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_32x64_hp(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64_hp<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, 3.0f);
}

// Case 9: f16 63x64 HP (precision_sensitive) - scalar=3.0 in f16 (0x4200)
extern "C" __global__ AICORE void TDIVS_f16_63x64_hp(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_63x64_hp(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_63x64_hp<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4200);
}

// Case 10: f32 16x64 HP subnormal - scalar=10.0f
extern "C" __global__ AICORE void TDIVS_f32_16x64_hp_subnormal(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_16x64_hp_subnormal(float *src, float *dst, void *stream) {
    TDIVS_f32_16x64_hp_subnormal<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, 10.0f);
}

// Case 11: f16 16x64 HP subnormal - scalar=10.0 in f16 (0x4900)
extern "C" __global__ AICORE void TDIVS_f16_16x64_hp_subnormal(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_16x64_hp_subnormal(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_16x64_hp_subnormal<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x4900);
}

// Case 12: f32 16x64 HP overflow - scalar=np.float32(1e-30) -> hex 0x0DA24260
extern "C" __global__ AICORE void TDIVS_f32_16x64_hp_overflow(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_16x64_hp_overflow(float *src, float *dst, void *stream) {
    TDIVS_f32_16x64_hp_overflow<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, bits_to_float(0x0DA24260U));
}

// Case 13: f16 16x64 HP overflow - scalar=np.float16(0.0001) -> hex 0x068E
extern "C" __global__ AICORE void TDIVS_f16_16x64_hp_overflow(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_16x64_hp_overflow(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_16x64_hp_overflow<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x068E);
}

// ========== HIGH_PRECISION mode - scalar / src direction ==========

// Case 14: f32 32x64 HP scalar/src (precision_sensitive) - scalar=1.0f
extern "C" __global__ AICORE void TDIVS_f32_32x64_hp_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_32x64_hp_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_32x64_hp_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, 1.0f);
}

// Case 15: f16 63x64 HP scalar/src (precision_sensitive) - scalar=1.0 in f16 (0x3C00)
extern "C" __global__ AICORE void TDIVS_f16_63x64_hp_scalar_src(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_63x64_hp_scalar_src(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_63x64_hp_scalar_src<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x3C00);
}

// Case 16: f32 16x64 HP subnormal scalar/src - scalar=np.float32(1e-20) -> hex 0x1E3CE508
extern "C" __global__ AICORE void TDIVS_f32_16x64_hp_subnormal_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_16x64_hp_subnormal_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_16x64_hp_subnormal_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, bits_to_float(0x1E3CE508U));
}

// Case 17: f16 16x64 HP subnormal scalar/src - scalar=np.float16(1e-5) -> hex 0x00A8
extern "C" __global__ AICORE void TDIVS_f16_16x64_hp_subnormal_scalar_src(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_16x64_hp_subnormal_scalar_src(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_16x64_hp_subnormal_scalar_src<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x00A8);
}

// Case 18: f32 16x64 HP overflow scalar/src - scalar=np.float32(1e38) -> hex 0x7E967699
extern "C" __global__ AICORE void TDIVS_f32_16x64_hp_overflow_scalar_src(__gm__ float *src, __gm__ float *dst, float scalar);
void LaunchTDIVS_f32_16x64_hp_overflow_scalar_src(float *src, float *dst, void *stream) {
    TDIVS_f32_16x64_hp_overflow_scalar_src<<<1, nullptr, stream>>>((__gm__ float *)src, (__gm__ float *)dst, bits_to_float(0x7E967699U));
}

// Case 19: f16 16x64 HP overflow scalar/src - scalar=np.float16(65000) -> hex 0x7BEF
extern "C" __global__ AICORE void TDIVS_f16_16x64_hp_overflow_scalar_src(__gm__ unsigned short *src, __gm__ unsigned short *dst, unsigned short scalar);
void LaunchTDIVS_f16_16x64_hp_overflow_scalar_src(unsigned short *src, unsigned short *dst, void *stream) {
    TDIVS_f16_16x64_hp_overflow_scalar_src<<<1, nullptr, stream>>>((__gm__ unsigned short *)src, (__gm__ unsigned short *)dst, (unsigned short)0x7BEF);
}