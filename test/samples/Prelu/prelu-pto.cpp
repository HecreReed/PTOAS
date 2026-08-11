// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

extern "C" __global__ AICORE void prelu_kernel_2d(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  unsigned v4 = 0;
  const int32_t v5 = 32;
  const int32_t v6 = 1;
  const int64_t v7 = 0;
  const int64_t v8 = 4096;
  const int64_t v9 = 8192;
  const int64_t v10 = 12288;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  pto::Shape<1, 1, 1, 32, 32> v11 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v12 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v13 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v4 + v4 * (unsigned) v5 + v4 * (unsigned) v6), v11, v12);
  pto::Shape<1, 1, 1, 32, 32> v14 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v15 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v16 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v2 + (v4 + v4 * (unsigned) v5 + v4 * (unsigned) v6), v14, v15);
  pto::Shape<1, 1, 1, 32, 32> v17 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v18 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v4 + v4 * (unsigned) v5 + v4 * (unsigned) v6), v17, v18);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v20;
  uint64_t v21 = (uint64_t) v7;
  TASSIGN(v20, v21);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v22;
  uint64_t v23 = (uint64_t) v8;
  TASSIGN(v22, v23);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24;
  uint64_t v25 = (uint64_t) v9;
  TASSIGN(v24, v25);
  Tile<TileType::Vec, uint8_t, 33, 32, BLayout::RowMajor, 32, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26;
  uint64_t v27 = (uint64_t) v10;
  TASSIGN(v26, v27);
  TLOAD(v20, v13);
  TLOAD(v22, v16);
  TPRELU(v24, v20, v22, v26);
  TSTORE(v19, v24);
  #endif // __DAV_VEC__

  return;
}
