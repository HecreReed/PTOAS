#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void shr_kernel_2d(__gm__ int32_t* v1, __gm__ int32_t* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int64_t v7 = 0;
  int64_t v8 = 4096;
  int64_t v9 = 8192;
  using T = float;
  unsigned v10 = (unsigned) v5;
  unsigned v11 = v4 * v10;
  unsigned v12 = v4 + v11;
  unsigned v13 = (unsigned) v6;
  unsigned v14 = v4 * v13;
  unsigned v15 = v12 + v14;
  __gm__ int32_t* v16 = v1 + v15;
  using GTShape_5467651040 = pto::Shape<32, 32>;
  using GTStride_5467651040 = pto::Stride<32, 1>;
  GTShape_5467651040 v17 = GTShape_5467651040();
  GTStride_5467651040 v18 = GTStride_5467651040();
  using GT_5467651040 = GlobalTensor<int32_t, GTShape_5467651040, GTStride_5467651040>;
  GT_5467651040 v19 = GT_5467651040(v16, v17, v18);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v7);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v8);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v9);
  TLOAD(v20, v19);
  TLOAD(v21, v19);
  TSHR(v22, v20, v21);
  unsigned v23 = (unsigned) v5;
  unsigned v24 = v4 * v23;
  unsigned v25 = v4 + v24;
  unsigned v26 = (unsigned) v6;
  unsigned v27 = v4 * v26;
  unsigned v28 = v25 + v27;
  __gm__ int32_t* v29 = v2 + v28;
  using GTShape_5467652640 = pto::Shape<32, 32>;
  using GTStride_5467652640 = pto::Stride<32, 1>;
  GTShape_5467652640 v30 = GTShape_5467652640();
  GTStride_5467652640 v31 = GTStride_5467652640();
  using GT_5467652640 = GlobalTensor<int32_t, GTShape_5467652640, GTStride_5467652640>;
  GT_5467652640 v32 = GT_5467652640(v29, v30, v31);
  TSTORE(v32, v22);
  return;
}


