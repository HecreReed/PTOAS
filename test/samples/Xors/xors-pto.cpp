#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void XorS_kernel_2d(__gm__ int32_t* v1, __gm__ int32_t* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 88;
  int32_t v6 = 32;
  int32_t v7 = 1;
  int64_t v8 = 0;
  int64_t v9 = 4096;
  using T = float;
  unsigned v10 = (unsigned) v6;
  unsigned v11 = v4 * v10;
  unsigned v12 = v4 + v11;
  unsigned v13 = (unsigned) v7;
  unsigned v14 = v4 * v13;
  unsigned v15 = v12 + v14;
  __gm__ int32_t* v16 = v1 + v15;
  using GTShape_5065018864 = pto::Shape<32, 32>;
  using GTStride_5065018864 = pto::Stride<32, 1>;
  GTShape_5065018864 v17 = GTShape_5065018864();
  GTStride_5065018864 v18 = GTStride_5065018864();
  using GT_5065018864 = GlobalTensor<int32_t, GTShape_5065018864, GTStride_5065018864>;
  GT_5065018864 v19 = GT_5065018864(v16, v17, v18);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v9);
  TLOAD(v20, v19);
  TXORS(v21, v20, v5);
  unsigned v22 = (unsigned) v6;
  unsigned v23 = v4 * v22;
  unsigned v24 = v4 + v23;
  unsigned v25 = (unsigned) v7;
  unsigned v26 = v4 * v25;
  unsigned v27 = v24 + v26;
  __gm__ int32_t* v28 = v2 + v27;
  using GTShape_5064998624 = pto::Shape<32, 32>;
  using GTStride_5064998624 = pto::Stride<32, 1>;
  GTShape_5064998624 v29 = GTShape_5064998624();
  GTStride_5064998624 v30 = GTStride_5064998624();
  using GT_5064998624 = GlobalTensor<int32_t, GTShape_5064998624, GTStride_5064998624>;
  GT_5064998624 v31 = GT_5064998624(v28, v29, v30);
  TSTORE(v31, v21);
  return;
}


