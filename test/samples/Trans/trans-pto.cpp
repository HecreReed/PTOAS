#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void vec_transpose_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int64_t v7 = 0;
  int64_t v8 = 4096;
  using T = float;
  unsigned v9 = (unsigned) v5;
  unsigned v10 = v4 * v9;
  unsigned v11 = v4 + v10;
  unsigned v12 = (unsigned) v6;
  unsigned v13 = v4 * v12;
  unsigned v14 = v11 + v13;
  __gm__ float* v15 = v1 + v14;
  using GTShape_5350209104 = pto::Shape<32, 32>;
  using GTStride_5350209104 = pto::Stride<32, 1>;
  GTShape_5350209104 v16 = GTShape_5350209104();
  GTStride_5350209104 v17 = GTStride_5350209104();
  using GT_5350209104 = GlobalTensor<float, GTShape_5350209104, GTStride_5350209104>;
  GT_5350209104 v18 = GT_5350209104(v15, v16, v17);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v7);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v8);
  TLOAD(v19, v18);
  TTRANS(v20, v19);
  unsigned v22 = (unsigned) v5;
  unsigned v23 = v4 * v22;
  unsigned v24 = v4 + v23;
  unsigned v25 = (unsigned) v6;
  unsigned v26 = v4 * v25;
  unsigned v27 = v24 + v26;
  __gm__ float* v28 = v2 + v27;
  using GTShape_5350210624 = pto::Shape<32, 32>;
  using GTStride_5350210624 = pto::Stride<32, 1>;
  GTShape_5350210624 v29 = GTShape_5350210624();
  GTStride_5350210624 v30 = GTStride_5350210624();
  using GT_5350210624 = GlobalTensor<float, GTShape_5350210624, GTStride_5350210624>;
  GT_5350210624 v31 = GT_5350210624(v28, v29, v30);
  TSTORE(v31, v21);
  return;
}


