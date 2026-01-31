#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void maxs_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  float v5 = 3.1400001f;
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
  __gm__ float* v16 = v1 + v15;
  using GTShape_5694846160 = pto::Shape<32, 32>;
  using GTStride_5694846160 = pto::Stride<32, 1>;
  GTShape_5694846160 v17 = GTShape_5694846160();
  GTStride_5694846160 v18 = GTStride_5694846160();
  using GT_5694846160 = GlobalTensor<float, GTShape_5694846160, GTStride_5694846160>;
  GT_5694846160 v19 = GT_5694846160(v16, v17, v18);
  unsigned v20 = (unsigned) v6;
  unsigned v21 = v4 * v20;
  unsigned v22 = v4 + v21;
  unsigned v23 = (unsigned) v7;
  unsigned v24 = v4 * v23;
  unsigned v25 = v22 + v24;
  __gm__ float* v26 = v2 + v25;
  using GTShape_5694846336 = pto::Shape<32, 32>;
  using GTStride_5694846336 = pto::Stride<32, 1>;
  GTShape_5694846336 v27 = GTShape_5694846336();
  GTStride_5694846336 v28 = GTStride_5694846336();
  using GT_5694846336 = GlobalTensor<float, GTShape_5694846336, GTStride_5694846336>;
  GT_5694846336 v29 = GT_5694846336(v26, v27, v28);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v8);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v9);
  TLOAD(v30, v19);
  TLOAD(v31, v29);
  TMINS(v31, v30, v5);
  TSTORE(v29, v31);
  return;
}


