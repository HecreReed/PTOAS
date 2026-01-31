#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void vec_add_scalar_kernel_2d(__gm__ int32_t* v1, __gm__ int32_t* v2) {
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
  __gm__ int32_t* v15 = v1 + v14;
  using GTShape_4989499200 = pto::Shape<32, 32>;
  using GTStride_4989499200 = pto::Stride<32, 1>;
  GTShape_4989499200 v16 = GTShape_4989499200();
  GTStride_4989499200 v17 = GTStride_4989499200();
  using GT_4989499200 = GlobalTensor<int32_t, GTShape_4989499200, GTStride_4989499200>;
  GT_4989499200 v18 = GT_4989499200(v15, v16, v17);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v7);
  Tile<TileType::Vec, int32_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  TLOAD(v19, v18);
  TNOT(v20, v19);
  unsigned v21 = (unsigned) v5;
  unsigned v22 = v4 * v21;
  unsigned v23 = v4 + v22;
  unsigned v24 = (unsigned) v6;
  unsigned v25 = v4 * v24;
  unsigned v26 = v23 + v25;
  __gm__ int32_t* v27 = v2 + v26;
  using GTShape_4989500736 = pto::Shape<32, 32>;
  using GTStride_4989500736 = pto::Stride<32, 1>;
  GTShape_4989500736 v28 = GTShape_4989500736();
  GTStride_4989500736 v29 = GTStride_4989500736();
  using GT_4989500736 = GlobalTensor<int32_t, GTShape_4989500736, GTStride_4989500736>;
  GT_4989500736 v30 = GT_4989500736(v27, v28, v29);
  TSTORE(v30, v20);
  return;
}


