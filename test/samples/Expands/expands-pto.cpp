#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void vec_expand_scalar_kernel_2d(__gm__ float* v1) {
  unsigned v2 = 1;
  unsigned v3 = 0;
  float v4 = 3.1400001f;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int64_t v7 = 0;
  using T = float;
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v8;
  TASSIGN(v8, v7);
  TEXPANDS(v8, v4);
  unsigned v9 = (unsigned) v5;
  unsigned v10 = v3 * v9;
  unsigned v11 = v3 + v10;
  unsigned v12 = (unsigned) v6;
  unsigned v13 = v3 * v12;
  unsigned v14 = v11 + v13;
  __gm__ float* v15 = v1 + v14;
  using GTShape_5450892960 = pto::Shape<32, 32>;
  using GTStride_5450892960 = pto::Stride<32, 1>;
  GTShape_5450892960 v16 = GTShape_5450892960();
  GTStride_5450892960 v17 = GTStride_5450892960();
  using GT_5450892960 = GlobalTensor<float, GTShape_5450892960, GTStride_5450892960>;
  GT_5450892960 v18 = GT_5450892960(v15, v16, v17);
  TSTORE(v18, v8);
  return;
}


