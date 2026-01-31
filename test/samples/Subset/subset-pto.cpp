#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void subset_pingpong_demo() {
  int64_t v1 = 0;
  int32_t v2 = 0;
  int32_t v3 = 32;
  using T = float;
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, 32, 64, SLayout::NoneBox, 512, PadValue::Null> v4;
  TASSIGN(v4, v1);
  __ub__ float* v5 = v4 + v2;
  __ub__ float* v6 = v4 + v3;
  TADD(v5, v5, v5);
  TADD(v6, v6, v6);
  return;
}


