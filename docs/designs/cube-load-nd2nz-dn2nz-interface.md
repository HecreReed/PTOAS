# `cube_load_nd2nz` / `cube_load_dn2nz` 接口统一性整理

## 1. 目标

本文只做一件事：基于 `pto-isa` 的真实 A5 用法，整理 `cube_load_nd2nz` 和 `cube_load_dn2nz` 对应底层接口的参数语义与典型使用场景，评估两者是否可以收敛到同一套上层接口模型。

本文不讨论 release 文档写法，也不讨论 LLVM emitter 细节，只关注：

- `pto-isa` 里底层 intrinsic 是怎么被调用的
- 每个场景下每个参数实际表达什么
- 哪些参数天然共通
- 哪些差异需要保留为 mode 区分

## 2. 底层接口长什么样

在 A5 `pto-isa` 中，这两条路径最终都走 `TLoadCubeInstr`，再分发到底层 intrinsic：

- `ND` 路径: `copy_gm_to_cbuf_multi_nd2nz`
- `DN` 路径: `copy_gm_to_cbuf_multi_dn2nz`

参考：

- [`include/pto/npu/a5/TLoad.hpp:235`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L235)

两者在 A5 上的调用形态基本一致：

```cpp
copy_gm_to_cbuf_multi_*d2nz(dst, src,
    0 /*sid*/,
    loop1SrcStride,
    0 /*l2_cache_ctrl*/,
    nValue,
    dValue,
    loop4SrcStride,
    false /*smallc0_en*/);
```

这里有两个重要事实：

1. `sid` 在 `pto-isa` 的这些场景里固定为 `0`
2. 目标侧的 NZ 落点结构，并不是通过 intrinsic 参数直接完整表达，而是预先通过 `set_mte2_nz_para(...)` 编程

也就是说，真实语义来自两部分：

- intrinsic 实参: 源侧遍历方式 + 一部分搬运形状
- `MTE2_NZ_PARA`: 目标侧 NZ 存放结构

## 3. 统一参数视角

虽然底层名字分成 `nd2nz` 和 `dn2nz`，但从 `pto-isa` 的真实使用看，它们可以先抽象成同一组语义参数，并收敛到统一的上层接口 `cube_load_frac`：

### 3.1 intrinsic 侧参数

| 统一名称 | A5 底层字段 | 语义 |
|---|---|---|
| `src` | `src` | GM 源指针 |
| `dst` | `dst` | CBUF/L1 目标指针 |
| `l2_cache_ctrl` | `l2_cache_ctrl` | L2 cache control 配置位 |
| `src_inner_stride` | `loop1SrcStride` | 源侧最内层重复单元之间的跨度，单位 byte |
| `n_value` | `nValue` | 一次连续搬运的内层长度 |
| `d_value` | `dValue` | 被打包进 NZ/C0 结构的那一维大小，常见是 `C`、`K` 或 `validRow/validCol` 中的一维 |
| `src_outer_stride` | `loop4SrcStride` | 源侧更外一层重复单元之间的跨度，单位 byte；无外层时通常为 `0` |
| `smallc0_en` | `smallc0_mode` | small C0 mode 开关；仅在 `D <= 4` 时可开启 |

### 3.2 `MTE2_NZ_PARA` 侧参数

`pto-isa` 中目标侧结构通过 `set_mte2_nz_para(...)` 传入：

```text
MTE2_NZ_PARA[63:48] = loop4DstStride
MTE2_NZ_PARA[47:32] = loop3DstStride
MTE2_NZ_PARA[31:16] = loop2DstStride
MTE2_NZ_PARA[15:0]  = groupCount
```

这里的 `loop2/3/4` 目标 stride 单位都不是 byte，而是 `C0_size`。

在 A5 上，`C0_size` 是硬件固定的 32B 地址单位。  
因此：

- `dst_loop*_stride = 1` 表示目标地址前进 `32B`
- `dst_loop*_stride = 4` 表示目标地址前进 `128B`

需要注意，`C0_size` 固定为 32B，但一个 `C0` 中包含多少个元素，取决于元素类型大小：

| 元素类型 | 每个 `C0` 可容纳的元素数 |
|---|---|
| `i8` / `u8` | `32` |
| `f16` / `i16` | `16` |
| `f32` / `i32` | `8` |

这里最后 16bit 在不同 mode 下叫法不同：

- `nd2nz` 场景里通常叫 `ndNum`
- `dn2nz` 场景里通常叫 `dnNum`

但从统一建模的角度，它们本质上都可以看成：

- `group_count`: 目标侧 NZ 排布中，由硬件一次处理的外层组数

因此目标侧可以统一抽象成：

| 统一名称 | A5 底层字段 | 语义 |
|---|---|---|
| `group_count` | `MTE2_NZ_PARA[15:0]` | 最内层之上的目标分组数；在不同场景里具体映射为 `ndNum` 或 `dnNum` |
| `dst_loop2_stride` | `MTE2_NZ_PARA[31:16]` | 目标 NZ 结构的 loop2 步长 |
| `dst_loop3_stride` | `MTE2_NZ_PARA[47:32]` | 目标 NZ 结构的 loop3 步长 |
| `dst_loop4_stride` | `MTE2_NZ_PARA[63:48]` | 目标 NZ 结构的 loop4 步长 |

## 4. `nd2nz` 的真实使用场景

### 4.1 场景 A: `MX_A_ND -> ZZ`

对应 `pto-isa`：

- `TLoadMxCubeADN2ZZ`
- [`include/pto/npu/a5/TLoad.hpp:723`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L723)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `n_value` | `validCol >> 1` | 每次搬运的列向长度 |
| `d_value` | `validRow` | 每次搬运的行向长度 |
| `src_inner_stride` | `GetByteSize(dtype, gStride4) * sizeof(uint16_t)` | 源内层相邻片段跨度 |
| `src_outer_stride` | `0` | 该场景无更外一层源重复 |
| `group_count` | `1` | 单组 |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `TileData::Cols >> 1` | 目标列方向 NZ 布局步长 |
| `dst_loop4_stride` | `0` | 无更外层目标重复 |

这个场景本质上是：

- 源是 ND 风格遍历
- 目标写入左矩阵使用的 ZZ 型 NZ 布局

### 4.2 场景 B: `MX_B_ND -> NN`

对应 `pto-isa`：

- `TLoadMxCubeBND2NN`
- [`include/pto/npu/a5/TLoad.hpp:744`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L744)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `n_value` | `validRow >> 1` | 每次搬运的行向长度 |
| `d_value` | `validCol` | 每次搬运的列向长度 |
| `src_inner_stride` | `GetByteSize(dtype, gStride3) * sizeof(uint16_t)` | 源内层相邻片段跨度 |
| `src_outer_stride` | `0` | 无更外层源重复 |
| `group_count` | `1` | 单组 |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `TileData::Rows >> 1` | 目标行方向 NZ 布局步长 |
| `dst_loop4_stride` | `0` | 无更外层目标重复 |

这个场景和上一条基本同构，只是 `A/B` 左右矩阵语义不同，导致 `n_value` / `d_value` 与目标 stride 的映射不同。

### 4.3 场景 C: 通用 `ND -> [N,C1,H,W,C0]`

对应 `pto-isa`：

- 通用 ND 到卷积 tile 的路径
- [`include/pto/npu/a5/TLoad.hpp:1000`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L1000)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `group_count` | `srcShape2` | 这里就是 `ndNum = H` |
| `n_value` | `srcShape3` | 这里是 `W` |
| `d_value` | `srcShape4` | 这里是 `C` |
| `src_inner_stride` | `bytes(gStride3)` | W 维相邻行组跨度 |
| `src_outer_stride` | `bytes(gStride2)` | H 维相邻组跨度 |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `dstShape2 * dstShape3` | 目标 `H*W` 组跨度 |
| `dst_loop4_stride` | `dstShape3` | 目标 `W` 步长 |

这个场景最能体现 `nd2nz` 的共性：

- `group_count` 真正承担的是一个外层 ND 组数
- `src_outer_stride` 在这里是真实有意义的，不是所有场景都能省掉

## 5. `dn2nz` 的真实使用场景

### 5.1 场景 A: `MX_A_DN -> ZZ`

对应 `pto-isa`：

- `TLoadMxCubeAND2ZZ`
- [`include/pto/npu/a5/TLoad.hpp:664`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L664)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `n_value` | `validCol >> 1` | 每次搬运的列向长度 |
| `d_value` | `validRow` | 每次搬运的行向长度 |
| `src_inner_stride` | `bytes(gStride3)` | 源内层相邻片段跨度 |
| `src_outer_stride` | `0` | 无更外层源重复 |
| `group_count` | `1` | 这里就是 `dnNum = 1` |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `TileData::Cols >> 1` | 目标列方向 NZ 布局步长 |
| `dst_loop4_stride` | `0` | 无更外层目标重复 |

### 5.2 场景 B: `MX_B_DN -> NN`

对应 `pto-isa`：

- `TLoadMxCubeBDN2NN`
- [`include/pto/npu/a5/TLoad.hpp:765`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L765)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `n_value` | `validRow >> 1` | 每次搬运的行向长度 |
| `d_value` | `validCol` | 每次搬运的列向长度 |
| `src_inner_stride` | `bytes(gStride4)` | 源内层相邻片段跨度 |
| `src_outer_stride` | `0` | 无更外层源重复 |
| `group_count` | `1` | 单组 |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `TileData::Rows >> 1` | 目标行方向 NZ 布局步长 |
| `dst_loop4_stride` | `0` | 无更外层目标重复 |

### 5.3 场景 C: `NCHW -> [N,C1,H,W,C0]`

对应 `pto-isa`：

- `TLoadNCHW`
- [`include/pto/npu/a5/TLoad.hpp:1027`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L1027)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `group_count` | `1` | 这里固定 `dnNum = 1` |
| `n_value` | `srcW` 或 `srcH * srcW` | 内层搬运单元；W 连续时可并成 `H*W` |
| `d_value` | `srcC` | 被 pack 进 `C0` 的通道数 |
| `src_inner_stride` | `bytes(gStride2)` | 相邻 `C` 分片对应的源跨度 |
| `src_outer_stride` | `0` | 该路径外层循环通常软件展开在外面 |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `dstH * dstW` | 目标 HW 组跨度 |
| `dst_loop4_stride` | `dstW` | 目标 W 步长 |

这里 `dn2nz` 的特点是：

- `group_count` 往往不是 `H` / `D` 这种大维度
- 外层 `H` 或 `N` 的重复，很多时候不是塞进 intrinsic，而是由外层 for 循环包住

### 5.4 场景 D: `NCDHW -> [N,D,C1,H,W,C0]`

对应 `pto-isa`：

- `TLoadNCDHW2NDC1HWC0`
- [`include/pto/npu/a5/TLoad.hpp:1128`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L1128)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `group_count` | `1` | 这里固定 `dnNum = 1` |
| `n_value` | `srcH * srcW` 或退化为 `srcW` | H/W 是否连续决定内层搬运长度 |
| `d_value` | `srcC` | 被 pack 进 `C0` 的通道数 |
| `src_inner_stride` | `bytes(gStride1)` | 相邻 `C` 分片的源跨度 |
| `src_outer_stride` | `0` | `D` / `H` 外层重复通常由外部循环承担 |
| `dst_loop2_stride` | `1` | 固定 |
| `dst_loop3_stride` | `dstH * dstW` | 目标 HW 组跨度 |
| `dst_loop4_stride` | `dstW` | 目标 W 步长 |

### 5.5 场景 E: `NCHW -> FractalZ`

对应 `pto-isa`：

- `TLoadNCHW2FractalZ`
- [`include/pto/npu/a5/TLoad.hpp:1085`](../../../../gitlab.com/cann/pto-isa/include/pto/npu/a5/TLoad.hpp#L1085)

参数映射：

| 参数 | 取值方式 | 含义 |
|---|---|---|
| `group_count` | `srcShape1` | 这里 `dnNum = N` |
| `n_value` | `gStride2` | 一次搬完整个 `H*W` |
| `d_value` | `srcShape2` | 这里是 `C` |
| `src_inner_stride` | `bytes(gStride2)` | 一个 `N` 组对应的源跨度 |
| `src_outer_stride` | `bytes(gStride1)` | 相邻更外层组跨度 |
| `dst_loop2_stride` | `dstShape1 * dstShape2` | 目标 loop2 步长 |
| `dst_loop3_stride` | `loop2DstStride * dstHW` | 目标 loop3 步长 |
| `dst_loop4_stride` | `1` | 连续存放 |

这个场景说明：

- `dn2nz` 也不是只能处理 `group_count = 1`
- `src_outer_stride` 也不是 `dn2nz` 专属的无效参数

## 6. 对比结论

从 `pto-isa` 的真实使用看，`nd2nz` 和 `dn2nz` 的差异并不在于“参数种类不同”，而在于“同一组参数对应的源布局遍历语义不同”。

两者共通点：

- 都需要 `src_inner_stride`
- 都需要 `n_value`
- 都需要 `d_value`
- 都可能需要 `src_outer_stride`
- 都需要目标侧 `group_count / dst_loop2_stride / dst_loop3_stride / dst_loop4_stride`
- 都经常配合外层软件循环使用

两者核心差异：

- `group_count` 在 `nd2nz` 中更像 ND 分组数，在 `dn2nz` 中更像 DN 分组数
- 源张量哪一维映射到 `n_value` / `d_value` / `src_inner_stride`，取决于源布局模式
- 某些 `dn2nz` 场景会把 `H` / `D` 外层维度拆到软件循环，而不是塞进 `group_count`

因此，如果只从“参数列表”看，这两条接口是可以统一的；真正需要保留差异的是：

- 一个显式 `nd2nz | dn2nz` mode keyword
- 每个 mode 自己的 shape-to-parameter 映射规则

## 7. 一个最小搬移示意

这里用一个最小例子，把这些参数如何驱动“多组 2D 矩阵 -> 多组 NZ 分形”的搬移过程画出来。

设：

- `group_count = 2`
- `n_value = 3`
- `d_value = 5`
- `src_inner_stride = 32B`
- `src_outer_stride = 256B`
- `dst_loop2_stride = 1`
- `dst_loop3_stride = 4`
- `dst_loop4_stride = 20`

这里可以把源理解成两组逻辑 2D 矩阵，每组都是 `N x D = 3 x 5`。

### 7.1 源侧视角

先不区分 `nd2nz` / `dn2nz` 的地址解释差异，只看统一抽象下的“分组 + 内层步长”：

```text
group 0 base = src + 0 * src_outer_stride
group 1 base = src + 1 * src_outer_stride

group g 内部有 3 个 N 单元：

N0 base = group_base + 0 * src_inner_stride
N1 base = group_base + 1 * src_inner_stride
N2 base = group_base + 2 * src_inner_stride

每个 N 单元里有 D=5 个元素：

N0: [d0 d1 d2 d3 d4]
N1: [d0 d1 d2 d3 d4]
N2: [d0 d1 d2 d3 d4]
```

如果画成两组源矩阵，可以看成：

```text
group 0:
  N0 -> [00 01 02 03 04]
  N1 -> [10 11 12 13 14]
  N2 -> [20 21 22 23 24]

group 1:
  N0 -> [30 31 32 33 34]
  N1 -> [40 41 42 43 44]
  N2 -> [50 51 52 53 54]
```

这里：

- `src_inner_stride` 决定 `N0 -> N1 -> N2` 怎么跳
- `src_outer_stride` 决定 `group 0 -> group 1` 怎么跳
- `n_value = 3` 决定每组取 3 条 N
- `d_value = 5` 决定每条 N 上取 5 个 D 元素

### 7.2 目标 NZ 视角

目标不是平铺成普通二维矩阵，而是按 NZ 分形排布到 L1。

可以先把它抽象成：

```text
group g 的目标基址 = dst + g * dst_loop4_stride * C0_size

group 内部：
  第 i 个 D-block 的目标基址 = group_dst_base + i * dst_loop3_stride * C0_size
  第 j 个 N 单元的目标基址 = d_block_dst_base + j * dst_loop2_stride * C0_size
```

在这个例子里：

- `dst_loop2_stride = 1` 表示相邻 N 单元在目标上紧邻排布
- `dst_loop3_stride = 4` 表示相邻 D-block 之间隔 4 个 `C0_size`
- `dst_loop4_stride = 20` 表示相邻 group 的整块矩阵在目标上隔 20 个 `C0_size`

如果只画逻辑落点关系，不展开完整 `C0`，可以看成：

```text
group 0 NZ:
  D-block 0:
    N0 <- [00 01 02 03 ...]
    N1 <- [10 11 12 13 ...]
    N2 <- [20 21 22 23 ...]
  D-block 1:
    N0 <- [04 pad pad pad ...]
    N1 <- [14 pad pad pad ...]
    N2 <- [24 pad pad pad ...]

group 1 NZ:
  D-block 0:
    N0 <- [30 31 32 33 ...]
    N1 <- [40 41 42 43 ...]
    N2 <- [50 51 52 53 ...]
  D-block 1:
    N0 <- [34 pad pad pad ...]
    N1 <- [44 pad pad pad ...]
    N2 <- [54 pad pad pad ...]
```

这里故意选 `d_value = 5`，就是为了看出尾块不满时的行为：

- 第一块装下前 4 个 D 元素
- 第二块只剩第 5 个元素
- 尾部由硬件补 pad

### 7.3 参数到底在控制什么

把这个例子压缩成一句话：

- `n_value` 决定每组有多少条 N 线要搬
- `d_value` 决定每条 N 线上有多少个 D 元素要 pack 进分形
- `src_inner_stride` 决定源上相邻两条 N 线怎么跳
- `src_outer_stride` 决定源上相邻两组矩阵怎么跳
- `dst_loop2_stride` 决定目标上相邻 N 线怎么摆
- `dst_loop3_stride` 决定目标上相邻 D-block 怎么摆
- `dst_loop4_stride` 决定目标上相邻 group 怎么摆

### 7.4 `nd2nz` 和 `dn2nz` 真正差在哪

上面的图故意只画了统一抽象，因为两条指令的参数框架本身是一样的。

真正的差异在于：源侧地址解释顺序不同。

- `nd2nz`: 更像把源看成 ND 矩阵，再按 `N x D` 逻辑去取数
- `dn2nz`: 更像把源看成 DN 矩阵，再按另一套源地址递推顺序去取数

但无论哪一种：

- `n_value` / `d_value` 仍然定义“这一组搬多大”
- `src_inner_stride` / `src_outer_stride` 仍然定义“源怎么走”
- `dst_loop2/3/4_stride` 仍然定义“NZ 分形怎么落”

因此从上层接口看，它们完全可以共享同一组参数模型，只在 `mode` 上区分源布局解释规则。

## 8. 建议的统一抽象

如果上层想统一接口，建议先统一成“参数语义层”，而不是强行复用现有底层名字。

可以考虑的统一抽象如下：

```text
cube_load_frac(
  src,
  dst,
  nd2nz | dn2nz,
  shape(n_value, d_value),
  src_layout(src_inner_stride, src_outer_stride?),
  dst_group(group_count, dst_loop2_stride, dst_loop3_stride, dst_loop4_stride),
  ctrl(l2_cache_ctrl, smallc0_en)
)
```

其中：

- `shape(...)` 只描述一次分形搬移的逻辑 `N x D` 大小
- `src_layout(...)` 只描述源侧地址递推
- `dst_group(...)` 只描述目标 NZ 分形排布
- `ctrl(...)` 只描述底层控制位

如果 `src_outer_stride` 不提供，则默认按 `0` 处理。

这套抽象的好处：

- `nd2nz` / `dn2nz` 共享同一组结构化参数
- 底层是否走 `copy_gm_to_cbuf_multi_nd2nz` 还是 `copy_gm_to_cbuf_multi_dn2nz`，由 `mode` 决定
- `MTE2_NZ_PARA` 的 4 个字段可以原样保留，不需要再隐式推导
这类接口不单独暴露 `padding` 参数。

- 当 `d_value` 不能完整填满目标分形时，尾部补齐由硬件按 zero padding 完成
- 当 `smallc0_en = true` 时，small C0 mode 会改变补齐与对齐方式，但仍然不是用户可配置的 pad value

因此，这里的 padding 语义属于指令内建行为，而不是像 `dma_load` 那样的显式接口参数。

如果直接写成接近 VPTO 的 syntax 草案，可以是：

```text
pto.mte_gm_l1_frac %src, %dst,
    nd2nz | dn2nz,
    shape(%n_value, %d_value),
    src_layout(%src_inner_stride[, %src_outer_stride]),
    dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride),
    ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<..., gm>, !pto.ptr<..., l1>,
    nd2nz | dn2nz,
    shape i64, i64,
    src_layout(i64[, i64]),
    dst_group i64, i64, i64, i64,
    ctrl i64, i1
```

推荐的 builder 视角也和语法保持一致：

```text
cube_load_frac(
  src, dst,
  mode,
  shape(n_value, d_value),
  src_layout(src_inner_stride, src_outer_stride = 0),
  dst_group(group_count, dst_loop2_stride, dst_loop3_stride, dst_loop4_stride),
  ctrl(l2_cache_ctrl, smallc0_en)
)
```

## 9. 哪些参数可以默认，哪些最好显式暴露

从 `pto-isa` 的现状看：

### 9.1 可以默认的

- `sid = 0`

`sid` 在当前调研到的 A5 `pto-isa` 使用点里都是固定值。

### 9.2 建议显式暴露的

- `mode`
- `shape(n_value, d_value)`
- `src_layout(src_inner_stride, src_outer_stride?)`
- `dst_group(group_count, dst_loop2_stride, dst_loop3_stride, dst_loop4_stride)`
- `ctrl(l2_cache_ctrl, smallc0_en)`

这些都是底层接口真实存在、并且会影响行为或未来扩展空间的参数。其中：

- `l2_cache_ctrl` 当前 `pto-isa` A5 用法里固定传 `0`
- `smallc0_en` 当前 `pto-isa` A5 用法里固定传 `false`
- 但从 `disa-cube.json` 看，这两个字段都属于原始接口语义的一部分，不应在统一接口里直接消失

### 9.3 可选暴露的

- `src_outer_stride`

这个参数不是每个场景都需要，但一旦做通用接口，最好保留。  
在结构化接口里，`src_outer_stride` 仍属于 `src_layout(...)` 的一部分，只是允许省略。

## 10. 初步判断

结论很直接：

- `cube_load_nd2nz` 和 `cube_load_dn2nz` 在参数语义层是可以统一的
- 不能统一掉的不是参数列表，而是 `mode` 对源布局遍历规则的解释
- 如果后续要做 VPTO 新接口，建议抽象成一个统一的 `cube_load_frac` 接口，再保留 `nd2nz | dn2nz` 这两个 mode keyword
- 这套统一接口更适合使用结构化分组：
  - `shape(...)`
  - `src_layout(...)`
  - `dst_group(...)`
  - `ctrl(...)`
- 在这套统一接口里，`l2_cache_ctrl` 和 `smallc0_en` 也应保留为显式参数；只有 `sid` 可以继续固定隐藏

如果下一步需要，我可以继续把这份设计文档再往前推进一层，直接写成一版面向 VPTO op 设计的 syntax 草案和 verifier 约束。 
