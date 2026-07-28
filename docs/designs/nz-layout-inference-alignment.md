<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# PTOAS `Layout::NZ` 推断与 pto-isa 标准五维形对齐设计

关联 issue: [#527](https://github.com/hw-native-sys/PTOAS/issues/527) —— PTOAS 支持 pto-isa 中的 `Layout::NZ` 推断。

本文只讨论 GM 侧 `GlobalTensor` 的 layout 推断（`tensor_view` / `partition_tensor_view` → `pto::Layout`），不涉及 tile 侧 `blayout/slayout`、也不涉及 fixpipe 的 `NZ2ND/NZ2DN/NZ2NZ` 转换。

---

## 1. 事实基准：pto-isa 的 NZ 五维标准形

pto-isa 侧二维逻辑 tensor 的 NZ 五维表示由 `TileShape2D` / `BaseShape2D` 定义
（`include/pto/common/pto_tile.hpp`，常量见 `include/pto/common/constants.hpp`：
`C0_SIZE_BYTE = 32`、`FRACTAL_NZ_ROW = 16`）：

```
C0     = 32 / sizeof(T)
shape  = [1, cols / C0, rows / 16, 16, C0]
stride = [rows * cols, rows * C0, 16 * C0, C0, 1]
```

即 5 个维度的语义固定为：

| 维度 | 含义 | 取值 |
|---|---|---|
| `d0` | batch（二维场景恒为 1） | `1` |
| `d1` | 列分块数 `n1` | `cols / C0` |
| `d2` | 行分块数 `m1` | `rows / 16` |
| `d3` | fractal 内行数 | `16`（`FRACTAL_NZ_ROW`） |
| `d4` | fractal 内列数 `C0` | `32 / sizeof(T)` |

fp32 且 `(rows, cols) = (128, 64)`（`C0 = 8`）：

```
shape  = [1, 8, 8, 16, 8]
stride = [8192, 1024, 128, 8, 1]
```

**关键结构不变量**（后文规则全部由它推出）：

```
shape[3] == 16
shape[4] == C0            <=> shape[4] * sizeof(T) == 32
shape[3] * shape[4] * sizeof(T) == 512      // 一个 fractal = 512B
stride[4] == 1
stride[3] == C0           == shape[4]
stride[2] == 16 * C0      == shape[3] * shape[4]
stride[1] == shape[2] * stride[2]
stride[0] == shape[1] * stride[1]
```

## 2. PTOAS 现状

### 2.1 同一条规则在三处重复实现

| 位置 | 函数 | 用途 |
|---|---|---|
| `lib/PTO/Transforms/InferPTOLayout.cpp:174` | `inferNZLayout()` | `pto-infer-layout` pass，给 `make_tensor_view` / `reinterpret_cast` / `subview` / `tload` / `tstore` 打 `layout` 属性 |
| `lib/PTO/IR/PTO.cpp:1836` | `inferLayout()`（`getLogicalViewLayout` 调用） | verifier 侧判断逻辑 layout（如 mgather/mscatter 的 ND 约束，`lib/PTO/IR/PTO.cpp:4366`） |
| `lib/PTO/Transforms/PTOToEmitC.cpp:4350` | `inferFallbackGlobalTensorLayout()` | EmitC 兜底：`layout` 属性缺失时重新推断 |

三份实现的 NZ 判定条件完全一致（右对齐到 5 维后）：

```cpp
shape[2] == 16
shape[2] * shape[3] * elemBytes == 512
stride[4] == 1
stride[3] == shape[4]
```

另有 `lib/PTO/Transforms/PTOCanonicalizeIR.cpp:88` 附近的 rank2 → rank5 规范化，
与 `rightAlignTo5D()` / `buildGlobalTensorShapeAndStride()` 共用同一套 padding 规则，
一并纳入"单一实现"的收敛范围。

### 2.2 现规则 = 标准形规则"错位一维"

把现规则与第 1 节不变量并排看：

| 条件 | 现规则 | pto-isa 标准形 | 结论 |
|---|---|---|---|
| fractal 行数 | `shape[2] == 16` | `shape[3] == 16` | **错位一维** |
| fractal 字节 | `shape[2] * shape[3] * eb == 512` | `shape[3] * shape[4] * eb == 512` | **错位一维** |
| 最内连续 | `stride[4] == 1` | `stride[4] == 1` | 一致 |
| C0 连续 | `stride[3] == shape[4]` | `stride[3] == shape[4]` | 一致 |
| fractal 跨度 | 未检查 | `stride[2] == shape[3] * shape[4]` | 缺失 |
| 分块跨度 | 未检查 | `stride[1] == shape[2] * stride[2]` | 缺失 |
| batch 跨度 | 未检查 | `stride[0] == shape[1] * stride[1]` | 缺失 |

两条 stride 条件本来就是对的，**只有两条 align 条件整体错了一维**，再叠加"外层
stride 完全不校验"，于是同时产生了漏判（标准形判成 ND）和误判（连续 ND 判成 NZ）。

### 2.3 复现（本地实测，非推演）

用不依赖 python binding 的等价 `.pto` 复现 issue 的两个用例：

```mlir
// 用例①：pto-isa canonical 2D NZ 五维形
%view = pto.make_tensor_view %dst,
  shape = [%c1, %c8, %c8, %c16, %c8],
  strides = [%c8192, %c1024, %c128, %c8, %c1]
  : !pto.tensor_view<1x8x8x16x8xf32>
%part = pto.partition_view %view,
  offsets = [%c0, %c0, %c0, %c0, %c0], sizes = [%c1, %c8, %c8, %c16, %c8]
  : !pto.tensor_view<1x8x8x16x8xf32> -> !pto.partition_tensor_view<1x8x8x16x8xf32>
%tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=128, cols=64,
    v_row=128, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>
pto.tstore ins(%tile : ...) outs(%part : !pto.partition_tensor_view<1x8x8x16x8xf32>)
```

`ptoas --pto-arch=a5` 实际输出：

```cpp
// ① canonical NZ 形 -> 被判成 ND（错）
GlobalTensor<float, pto::Shape<1, 8, 8, 16, 8>, pto::Stride<8192, 1024, 128, 8, 1>,
             pto::Layout::ND>

// ② 迎合现规则的形状 -> 被判成 NZ（本身是连续 ND 数据）
GlobalTensor<float, pto::Shape<4, 1, 16, 8, 16>, pto::Stride<2048, 2048, 128, 16, 1>,
             pto::Layout::NZ>
```

**第三个症状（issue 未提，但更致命）**：用户显式标注也救不回来。给用例①的
`make_tensor_view` 加上 `{layout = #pto.layout<nz>}`：

```
error: layout mismatch: user-specified layout=nz but inferred=nd
```

来源是 `lib/PTO/Transforms/InferPTOLayout.cpp:275` 的 `verifyOrSetLayoutAttr()`：
推断结果被当作"真值"去否决用户显式声明。也就是说目前**既推不出 NZ，也没有逃生通道**。

## 3. 一个必须先承认的约束

把第 1 节的 stride 展开：

```
stride[2] = shape[3] * stride[3]
stride[1] = shape[2] * stride[2]
stride[0] = shape[1] * stride[1]
```

这正是五维**连续（行主序累积积）**的定义。结论：

> **NZ 五维标准形，在 shape/stride 数值上与"同 shape 的连续 ND 五维视图"完全相同，
> 无法区分。**

差异只存在于"这 5 个维度分别代表什么"这个语义层面，不存在于内存 pattern 层面。
由此推出两条设计原则：

- **P1：layout 应该被"携带"，而不是被"猜"。** 显式 `layout` 属性必须是权威来源，
  推断只是缺省兜底。
- **P2：纯 pattern 推断必须带消歧门槛。** 否则任何"末两维恰好是 `(16, 32/sizeof(T))`
  的连续视图"都会被升级成 NZ —— 包括最常见的 `16 x C0` 二维小 tile。

P2 不是理论担忧，第 5 节有实测数据。

## 4. 设计方案

### 4.1 收敛为唯一实现

新增 `include/PTO/IR/PTOLayoutUtils.h` + `lib/PTO/IR/PTOLayoutUtils.cpp`，导出：

```cpp
namespace mlir::pto {

// C0 = 32 / elemBytes；elemBytes 不能整除 32 时返回 nullopt（sub-byte / packed 类型）
std::optional<int64_t> getNZC0Elems(unsigned elemBytes);

// 结构必要条件：shape/stride 是否是 pto-isa NZ 五维标准形
bool isNZCompatible5D(ArrayRef<int64_t> shape5D, ArrayRef<int64_t> stride5D,
                      unsigned elemBytes);

struct LayoutInferOptions {
  bool allowNZFromPatternOnly = true;   // 无外部证据时是否允许纯 pattern 判 NZ
  std::optional<Layout> preferredMinor2D;  // 现有 ND/DN 歧义消解入口
};

std::optional<Layout> inferLayout5D(ArrayRef<int64_t> shape5D,
                                    ArrayRef<int64_t> stride5D,
                                    unsigned elemBytes,
                                    const LayoutInferOptions &opts,
                                    bool *isMinor2DAmbiguous = nullptr);
} // namespace mlir::pto
```

`InferPTOLayout.cpp` / `PTO.cpp` / `PTOToEmitC.cpp` 三处改为调用同一实现，
删除各自的私有副本。这一步是纯重构（NFC），单独成 PR，便于回归定位。

### 4.2 `isNZCompatible5D`：结构必要条件

```cpp
bool isNZCompatible5D(shape, stride, elemBytes) {
  auto c0 = getNZC0Elems(elemBytes);            // 32 % elemBytes != 0 -> false
  if (!c0) return false;
  return shape[3] == 16 && shape[4] == *c0
      && stride[4] == 1
      && stride[3] == *c0
      && stride[2] == 16 * *c0
      && stride[1] == shape[2] * stride[2]
      && stride[0] == shape[1] * stride[1];
}
```

用途有两个，且**两个用途门槛不同**（这是本设计的核心）：

- 校验显式 `layout = nz` 是否自洽 —— 只看这个谓词；
- 无显式 layout 时的推断 —— 还要过 4.3 的门槛。

### 4.3 推断门槛：只认"不可能是二维 ND 视图"的形状

```cpp
bool inferNZFromPattern(shape, stride, elemBytes) {
  return isNZCompatible5D(shape, stride, elemBytes)
      && shape[0] == 1                        // 二维 NZ，batch 维恒为 1
      && (shape[1] > 1 || shape[2] > 1);      // 至少有一个分块维非退化
}
```

两个附加条件的理由：

- `shape[0] == 1`：issue 要求的是**二维** NZ；`batch > 1` 的形状在 pto-isa 侧没有
  对应的 `TileShape2D`，留给显式标注，避免把连续五维 ND 张量整片吃掉。
- `shape[1] > 1 || shape[2] > 1`：`[1,1,1,16,C0]` 是 rank2 视图规范化后的标准产物
  （`PTOCanonicalizeIR`），也是最常见的 `16 x C0` 向量 tile。此时 NZ 与 ND 的字节
  排布**完全等价**（单 fractal），判成 NZ 没有任何收益，却会改变生成的 C++ 模板参数、
  进而影响 pto-isa 侧的重载选择。实测这条门槛消除了 100% 的新增误判（第 5 节）。

保留 `LayoutInferOptions::allowNZFromPatternOnly`，是为了给"上层已经有更强证据"的
调用点（例如 mgather/mscatter 的 ND-only 校验路径）一个关掉 pattern 推断的开关。

### 4.4 显式 layout 优先

`verifyOrSetLayoutAttr()` 的语义调整为：

| 显式属性 | 结构自洽性 | 现行为 | 新行为 |
|---|---|---|---|
| `nz` | `isNZCompatible5D == true` | **报错** | 接受，保留 `nz`，不打 `pto.inferred_layout` |
| `nz` | `isNZCompatible5D == false` | 报错 | 报错，但错误信息升级为"哪一条不变量不满足" |
| `nd`/`dn` | 与推断不同 | 现有 minor-2D 歧义豁免 | 不变 |

这条直接解决 2.3 的第三个症状：即使 pattern 推断因门槛保守而不升级 NZ，用户/前端
也永远有一条显式通道。诊断信息形如：

```
error: layout mismatch: user-specified layout=nz but shape/stride is not an NZ
       5D form: expected shape[3]==16 (got 8), shape[4]==8 (C0 for f32, got 16)
```

### 4.5 `partition_view` / `memref.subview` 的 NZ 传播

现状：`InferPTOLayout.cpp` 的 subview 分支无条件继承源 layout。对 NZ 这是不安全的
—— 在 fractal 内部切分后，结果已经不是合法 NZ。规则改为：

1. 源为 NZ 时，只有当切分满足以下条件才继承 NZ：
   - `d3`/`d4` 维保持完整（offset 为 0、size 等于源 size）；
   - `d1`/`d2` 维的 size 可缩小，但 stride 保持不变；
   - offset 在 `d1`/`d2` 上按整块对齐。
2. 不满足时不 silently 退回 ND，而是发 `emitError`（"NZ view cannot be partitioned
   inside a fractal"），避免生成静默错码。

issue 明确提到 `partition_tensor_view`，这条属于本次范围内。

### 4.6 动态形状（阶段二）

pto-isa 的 `TileShape2D`/`BaseShape2D` 允许 `rows`/`cols` 为 `DYNAMIC`，此时
`shape[1]`、`shape[2]`、`stride[0]`、`stride[1]` 动态，而结构维
（`shape[3]`、`shape[4]`、`stride[2..4]`）仍是静态常量。因此可以在
"结构维静态 + 分块维动态"时仍判定 NZ。当前实现要求全部 const-fold，直接放弃推断
（`getStaticShapeAndStride()` 返回 false），属于可独立推进的增强，放到阶段二。

### 4.7 与旧规则的兼容策略

| 方案 | 说明 | 评价 |
|---|---|---|
| A. 直接替换 | 只保留标准形规则 | **推荐**。旧规则命中的非标准形本质是误判，继续保留会持续生成错误的 `GlobalTensor` 模板参数 |
| B. 并集（新规则 OR 旧规则） | 兼容一切现有行为 | 不推荐：把"连续 ND 判成 NZ"固化成契约 |
| C. 替换 + 一个版本的过渡开关 | 加 `--pto-legacy-nz-infer`（默认关） | 若下游有存量依赖再启用；本次实测影响面极小（第 5 节），倾向不引入 |

推荐 A，若 review 中出现下游存量用例再降级到 C。

## 5. 影响面实测

方法：用当前 `ptoas` 对 `test/lit/pto/*.pto`（476）+ `test/lit/tile_fusion/*.pto`（42）
分别按 `--pto-arch=a5` 和 `--pto-arch=a3` 跑一遍，抽取全部生成的
`GlobalTensor<elem, Shape<...>, Stride<...>, Layout::X>`（含 `using GTShape_*` 别名形式），
再离线对同一批 shape/stride 分别套用旧规则、标准形规则、带门槛的标准形规则。

样本：242 个文件产生输出，共 **1558** 个 `GlobalTensor` 实例化点，
**64** 组去重后的 `(elem, shape, stride)`。当前 layout 分布：ND 1007 / NZ 529 / DN 22。

| 项 | 结果 |
|---|---|
| 当前判为 NZ 的去重形状 | 5 组 |
| 其中符合 pto-isa 标准形（改后仍为 NZ） | 2 组：`half`/`bfloat16_t` `[1,1,16,16,16]` `stride=[4096,4096,256,16,1]` |
| 其中由显式属性设置（不走推断，不受影响） | 1 组：`tinsert_a5_vec_mat_mode_lowering` 的 `half [1,1,1,32,32]` |
| **改后会失去 NZ** | 2 组：`int8_t [1,1,16,32,16]`、`int64_t [1,1,16,4,1]` —— 全部只出现在 `test/lit/pto/globaltensor_layout_bytewidth_emitc.pto` |
| 不带门槛时新增 NZ | 4 组 / 80 个点 / 38 个文件，**全部**是 `[1,1,1,16,C0]` 退化单 fractal（`float`/`half`/`float8_e4m3_t`/`hifloat8_t`） |
| **带门槛（4.3）新增 NZ** | **0** |

两点解读：

1. 门槛条款把误判从 80 个点压到 0，验证了 P2 的必要性。
2. 唯一需要改期望值的测试是 `globaltensor_layout_bytewidth_emitc.pto`
   （`comm.tbroadcast` 的 i8/i64 用例）。这两组形状按 pto-isa 定义本来就不是 NZ
   （i8 的 `C0 = 32`，标准形应为 `[1, cols/32, rows/16, 16, 32]`），当前期望值本身
   固化了 2.2 的错位 bug。改测试前需确认 `comm.tbroadcast` 侧对 `Layout` 是否敏感。

另需交叉验证：`PTO.cpp:4366` 的 mgather/mscatter "mem partition view 必须是 ND"校验
走的是同一套推断。规则收紧后，理论上存在"原本判 ND 的五维 mem view 变成 NZ 导致新
报错"的可能；带门槛后本地 lit 语料未观察到，但需要在实现 PR 里补一条针对性用例。

## 6. 实施拆分

| PR | 内容 | 风险 |
|---|---|---|
| P1 | 抽出 `PTOLayoutUtils`，三处调用点收敛，行为完全不变（NFC） | 低 |
| P2 | 标准形规则 + 4.3 门槛 + 4.7 方案 A；更新 `globaltensor_layout_bytewidth_emitc.pto` | 中 |
| P3 | 4.4 显式 layout 优先 + 诊断信息细化 | 低 |
| P4 | 4.5 partition/subview NZ 传播校验 | 中 |
| P5 | 4.6 动态形状支持（可选） | 中 |

跨层同步检查（按 `.claude/rules/cross-layer-sync.md`）：本设计不改 ODS 算子签名，
`PTO_LayoutAttr` 已存在 `nz`；需要同步的是 IR verifier（`PTO.cpp`）、pass
（`InferPTOLayout.cpp`）、EmitC（`PTOToEmitC.cpp`）、以及 python 侧
`make_tensor_view` 的 `layout` 传参样例与文档。

## 7. 测试计划

新增 `test/lit/pto/issue527_nz_canonical_view_infer.pto`（a5 + a3 双 RUN），覆盖：

1. **正例**：canonical `[1,8,8,16,8]` fp32 → `pto::Layout::NZ`；
2. **正例**：`half` `[1,4,8,16,16]`、`int8_t` `[1,2,8,16,32]` → NZ（验证 C0 随 dtype 变化）；
3. **负例**：issue 用例② `[4,1,16,8,16]` → `ND`（旧规则误判被修掉）；
4. **负例**：`[1,1,1,16,8]` 单 fractal → 保持 `ND`（门槛条款，防回归 80 个点）；
5. **负例**：stride 被打断（如 `stride[1]` 非 `shape[2]*stride[2]`）→ `ND`；
6. **显式通道**：canonical 形 + `{layout = #pto.layout<nz>}` → 不再报
   `layout mismatch`，且输出 NZ；
7. **显式冲突**：`[4,1,16,8,16]` + `{layout = #pto.layout<nz>}` → 报错且信息指明
   是哪条不变量不满足；
8. **partition**：对 NZ 视图在 `d1` 上整块切分 → 继承 NZ；在 `d3` 内部切分 → 报错。

E2E：`test/tilelang_st/npu/a5` 下补一条 NZ store 用例（若板卡资源允许），
用 `nz_store_probe.py` 的逻辑做数据比对，确认生成的 C++ 与 pto-isa 语义一致。

## 8. 待确认问题

1. **是否接受 4.7 方案 A**（直接替换旧规则）？影响面只有一个 lit 测试的 2 组期望值。
2. **`batch > 1` 的 NZ**（`shape[0] > 1`）是否需要推断？当前设计只在显式标注时接受。
3. **sub-byte / packed 类型**（`int4`、`float4_e1m2x2`）：`32 % elemBytes` 语义如何定义？
   当前设计直接不推断 NZ，需 pto-isa 侧确认是否存在这类 NZ 用法。
4. `comm.tbroadcast` 路径对 `GlobalTensor` 的 `Layout` 模板参数是否敏感？
   决定 `globaltensor_layout_bytewidth_emitc.pto` 的期望值怎么改。
5. `mgather`/`mscatter` 的 "mem 必须 ND" 约束，在 mem 确实是 NZ 五维时是否应放开，
   还是维持报错（本设计维持现状，仅补测试）。
