<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# TEXTRACT ND to 2xNZ 双输出接口设计

## 1. 范围与基线

本文定义 PTOAS 对 PTO-ISA `TEXTRACT` 双输出 ND-to-2xNZ overload 的完整承载方案，
覆盖 PTO IR、verifier、EmitC、DPS、内存效应、同步、内存规划、TileLib、依赖 pin、
文档和测试。

本文是设计文档，不包含功能实现。

核对时间为 2026-08-17，使用的最新版基线如下：

- PTOAS：`hw-native-sys/PTOAS main@39c601fe386fa423098abbaddf4ab7b584179510`。
- PTO-ISA GitCode：`cann/pto-isa master@bb69abb8a3f71192a125bf14f909133c41f3a519`。
- PTO-ISA GitHub mirror：
  `hw-native-sys/pto-isa main@52d4ad3228ff69ea6e2d4a68305b95e51c81be2d`。
- 原始功能提交：
  - GitCode `90e9d50caf8ec107feb9bac970892130e4f6e985`；
  - GitHub mirror 对应提交 `29a8eadbdf2c879e0580482ed14d2e8f6871f096`；
  - 提交时间为 2026-08-14 18:02:50 +0800；
  - 标题为 `textract nd->2x nz vf operators a2a3 and a5 NZ and NZ+1`。

最新版 PTO-ISA 的 A2/A3、A5 头文件和两套 NPU ST 仍包含该接口；8 月 17 日的最新
`TEXTRACT` 变更只修改 GEMV sub-fractal 单输出路径，没有移除或改变本 overload。
CPU backend 当前没有 `TEXTRACT_ND2XNZ_IMPL`，cost-model 的独立 `pto_instr.hpp` 也没有
七参数 overload；不能把 common header 中公开 overload 的存在等同于所有 backend 都可实例化。

当前 PTOAS `TExtractOp` 只有一个 `$dst`，并且仓库内没有 ND-to-2xNZ 的 ODS、
lowering、verifier、TileLib template 或回归测试，因此缺口仍然存在。

## 2. 最终决策

1. 新增独立的 `TExtractNd2xNzOp`，文本名为 `pto.textract.nd2xnz`。它是
   `TEXTRACT` 指令族下的固定双输出语义形态，不修改现有 `TExtractOp`。
2. op 固定携带一个 ND source、两组独立 index 和两个 NZ DPS destination；两个
   destination 可以有不同 physical/valid shape，但 element type 必须与 source 相同。
3. op 实现 `PTO_DpsInitOpInterface`，两个 destination 都是 DPS init；不增加 SSA result。
4. op 的 pipe 固定为 `PIPE_V`；内存效应为 `Read(src)`、`Write(dst0)`、`Write(dst1)`。
5. `src`、`dst0`、`dst1` 三者必须两两不重叠。legacy 和 modern PlanMemory 都必须执行
   该 no-alias 契约；level3 的显式地址必须通过独立的静态地址 gate。
6. 首版不支持 runtime-bound tile provenance。`DeclareTileOp`、`TAssignOp`、`TPopOp`/
   `TPopFromAicOp`/`TPopFromAivOp` 绑定或产生的 tile，以及它们的 view/subview/cast 派生值，
   在所有 level 都必须在 PlanMemory 之前被拒绝；只有 planner-owned `alloc_tile` 或已物化的
   `alloc_multi_tile` slot 才能作为该 op 的 local tile provenance；level1/2 的地址由 planner
   产生，level3 则要求调用方提供可静态证明的地址。这样 no-alias 契约不会依赖
   planner 对 runtime handle 的猜测。
7. EmitC 精确生成七参数公开 API：
   `TEXTRACT(dst0, dst1, src, indexRow0, indexCol0, indexRow1, indexCol1)`。
   不直接生成内部名字 `TEXTRACT_ND2XNZ_IMPL`。
8. A2/A3 与 A5 共用 IR 形态，verifier 按 target arch 分派 dtype 和 compact mode 规则；
   只有通过目标 backend compile 和数值测试的组合才进入正向支持集合。
9. A5 TileLib 增加独立 template；A2/A3 当前没有 TileLib 目录，继续走 tile-level
   EmitC 路径。
10. 实现 PR 必须把所有实际实例化本 overload 的 PTO-ISA pin 升到包含该接口及对应 backend
   implementation 的后继提交。pin 更新必须逐 remote 做 ancestry、API 和编译验证，不允许
   跨 remote 比较 SHA；只含公开 API、缺少 CPU implementation 的 revision 不能宣称 CPU-sim 支持。
11. partial-valid/odd/`1x1` 是定义明确的 TEXTRACT/UB-only 支持，不自动获得 NZ TSTORE
    支持；完整 TSTORE 只对 full-valid destination 宣称，除非另有经过验证的
    full-valid materialization。
12. CPU-sim、cost-model 和其他 optional backend 的 unsupported 判断必须来自 driver 注入的
    capability manifest，并在最终 backend lowering 前失败，不能依赖后期 C++ 实例化。

## 3. PTO-ISA 真实契约

### 3.1 公开 API

最新版 `include/pto/common/pto_instr.hpp` 中的接口为：

```cpp
template <
    typename Dst0TileData, typename Dst1TileData, typename SrcTileData,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<SrcTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT(
    Dst0TileData& dst0, Dst1TileData& dst1, SrcTileData& src,
    uint16_t indexRow0 = 0, uint16_t indexCol0 = 0,
    uint16_t indexRow1 = 0, uint16_t indexCol1 = 0,
    WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TEXTRACT_ND2XNZ, dst0, dst1, src,
                 indexRow0, indexCol0, indexRow1, indexCol1);
  return {};
}
```

因此 PTOAS 必须保留以下事实：

- 两个 window 的 index 相互独立；
- 两个 destination 的类型是独立 C++ template parameter，shape 不要求相同；
- 两个 destination 与 source 属于同一次公开 `TEXTRACT` 调用；
- wait event 不进入 PTO IR operand，仍由 PTOAS 的同步 pass 管理。

### 3.2 数值语义

对 `k in {0, 1}`，逻辑 window 为：

```text
window_k[r, c] = src[indexRow_k + r, indexCol_k + c]
  0 <= r < dst_k.validRows
  0 <= c < dst_k.validCols
```

`window_k` 写入 `dst_k` 的 NZ 排布。令 `c0 = 32 / sizeof(T)`，plain NZ 的核心线性
偏移为：

```text
dstOffset = floor(c / c0) * dstPhysicalRows * c0
          + r * c0
          + (c % c0)
```

A5 `RowPlusOne` 在相邻 NZ column block 之间增加一行 bank-conflict padding；padding
不是逻辑输出，`valid_shape` 不随之增大。destination 的 valid 区域之外不定义新值，
调用方不能依赖未写 padding 的内容。

两个 window 可以重叠读取同一 source 区域；这不影响语义。两个 destination 不能互相
重叠，也不能与 source 重叠，因为原生实现按 window 顺序读写，alias 会破坏尚未读取的
source 或使两个可观察输出互相覆盖。

### 3.2.1 partial-valid 与 TSTORE 边界

`TEXTRACT` 的 partial-valid 语义只承诺上面的 logical window。它不承诺对
`valid_shape` 以外的物理元素、NZ block 尾部或 `RowPlusOne` gap 写入任何值。实现和
测试不得把“目的 tile 之前恰好被清零”推导成 TEXTRACT 的输出保证。

因此实现 PR 将测试和 codegen 明确分成两个模式：

1. **full-store eligible**：对每个 destination 都有
   `validRows == physicalRows` 且 `validCols == physicalCols`（对 A5
   `RowPlusOne` 还必须使用已经验证过的 emitted virtual-row 适配器）。该模式才生成
   `TLOAD -> TEXTRACT -> TSTORE`。plain NZ 的二维 destination 使用确定的 canonical
   GlobalTensor NZ shape：

   ```text
   [1, physicalCols / c0, physicalRows / 16, 16, c0]
   ```

   于是 PTO-ISA 的 NZ TSTORE 断言
   `validRow == shape[2] * shape[3]`、
   `validCol == shape[0] * shape[1] * shape[4]` 在 debug 和 release 都成立；GlobalTensor
   strides 再按真实 GM view 填入，不能用 valid shape 代替 physical shape。

2. **partial-valid / UB-only**：允许 `valid != physical`、odd `validCol` 和 `1x1`。
   测试 harness 在 TEXTRACT 前显式把两个 destination 的完整物理 UB footprint 初始化为
   固定字节模式（默认 zero）；该初始化是 harness/codegen helper 的前置动作，不是
   TEXTRACT 的隐式语义，也不能用 `TFILLPAD` 代替对整个物理 footprint 的清零。该模式
   不把 partial descriptor 直接传给 TSTORE，也不把未定义区作为 production 输出写回 GM；
   后文 test-only full-valid dump alias 可以单独执行观测 TSTORE。golden 只比较两个
   destination 各自的 valid logical region。预初始化值只用于避免读取未初始化内存和诊断
   越界写，未定义区不参与数值比较。

当前 PTOAS 没有一个能把静态 partial tile 安全地改成 full-valid tile 的现有操作：
`pto.set_validshape` 仅接受 `v_row=?/v_col=?` 的本地动态 tile，并且只修改运行时元数据。
因此首版不把同一个 partial descriptor 伪装成 full-valid，也不在 generic `TSTORE` 中放宽契约。
对于 partial-valid destination，若调用方强行把它直接接入 TSTORE，新增的链路验证必须在
PTOAS 阶段报错：

```text
pto.textract.nd2xnz partial-valid destination is not eligible for NZ TSTORE;
use UB-only validation or provide a proven full-valid materialization
```

NPU ST 需要一个可编译的 UB 观测路径，但不能依赖目前并不存在的 raw-UB dump API。实现 PR
在 NPU testcase support 中增加 test-only helper `prepareAndDumpPartialNzForTest`，固定使用
level3 静态地址和现有公开指令，流程如下：

1. 对每个 partial destination 同时建立 `partial descriptor` 和 `full-valid dump alias`；二者
   physical shape、dtype、layout、compact mode 完全相同，且两个 `pto.alloc_tile` 使用同一个
   非负静态 UB address。partial descriptor 保留测试所需 valid shape，dump alias 的
   `validRows/validCols` 则等于 physical extent。两个 destination 各用一组互不重叠的地址。
2. 用 canonical physical-size NZ GlobalTensor 和 `TLOAD` 把 full-size zero GM tensor 载入
   dump alias，从而初始化该 destination 的完整物理 UB footprint。不能用 `TASSIGN` 构造这条
   alias，因为 runtime-bound provenance gate 会且应该拒绝它。
3. 用 partial descriptor 执行一次双输出 `TEXTRACT`。InsertSync/GraphSync 必须从两组 alias 的
   相同静态 absolute range 推导 `TLOAD -> TEXTRACT` 的 WAW 依赖；fixture 不能依赖源码顺序
   恰好生效。
4. 用 full-valid dump alias 执行 generic `TSTORE` 到独立的 physical-size GM output。传给
   TSTORE 的 descriptor 是 full-valid，因此 PTO-ISA debug assertion 保持开启也成立；
   `PTOValidateNd2xNzStoreUsePass` 仍会拒绝 partial descriptor 自身的直接 TSTORE use。
5. host golden 只从 physical GM output 解码和比较 partial descriptor 的 valid logical
   coordinates。valid 区外、NZ tail 和 `RowPlusOne` gap 均不比较；GM output 两侧的 guard bytes
   单独验证，以发现越界写。

该 helper 生成普通、可在 A2/A3/A5 编译执行的 `TLOAD/TEXTRACT/TSTORE` kernel，不要求
simulator 特权接口。full-valid alias 只属于测试 fixture 的观测适配器，不扩大 production
partial-valid TSTORE 语义，也不把清零后的 padding 当作 TEXTRACT 输出。

后续若要支持 partial-valid 的完整写回，必须先增加一个经过 backend 验证的
full-valid materialization（动态 valid tile、物理 extent 和 `SetValidShape` 的顺序均需
有 IR/EmitC/VPTO 语义），或者让 PTO-ISA 为 partial NZ TSTORE 提供明确的 padding 语义并
移除上述 debug assertion。两者落地前，`1x1` 和 odd-valid case 只能宣称 TEXTRACT/UB
覆盖，不能计入完整 TSTORE coverage。

### 3.3 共同结构约束

PTO-ISA A2/A3 与 A5 的本 overload 都直接检查：

- source 和两个 destination 均为 `TileType::Vec`，即 PTOAS `loc=vec`；
- source 是 ND：`BLayout::RowMajor` + `SLayout::NoneBox`；
- destination 是 NZ：`BLayout::ColMajor` + `SLayout::RowMajor`；
- source/destination element type 相同；
- source row stride 的 byte 数为 32B 对齐；
- destination physical cols 是 `c0` 的整数倍；
- 每个 window 分别满足：

```text
indexRow_k + dst_k.validRows <= src.physicalRows
indexCol_k + dst_k.validCols <= src.physicalCols
```

这里 bounds 使用 destination 的 valid shape 和 source 的 physical shape，不能复用
当前单输出 helper 中的 `dst.physicalShape` 检查。

destination 的 `validRows`、`validCols` 不要求等于 physical shape，也不要求都是
fractal 倍数。plain NZ 的 PTOAS physical rows 仍须按 16 rows 对齐，使类型能被 NZ
GlobalTensor/TSTORE 链路承载；这是完整存储链路约束，不是上述 overload 自己的
`CheckTExtractNdToNz` static assert。PTO-ISA 已覆盖 `1x1`、非对齐 index，以及 A2/A3
`int8` odd validCol；PTOAS verifier 不得添加这些上游不存在的限制。

### 3.4 架构差异

| 约束 | A2/A3 | A5 |
|---|---|---|
| header dtype 集合 | `i8`, `i32`, `f16`, `bf16`, `f32` | A2/A3 集合，加 `hif8`, `f8E4M3`, `f8E5M2`, `f8E8M0`, `f4E2M1x2`, `f4E1M2x2` |
| plain NZ | 支持 | 支持 |
| NZ+1 / `RowPlusOne` | 不支持 | 支持 |
| 非 32B source base | scalar fallback | SIMD unaligned path |
| `1x1` | scalar path | scalar path |
| `i8` odd validCol | f16 widen/reshape/narrow | 原生 byte SIMD |

A2/A3 的 `indexCol * sizeof(T)` 不满足 32B 对齐时会走 scalar fallback，而不是非法输入；
A5 的 sub-c0 `indexCol` 由 `vldas`/`vldus` 路径处理。因此 verifier 只验证 bounds，不验证
index 对齐。

### 3.5 FP4 维度域

PTOAS 的 `!pto.f4E*M*x2` 是一个 byte 存两个 FP4 的 packed type。EmitC 生成 Tile type时，
`renderTileTemplateDim` 会把 packed dimension 放大 2 倍：

- RowMajor ND 的 packed dimension 是 column；
- ColMajor NZ 的 packed dimension 是 row。

所以 FP4 的 alignment 和 bounds 必须在“最终生成的 PTO-ISA Tile dimension”上校验，
不能直接拿 raw PTO IR shape 与普通 byte dtype 共用公式。实现应把
`renderTileTemplateDim` 的维度归一化逻辑提取到共享 PTO type utility，并区分 physical
与 valid dimension，例如：

```cpp
int64_t getPTOIsaPhysicalTileDim(TileBufType tile, unsigned dim);
int64_t getPTOIsaValidTileDim(TileBufType tile, unsigned dim);
```

verifier 和 EmitC 都调用同一 helper，防止一边校验 raw dim、另一边输出 doubled dim。

但 header 支持列表不等于已经验证：8 月 14 日新增的 A5 ST 没有实例化两种 FP4。当前
A5 implementation 又固定对 `validCol/indexCol` 除 2，而 PTOAS 对 ColMajor NZ 的 packed
dimension 是 row；这条轴向和 source row-stride 必须先用最小生成 C++ 与设备 golden
确认。第一版 verifier 在验证完成前拒绝 FP4，诊断为“目标 PTO-ISA revision 的 ND-to-2xNZ
FP4 path 尚未验证”；验证通过后再把两种 FP4 加入正向集合。不能只靠 static assert
成功或 EmitC 文本正确就放行。

### 3.6 当前 backend 与测试缺口

最新版源码存在四项必须在实现 PR 中显式处理的缺口：

1. `include/pto/cpu/TExtract.hpp` 没有 `TEXTRACT_ND2XNZ_IMPL`。在 CPU-sim 下实例化七参数
   overload 会编译失败；PTOAS 不能承诺 CPU-sim 数值支持，除非先有对应 PTO-ISA upstream
   implementation 并升级 GitHub pin。
2. `include/pto/costmodel/pto_instr.hpp` 没有七参数 overload，cost-model build 同样不能
   实例化生成代码；首版必须标记为 unsupported，或先补齐 upstream API 与 latency model。
3. A5 新 ST 的 template 虽然有 `CompactMode` 参数，但实际 dispatch 只实例化
   `CompactMode::Null`，没有执行 `RowPlusOne`。
4. A5 新 ST dispatch 到 `float8_e8m0_t` 为止，没有实例化两种 FP4。

因此测试矩阵把“header 声明”“NPU backend 可编译”“NPU 数值已验证”“CPU backend 可用”
分开记录。FP4、RowPlusOne 和 CPU-sim 不能因提交标题或类型白名单被自动标为 supported。

unsupported gate 必须有可执行载体，不能留在 op verifier 的架构白名单里。实现 PR 增加
driver 注入的 codegen environment/capability contract：

- driver 新增两个输入：

  ```text
  --pto-codegen-env=npu|cpu-sim|costmodel
  --pto-capability-manifest=<path>
  ```

  `--pto-capability-manifest` 是唯一 manifest 来源；不从环境变量、当前工作目录或输入 IR
  属性猜测。CMake/harness 在实际 PTO-ISA include root 上运行 compile probe，把 JSON 写入
  build tree（例如 `${CMAKE_BINARY_DIR}/pto/capabilities/<environment>-<backend>-<arch>.json`），
  再把该绝对路径传给 `ptoas`。最终生成 C++ 的 harness 必须复用 manifest 选出的同一个
  `include_root`，不能 probe 一个头文件树、编译时再换另一棵树。
- manifest 固定为 `schema_version = 1`、带 `entries` 的 JSON；每个 entry 的 key 是
  `(environment, backend, arch)`，并记录 probe 实际使用的 PTO-ISA revision、canonical
  include root 和输入树指纹：

  ```json
  {
    "schema_version": 1,
    "entries": [{
      "environment": "cpu-sim",
      "backend": "emitc",
      "arch": "a3",
      "pto_isa_revision": "<remote-local-sha>",
      "include_root": "/abs/path/to/pto-isa/include",
      "include_tree_sha256": "sha256:<probe-input-digest>",
      "capabilities": {
        "textract.nd2xnz": {
          "supported": false,
          "probe": "missing pto::TEXTRACT_ND2XNZ_IMPL"
        }
      }
    }]
  }
  ```

  对 `vpto`，probe 验证 TileLib/template 注册、VPTO lowering 和 intrinsic verifier；对
  `emitc` 的 NPU/CPU-sim/cost-model，probe 验证生成 C++ 对上述实际 include root 的编译。
- driver 读取 manifest 后必须：校验 JSON/schema；按 effective environment、backend、arch
  精确选择且只能选择一个 entry；canonicalize 并检查 `include_root` 存在；重新计算
  `include_tree_sha256`；并把 `pto_isa_revision` 与 CMake/harness 为当前编译任务配置的
  PTO-ISA revision 校验。实现 PR 由 CMake 从同一依赖选择生成只读的
  `PTOAS_PTO_ISA_INCLUDE_ROOT`/`PTOAS_PTO_ISA_REVISION` build configuration；driver 将
  manifest entry 与这两个值及 digest 一起核对，不依赖 PTO-ISA checkout 必须带 `.git`。
  缺少 configured revision、include root 不可读、digest/revision 不匹配都是 stale manifest，
  必须在 PTOAS 阶段失败。解析出的 include root/revision 同时回传给最终 C++ 编译 harness，
  形成 probe 与实际编译的一致性闭环。
- 校验成功后，driver 才写入 module attributes `pto.codegen_env` 和
  `pto.codegen_capabilities`。二者是 driver 保留属性；输入 IR 中的同名属性必须被拒绝，
  不能手写 IR 自行宣称 capability。manifest 缺失时，`--emit-pto-ir` 可保留
  `unknown` 并输出 IR；任何 EmitC/VPTO strict final-codegen path 都必须要求该文件并失败。
- 在 generic `mlir::verify` 之后、EmitC/VPTO 最终 lowering 之前运行
  `PTOValidateCodegenCapabilitiesPass`。该 pass 对每个 `pto.textract.nd2xnz` 查已注入 entry；
  environment unknown、entry 缺失或 capability 为 false 时，报告 backend、environment、
  arch、required capability、PTO-ISA revision、manifest path 和 probe 诊断，不把错误延迟到
  C++ 模板实例化。
- CPU-sim 和 cost-model 的 probe 失败时分别记录缺少的
  `TEXTRACT_ND2XNZ_IMPL`/七参数 wrapper，并写入 `false` capability。只有 probe 成功且对应
  pin 同时包含 implementation 与（cost-model 所需的）latency model，才能放行。

这套 contract 也适用于未来其他 optional PTO-ISA overload；普通 op verifier 仍只校验
IR 结构和硬件共同契约，不读取最终 include 路径。

PTODSL micro-op surface 不是当前缺口。PTOAS 基线的 `ptodsl/ptodsl/pto.py` 已公开导出
`vldas`、`vldus` 和 `vsstb`，`ptodsl/ptodsl/_ops.py` 已实现三者 builder，且
`ptodsl/tests/test_jit_compile.py` 覆盖普通 `vsstb` 和 post-update 形态。个别 DSL ST 中
“`vsstb.post` 尚未暴露”的注释已经落后于当前源码，不能据此要求新增另一套 surface。

## 4. PTO IR 设计

### 4.1 ODS

建议 ODS 形态如下：

```tablegen
def TExtractNd2xNzOp : PTO_TOp<"textract.nd2xnz", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract two ND windows into two NZ destinations";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow0,
    Index:$indexCol0,
    Index:$indexRow1,
    Index:$indexCol1,
    PTODpsType:$dst0,
    PTODpsType:$dst1
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow0 `,` $indexCol0 `,`
                $indexRow1 `,` $indexCol1 `:`
                qualified(type($src)) `,` type($indexRow0) `,`
                type($indexCol0) `,` type($indexRow1) `,`
                type($indexCol1) `)`
    `outs` `(` $dst0 `,` $dst1 `:`
                 qualified(type($dst0)) `,` qualified(type($dst1)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 5, 2);
    }
  }];
}
```

全部 operand 固定存在，不需要 `AttrSizedOperandSegments`，两个 DPS init 在 operand list
中连续，符合 `PTO_DpsInitOpInterface` 的单一 `MutableOperandRange` 契约。

### 4.2 汇编示例

```mlir
%src = pto.alloc_tile
  : !pto.tile_buf<vec, 64x128xf16,
                  blayout=row_major, slayout=none_box>
%dst0 = pto.alloc_tile
  : !pto.tile_buf<vec, 32x64xf16, valid=32x64,
                  blayout=col_major, slayout=row_major>
%dst1 = pto.alloc_tile
  : !pto.tile_buf<vec, 16x32xf16, valid=13x29,
                  blayout=col_major, slayout=row_major>

pto.textract.nd2xnz
  ins(%src, %r0, %c0, %r1, %c1 :
      !pto.tile_buf<vec, 64x128xf16,
                    blayout=row_major, slayout=none_box>,
      index, index, index, index)
  outs(%dst0, %dst1 :
       !pto.tile_buf<vec, 32x64xf16, valid=32x64,
                     blayout=col_major, slayout=row_major>,
       !pto.tile_buf<vec, 16x32xf16, valid=13x29,
                     blayout=col_major, slayout=row_major>)
```

该例刻意使用不同的 destination shape，以固定“两路不是同型数组”的接口语义。

### 4.3 为什么不扩展现有 `TExtractOp`

现有 `TExtractOp` 已同时承载 base、FP、preQuantScalar、relu 和 accToVecMode 形态。把它改成
可选第二输出会产生以下问题：

- `$dst`、`$fp`、`$preQuantScalar` 已形成稳定的 ODS operand segment 和生成 accessor；
- 双输出要求第二组 index 和第二个 DPS init 同时出现，任意 optional 组合都会产生大量
  无语义状态；
- 两个 DPS init 必须是连续 `MutableOperandRange`，插入第二个 dst 会改变现有 operand
  layout 和自动生成 builder；
- 需要 custom parser/printer 才能在一个 mnemonic 下稳定区分单输出、FP、preQuant 和
  双输出格式；
- 单输出 verifier 已按 MAT/ACC/VEC 多种 loc pair 分派，继续嵌入 ND-to-2xNZ 会扩大回归面。

独立的 dotted mnemonic 与现有 `pto.tmatmul.mx`、`pto.tquant.mx` 风格一致，同时保持
`textract` 指令族关系。最终生成的 PTO-ISA C++ 名字仍然是 `TEXTRACT`，没有 ABI 分叉。

被拒绝的另一方案是新增完全无关联的 `pto.textract2`。该名字没有表达 ND-to-NZ 约束，
也不利于后续文档和 template 注册按指令族组织。

## 5. Verifier 设计

### 5.1 公共校验顺序

`TExtractNd2xNzOp::verify()` 负责以下结构和硬件共同契约中的 1-10 项，诊断中必须带
`src`、`dst0` 或 `dst1` 名称。frontend lowering 完成后，backend-boundary validation 再按
同一顺序执行 11-13 项；这些项不能被误解为依赖 planner 的 late check：

1. 对三个 tile 调用 `verifyTileBufCommon`；A2/A3 禁止 low precision，A5 允许。
2. 要求三个 operand 都是 rank-2 `!pto.tile_buf`。
3. 要求四个 index 都是 `index` type 和可折叠常量；拒绝负数和大于 `UINT16_MAX` 的值。
4. 把 raw physical/valid shape 归一化为 PTO-ISA Tile dimension，FP4 与 RowPlusOne 使用
   第 3.5、5.4 节规则。
5. 要求 `src loc=vec`、ND layout；两个 dst 都是 `loc=vec`、NZ layout，fractal size 为
   512 bits。
6. 要求 `srcElem == dst0Elem == dst1Elem`。
7. 按 target arch 校验 dtype 和 compact mode。
8. 校验 source row-stride bytes 32B 对齐、每个 dst 的 plain-NZ logical physical rows
   16 对齐、emitted physical cols c0 对齐。
9. 首版要求两个 dst 的 valid shape 静态且非零；确保归一化后的 valid extent 不超过
   `UINT16_MAX`，并检查其不大于对应 physical extent。
10. 对每个 window 独立执行 constant bounds 校验，使用 dst valid shape。
11. 在 alias/range 检查前运行 `PTOValidateNd2xNzProvenancePass`。该 pass 在
    `SerialFrontendPipeLoweringPass` 完成后、legacy/modern PlanMemory 之前执行；对 `src`、
    `dst0`、`dst1` 递归穿过 `subview`、`bitcast`、`treshape`、unrealized conversion cast 和
    其他 view-preserving cast。若路径到达 `DeclareTileOp` 或 `TAssignOp`，或包含由
    `TPopOp`/`TPopFromAicOp`/`TPopFromAivOp` 绑定的 tile handle，则把该 operand 分类为
    runtime-bound 并拒绝。`TPop` 没有 SSA result 时，按其 operand 的 declared-tile root 和
    binding user 一起追踪，不能因没有 result 就当作普通 allocation。
12. runtime-bound provenance 的固定诊断为：

    ```text
    pto.textract.nd2xnz does not support runtime-bound tile provenance for
    src|dst0|dst1; use alloc_tile with planner-owned or statically known level3 address
    ```

    诊断必须指出 operand 名称、命中的 op（例如 `pto.declare_tile`、`pto.tassign` 或
    `pto.tpop`）和定义位置。通过该 pass 的 level1/level2 operand 才进入现有 planner；level3
    还必须满足第 7.1 节的静态 address gate。`alloc_tile`、`multi_tile_get` 物化的
    `alloc_multi_tile` slot 及其合法 view chain 是首版唯一的 local tile provenance 正向来源。
13. 若 operand 可以解析到静态 byte range，拒绝三组 pair 中任意重叠；其余 alias 情况交给
   PlanMemory 语义冲突处理和规划后验证。level3 例外见第 7 节：无法证明地址为静态
   常量时直接拒绝，不把“未知”当成“不重叠”。

### 5.2 dtype helper

不要复用当前 `isA2A3ExtractElemType`，因为它没有 `i32`，而本 overload 的 A2/A3
PTO-ISA 明确支持 `int32_t`。新增窄范围 helper：

```text
isA2A3Nd2xNzElemType          = i8 | i32 | f16 | bf16 | f32
isA5LowpCandidateNd2xNzElemType = A2A3 set | hif8 | f8 variants
isA5AllCandidateNd2xNzElemType  = lowp candidate set | packed f4 variants
```

helper 只服务该 overload，不静默扩大单输出 `TExtractOp` 的合法集合。candidate 集合只用于
诊断和测试路由；verifier 的 enabled 集合由 support gate 决定。hif8/fp8 在上游 ST 中只覆盖
aligned window，因此实现 PR 还必须补至少一个 1-byte low-precision sub-c0 window 的设备
golden，才能无条件放行动态/非对齐 index。FP4 未通过第 3.5 节门槛前不能进入正向路径。

### 5.3 shape 与动态值

- physical shape 必须能生成合法的静态 PTO-ISA Tile template；动态 physical shape 直接拒绝。
- 首版要求 destination valid shape 静态。PTO-ISA 会把 `GetValidRow/GetValidCol()` 窄化为
  `uint16_t`，A5 vector path 还会直接计算 `validRow - 1`；其现有 runtime bounds assert 不拒绝
  0。仅依靠 physical upper bound 无法证明动态值非零，因此不能无保护地沿用 PTOAS dynamic
  valid 机制。后续若要放开，必须先在调用前生成 `0 < valid <= physical` 的 runtime guard，
  并增加动态 0、上界和越界测试。
- 首版要求 index 可折叠为静态常量，并在 PTO IR 阶段完成范围和 bounds 检查。PTO-ISA 的
  C++ 形参是 `uint16_t`，而 A5 TileLib/VPTO 路径会绕过 EmitC；仅在 EmitC 插 guard 会造成
  后端语义分叉。后续只有在两个 backend 共用的 runtime-check 机制落地后才开放动态 index，
  且必须在窄化前检查 `[0, UINT16_MAX]` 和 window bounds。
- `validRows` 和 `validCols` 必须大于 0；空 window 不是已定义的 no-op。
- A5 的 aligned TileLib path 还要证明 `block_stride`、`repeat_stride` 及其 RowPlusOne 变体
  均落在 16-bit hardware control field 内。不能只检查 raw physical rows 小于等于
  `UINT16_MAX`，因为 `align16(validRows) + 1` 可能跨过边界。
- static bounds 的加法使用 checked arithmetic，避免超大常量溢出后误判为合法。

首版的完整 TSTORE eligibility 由独立的 `PTOValidateNd2xNzStoreUsePass` 在
backend-boundary 检查，而不是由 `TExtractNd2xNzOp::verify()` 假定。该 pass 沿着两个 DPS
destination 的 view/use chain 到后续 `TSTORE`，要求 plain NZ 的 valid shape 等于 physical
shape；发现 partial-valid destination 时给出第 3.2.1 节的稳定诊断。它在
`PTOResolveBufferSelect` 之后、TSTORE/EmitC/VPTO 最终 lowering 之前运行，因而也能看到
multi-buffer materialization。这样 partial-valid op 本身仍可用于 UB-only 测试，但不会在
debug PTO-ISA 上晚期触发 TSTORE assertion；test-only full-valid dump alias 不属于该
destination view/use chain。

### 5.4 compact mode

- A2/A3：拒绝 `CompactMode::RowPlusOne`；`Null`/`Normal` 都按 plain NZ 处理。
- A5：header/implementation 宣称允许 `Null`/`Normal` 和 `RowPlusOne`，但第一版只有在下述
  representation adapter 与 NPU golden 同时落地后才放行 `RowPlusOne`。
- 两个 destination 的 compact mode 可以不同；A5 可以在一次调用中一边 plain NZ、另一边
  NZ+1，因为二者是独立 template parameter。

PTOAS 当前把 `tile_buf.shape` 保持为不含 gap 的 logical physical extent，用
`compact=RowPlusOne` 在 planner/sync 中把 major stride 增加 1。PTO-ISA 新 ST 则把 A5 NZ+1
destination 的 `Tile::Rows` 显式构造成 `align16(rows) + 1`，而 `TSTORE` 也从
`TileData::Rows` 取得 NZ source stride。直接把 PTOAS raw rows 原样 EmitC 会让 TEXTRACT 按
`align16(validRows)+1` 写、TSTORE 却按 raw rows 读，不能接受。

实现需要统一修正 Tile type materialization：对 ColMajor `RowPlusOne`，emitted physical
rows 表示带 gap 的 virtual rows，而 valid rows 仍是 logical rows；planner/sync/semantic range
继续按 PTOAS 的 implicit-gap footprint 计算，不能再重复加一。该转换应进入共享 emitted
dimension helper，并补全 `TEXTRACT -> TSTORE` 的混合 plain/NZ+1 设备测试。若无法在不改变
既有 RowPlusOne 用户语义的前提下完成，则第一版 verifier 必须拒绝本 op 的 RowPlusOne，
不能只检查 `CompactMode::RowPlusOne` token 已输出。

### 5.5 不增加的限制

以下输入在 PTO-ISA 有明确路径，PTOAS 不得拒绝：

- 两路 window 在 source 中互相重叠；
- 两个 destination 的 shape/valid shape 不相等；
- `indexCol` 不是 c0 或 32B 对齐；
- `validCol` 不是 c0 倍数；
- `validRows/validCols == 1`；
- A2/A3 `i8` odd validCol。

## 6. DPS、内存效应与同步

### 6.1 DPS

`getDpsInitsMutable()` 必须同时返回 `dst0` 和 `dst1`。现有以下消费者已经按 range 迭代，
设计上无需新增特判，但必须增加双输出回归：

- legacy `PTOPlanMemory`；
- `PTOPlanMemoryModern`；
- `PTONormalizeUncoveredTileSections`；
- TileFusion liveness/region generation；
- `PTOMarkLastUse`。

TileFusion 当前只把白名单中的 elementwise/reduction op 视为可融合 compute，单输出
`textract` 本身也不是白名单成员。首版将 `pto.textract.nd2xnz` 保持为 hard boundary，
不在本功能中引入 multi-output fusion 策略。

### 6.2 MemoryEffects

```cpp
void TExtractNd2xNzOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDst0Mutable(), MemoryEffects::Write::get());
  addEffect(effects, &getDst1Mutable(), MemoryEffects::Write::get());
}
```

source 不声明 Write。A2/A3 odd-i8 路径使用固定 tmp UB scratch，但不修改 source；该内部
scratch 由 PTO-ISA 保留区管理，不作为 PTO IR operand。

### 6.3 pipe 与自动同步

本 overload 的外部执行类别固定为 `PIPE_V`：

- A5 主路径完全是 vector frontend；
- A2/A3 vector 路径使用 `vcopy`/`vconv`；
- A2/A3 scalar fallback 内部自行插入 `PIPE_V <-> PIPE_S` flag/wait，公共调用的输入输出
  依赖仍以 `PIPE_V` 建模。

InsertSync/GraphSyncSolver 必须从 effects 得到一个 read 和两个 write。至少覆盖：

```text
TLOAD(src) [PIPE_MTE2]
  -> TEXTRACT ND2XNZ [PIPE_V]
    -> TSTORE(dst0) [PIPE_MTE3]
    -> TSTORE(dst1) [PIPE_MTE3]
```

full-valid 测试既要证明 MTE2-to-V 依赖存在，也要证明两个 destination 的 V-to-MTE3 消费
都被看到；partial-valid UB-only 测试只验证 TEXTRACT 的 V-side producer 和两个 destination
的 liveness，不把 partial descriptor 直接传给 NZ TSTORE。两类测试都不能只检查第一个 DPS init。

## 7. No-alias 与内存规划

`getSemanticNoAliasPairs()` 为该 op 返回：

```text
(src, dst0)
(src, dst1)
(dst0, dst1)
```

这三组 pair 同时进入：

- legacy planner 的 `RecordSemanticConflict`；
- modern planner 的 `addForbidAliasBetweenRoots`；
- `verifySemanticNoAliasRanges` 的显式/规划后 byte-range 校验。

不能只比较 SSA Value 是否相同。`subview`、`bitcast`、`treshape`、multi-buffer slot 和显式
地址可能以不同 Value 指向重叠范围，必须复用现有 semantic range 解析。

runtime-bound gate 是 no-alias 契约的前置条件，而不是 range resolver 的可选优化。当前
legacy planner 会跳过 `DeclareTileOp`，InsertSync 也只能把 declared tile 自身作为没有绝对
地址的 symbolic root；因此不能声称 planner 或 semantic range 已经证明三个 runtime-bound
tile 两两不重叠。首版对这类输入统一拒绝，避免 level1/2 静默跳过约束，也避免 level3 只看
`alloc_tile.addr` 而漏掉 declared/tpop provenance。

### 7.1 level3 显式地址规则

level1/level2 由 legacy/modern planner 产生地址并在规划后检查三组 semantic range；
level3 跳过 planner，`pto.alloc_tile.addr` 由调用方提供。当前 `SemanticRange` 对不同
allocation root 只有在双方都有 absolute address 时才比较，因此两个 allocation 使用同一
动态 `%base`（或由 `%base` 派生的同一动态地址）会被错误地视为“无法证明重叠”并放行。

首版选择保守、可执行的规则；它只接收已经通过
`PTOValidateNd2xNzProvenancePass` 的 allocation-backed operand：

- 在 driver 已解析 `effectiveLevel`、backend 和 capability environment 后运行
  `PTOValidateNd2xNzAddressPass`；对 level1/level2 不改变现有 planner 行为，但 provenance
  gate 仍然生效。
- 当 module 含 `pto.textract.nd2xnz` 且 level3 生效时，沿 `src`、`dst0`、`dst1` 各自的
  view chain traceback 到 `pto.alloc_tile`。每个 local allocation 的 `addr` 必须是可
  折叠为非负整数的静态常量；canonicalizer 能折叠的 `arith.addi`/index cast 常量表达式
  可以接受，含 block argument、函数参数或动态 `%base` 的地址一律拒绝。
- 诊断固定为：

  ```text
  pto.textract.nd2xnz requires statically known level3 addresses for
  semantic no-alias verification (src|dst0|dst1)
  ```

  并附 operand 名称及其 `alloc_tile` 定义位置。这样三种动态同址 pair
  (`src=dst0`、`src=dst1`、`dst0=dst1`) 以及 `%base + constant` 形成的同址 pair 都在
  PTOAS 阶段失败，而不是依赖 C++ 或 NPU 行为。
- 不在本 PR 中扩大通用 `SemanticRange`。后续若要支持 level3 动态地址，必须将 range
  扩展为“symbolic address root + constant offset”，解析 `arith.addi` 等保持同一 root，
  并对无法证明的不同 root 采用保守拒绝；通过 dedicated range tests 后才能删除本 gate。

因此 `DeclareTileOp`、`TAssignOp`、`TPopOp` 及其 subview 即使带有看似常量的 `addr` 也不能
绕过 gate；只有从 `AllocTileOp`/materialized multi-tile slot 回溯出的常量地址才能进入
semantic range overlap 检查。

`PTOValidateNd2xNzAddressPass` 同样放在 `PTOResolveBufferSelect` 之后。level3 的
`AllocMultiTileOp` 若在 materialization 后仍产生动态地址 select，按同一规则拒绝；只有
最终每个 destination 都能回溯到静态 non-negative address 时才进入 semantic range overlap
校验。

两个 dst 的 liveness 从同一 op 开始，planner 必须分别保留到各自最后一次消费。测试使用
不同大小和不同最后消费点，固定不能因只读取 `getDpsInits().front()` 而提前复用第二路内存。

NZ+1 footprint 已由 legacy/modern planner、sync translator 和 GraphSync 的现有
`RowPlusOne` 逻辑按 implicit gap 处理。实现不新增另一套 allocator size 公式，但必须把
第 5.4 节的 emitted virtual-row adapter 与这套 footprint 对齐，避免 shape 和 compact
同时各加一次 padding。增加 `dst0=plain`、`dst1=RowPlusOne` 的规划、EmitC 与 full-valid
TSTORE 端到端测试，证明两路分别使用自己的 stride，且相邻 allocation 不重叠；任何
partial-valid RowPlusOne case 只进入 UB-only 测试。

## 8. EmitC lowering

新增独立 conversion pattern：

```cpp
struct PTOExtractNd2xNzToEmitC
    : public OpConversionPattern<pto::TExtractNd2xNzOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      pto::TExtractNd2xNzOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value, 7> operands{
        adaptor.getDst0(), adaptor.getDst1(), adaptor.getSrc(),
        adaptor.getIndexRow0(), adaptor.getIndexCol0(),
        adaptor.getIndexRow1(), adaptor.getIndexCol1()};
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "TEXTRACT", nullptr, nullptr, operands);
    return success();
  }
};
```

具体 builder 参数按仓库当前 MLIR EmitC API 调整，但最终调用必须固定为：

```cpp
TEXTRACT(dst0, dst1, src, indexRow0, indexCol0, indexRow1, indexCol1);
```

不生成 template argument，不生成 `TEXTRACT_ND2XNZ`，也不拆成两次单输出 `TEXTRACT`。
拆分会选择普通 Vec-to-Vec path，既不能表达 ND-to-NZ layout conversion，也不能保证与
双输出 overload 的 backend dispatch 一致。

pattern 加入主 conversion pattern set。EmitC 测试分别使用 A3/A5，并把两个 dst 的
类型和四个静态 index 设为可区分值，避免只检查 `TEXTRACT(` 而漏掉参数交换。

## 9. TileLib / VPTO 设计

### 9.1 注册

新增 `lib/TileOps/a5/textract_nd2xnz.py`，并在 `lib/TileOps/__init__.py` 注册：

```text
("a5", "pto.textract.nd2xnz") -> ".a5.textract_nd2xnz"
```

template 参数顺序与 ODS operand 顺序一致：

```python
def template_textract_nd2xnz(
    src, index_row0, index_col0, index_row1, index_col1, dst0, dst1):
    ...
```

`dtypes` schema 包含一个 source、四个 `i32` scalar 和两个 destination。constraint 再校验
三 tile 的 loc/layout/dtype，以及两个 destination 各自的 compact mode。

### 9.2 A5 展开算法

当前 PTODSL 可直接使用以下公开签名，不需要先扩展 Python surface：

```python
align = pto.vldas(source)
value, align = pto.vldus(source, align)
pto.vsstb(value, destination, block_stride, repeat_stride, mask)
```

`vsstb` wrapper 会把两个 stride coercion 到 signless `i16`。template 只能传入 verifier 已证明
可编码的 bit pattern；不得依赖 Python 整数截断。普通形态不产生 updated destination，窗口
helper 应显式维护 destination offset。post-update surface 虽然存在，但只有其指针推进语义与
ND-to-NZ loop 经 VPTO 和设备测试证明一致后才可采用，不能因接口存在就默认等价。

template 复用一个 compile-time specialized window helper，两次调用分别处理 dst0/dst1，
不能假设两路 shape 相同：

1. 根据 dtype 计算 storage width、c0、vector lanes；FP4 使用 packed logical dimension。
2. 计算 source window base。
3. `1x1` 走 scalar load/store。
4. c0-aligned source base 使用 `vlds` + `vsstb`。
5. sub-c0 base 使用 `vldas` + `vldus` + exact masked store。
6. destination block stride 从该 destination 的 valid rows 和 compact mode 独立计算：
   plain 为 aligned rows，NZ+1 为 aligned rows + 1。
7. 尾列 predicate 只允许写 valid element，不能污染下一个 NZ block 或另一 destination。
8. 所有 `vsstb` control field 在构造 `i16` 前完成静态 range proof；失败时 template 不匹配，
   由 tile op verifier 给出面向 shape 的诊断。

TileLib template 的目标不是逐行复刻 PTO-ISA C++，而是保证同一 logical mapping、tail 和
NZ+1 stride。aligned、unaligned、FP4、1x1 分支都要经过 VPTO-to-LLVM intrinsic 检查和
设备 golden，未验证的 dtype 不能被宽泛 `NUMERIC_DTYPES` 提前注册。

### 9.3 A2/A3

仓库当前只有 `lib/TileOps/a5`，没有 A2/A3 TileLib template tree。本功能不借机新增整套
A2/A3 TileLib 基础设施。A2/A3 的 `pto.textract.nd2xnz` 通过 tile-level EmitC 调用
PTO-ISA，后续若引入 A2/A3 TileLib，再单独实现与原生 scalar/widen fallback 等价的模板。

## 10. Python builder 与文档接口

ODS 自动生成 Python op class 后，推荐使用：

```python
from ptoas.mlir.dialects import arith, func, pto

pto.TExtractNd2xNzOp(
    src, index_row0, index_col0, index_row1, index_col1, dst0, dst1)
```

这里的 `pto` 明确指 PTOAS ODS-generated low-level dialect binding；
`from ptodsl import pto` 是 PTODSL micro-op surface，不能用来构造该 IR op。实现 PR
增加最小 Python builder/sample smoke，证明 import、argument 顺序和文本汇编一致。

需要同步更新：

- `docs/PTO_IR_manual.md`：语义、汇编、shape/layout/dtype/compact 表；
- `docs/release/PTO-tile-Instruction-SPEC-v0.4.md`：新增双输出形态；
- `ReleaseNotes.md`：记录 `pto.textract.nd2xnz` 和架构差异；
- TileLib template 列表或生成文档（若实现 PR 修改对应索引）。

## 11. PTO-ISA pin 方案

### 11.1 当前问题

PTOAS 当前三个 pin 都早于 8 月 14 日功能提交：

| target | 当前 pin | remote |
|---|---|---|
| default CI/container/remote validation | `27386d906e8fdcbd93aec84197939bc0b2c6caea` | GitCode |
| CPU-simulator CI | `e948507e18ec4f39037a04914b97e77f5b9d75e3` | GitHub |
| CANN 9.0 dev image | `662d7f2a916d6bbde3109ce4a16ed5c28f5d900a` | GitCode |

若只实现 PTOAS lowering，生成的 C++ 会在这些 pin 上报“没有匹配的 `TEXTRACT` overload”。
即使把 GitHub pin 升到当前 latest，CPU-sim 实例化仍会因缺少
`TEXTRACT_ND2XNZ_IMPL` 失败；这是 backend 缺口，不是继续换一个更晚 pin 就能自动解决。

### 11.2 实现阶段更新规则

实现开始前先 rebase 当时的 PTOAS `main`，重新解析各 target 当前 pin，再执行：

1. GitCode target 的 candidate 必须是当前 pin 的 descendant，且不早于 `90e9d50`。
2. GitHub target 的 candidate 必须是当前 pin 的 descendant，且不早于 `29a8ead`。
3. 分别检查 candidate 中公开 API、A2/A3/A5/CPU implementation 和测试目录，不能只按
   提交标题判断。
4. GitCode NPU target 分别编译最小七参数调用。GitHub target 先检查公共 API；只有 candidate
   同时新增 CPU implementation 时，才编译并执行 CPU-sim 数值 smoke。否则把该 op 标记为
   CPU-sim unsupported，并链接对应 upstream dependency。
5. CANN 9.0 dev target 必须用实际镜像编译验证。若最新版头文件与该工具链不兼容，不能静默
   保持旧 pin 并宣称该 target 支持新 op；应在实现 PR 中明确解决兼容 pin 或标注 target gate。
6. 使用现有 `.github/scripts/update_pto_isa_pin.py` 更新，不新建第二套 updater。
7. GitCode SHA 和 GitHub SHA 是不同 commit identity，只在各自 remote 内做 ancestry 检查。

本设计核对时可用的 latest candidate 是 GitCode `bb69abb` 和 GitHub `52d4ad3`，但实现 PR
不得把本文 SHA 当成永久常量；应选择 rebase 当日经完整验证的最新 descendant。

## 12. 测试方案

### 12.1 ODS、parser 与 verifier lit

正向（FP4 与 RowPlusOne 仅在对应 support gate 已满足时启用）：

- 两路相同 shape；
- 两路不同 physical/valid shape 和不同 index；
- A2/A3 `i8/i32/f16/bf16/f32`；
- A5 support gate 已启用的 low-precision 集合；
- A5 plain + plain、plain + RowPlusOne、RowPlusOne + RowPlusOne；
- `1x1`、非 c0 index、非 c0 validCol、A2/A3 odd-i8 validCol；
- parse-print-parse 保持两个 destination 与 index 配对。

负向：

- 非 tile、非 rank-2、非 index；
- dynamic valid shape 或 dynamic index（首版）；静态 0 valid row/col；
- source/destination loc 错误；
- ND/NZ layout 或 fractal size 错误；
- 三者 dtype 不一致或架构不支持 dtype；
- source row stride 非 32B 对齐；
- destination plain-NZ logical physical rows 或 emitted physical cols 不满足 NZ；
- A5 aligned path 的 `vsstb` block/repeat stride 或 RowPlusOne virtual rows 超出 16-bit field；
- window0/window1 各自的负 index、row 越界、col 越界；
- A2/A3 任一路使用 RowPlusOne；
- 三种显式 alias pair 各一例；
- `DeclareTileOp` 作为 src、dst0、dst1 的各一例；`TAssignOp` result、`TPopOp`/
  `TPopFromAicOp`/`TPopFromAivOp` 派生值以及它们的 subview/cast 各一例；所有 level 都必须
  在 planner 前得到 runtime-bound provenance 诊断；
- level3 下任一 operand 的 `alloc_tile.addr` 含动态 root；
- 最终 codegen 时 environment unknown、capability manifest 缺失或
  `textract.nd2xnz=false`；
- support gate 未满足时使用 FP4 或 RowPlusOne；
- FP4 gate 打开后，覆盖 raw dimension 合法但 emitted dimension 非法，以及反向边界。

### 12.2 EmitC 与 C++ compile

- A3/A5 FileCheck 精确匹配七参数顺序；
- dynamic index 的 verifier 诊断，确保 EmitC 与 VPTO 不出现后端分叉；
- 两个 dst 使用不同 opaque Tile type，确保 lowering 没有误用 dst0 type；
- A5 RowPlusOne 的 destination Tile type 包含正确 `CompactMode::RowPlusOne`；
- A5 RowPlusOne 的 emitted virtual rows 与 PTO-ISA/TSTORE stride 一致，且 planner footprint
  没有重复加 padding；
- FP4 检查 doubled packed dimension；
- 生成 C++ 对 implementation PR 选定的 GitCode A3/A5 pin 做 compile-only；
- capability manifest 分别覆盖 path missing/unreadable/malformed、tuple 无匹配或多匹配、
  include root 不存在、include-tree digest/revision mismatch、NPU probe success、CPU-sim
  missing implementation、cost-model missing wrapper/latency；所有失败都应在
  `PTOValidateCodegenCapabilitiesPass` 给出稳定诊断；
- manifest 选择出的 include root 必须被后续 C++ compile harness 复用；故意替换 include root
  或 PTO-ISA revision 的 probe/compile split 必须失败；
- `--emit-pto-ir` 在 capability unknown 时仍可输出 IR，但相同输入进入 EmitC/VPTO 最终
  codegen 必须失败；
- GitHub CPU backend 未补齐时，记录 compile probe 的缺失符号和 upstream dependency；补齐后
  再启用 CPU-sim 两输出 byte-exact comparison。

### 12.3 effects、sync 与 PlanMemory

- effects 测试看到一个 Read、两个 Write；
- full-valid case 的 `TLOAD -> ND2XNZ -> 2xTSTORE` 自动同步覆盖两路；
- 两个 dst 的 consumer 位于不同 block/loop 时 liveness 都正确；
- legacy/modern planner 都为两个 live destination 分配不重叠范围；
- source/dst0/dst1 的 subview overlap 被拒绝；
- provenance pass 位于 frontend pipe lowering 之后、两个 planner 之前；declared tile、tassign、
  tpop 及其 view chain 在两个 planner 运行前均被拒绝；静态 alloc-backed 正向 case 继续进入
  对应 planner；
- level3 三组 pair 分别使用同一动态 `%base` 时拒绝，`%base + constant` 的动态派生地址也
  拒绝；三个静态非重叠常量地址通过，静态重叠地址继续由 range verifier 拒绝；
- level3 static full-valid dump alias 与 partial descriptor 共用同一 UB address 时，
  `TLOAD -> TEXTRACT -> TSTORE` 的 range/effect 顺序可验证，且不触发 TEXTRACT 三 tile no-alias
  误报；
- plain 与 RowPlusOne 混合时使用各自 footprint；
- 已有单输出 `textract` sync/plan-memory tests 全部保持不变。

### 12.4 TileLib / VPTO

- aligned f16：展开为 load + block-stride store，不残留 tile op；
- unaligned f32：出现 unaligned load path；
- public surface probe 精确生成 `vldas`、`vldus` 和普通 `vsstb`，不新增私有 builder；
- 两路不同 shape/index；
- tail validCol mask；
- `1x1`；
- A5 NZ+1 block stride；
- hif8/fp8 检查正确 vreg/intrinsic type，并至少覆盖一个 1-byte low-precision sub-c0 window；
  FP4 support gate 打开后再增加 FP4 检查；
- `vsstb` control field 最大合法值和首个非法值；
- VPTO LLVM verifier 和现有 CANN output version 组合通过。

### 12.5 NPU ST

PTOAS 新增独立 testcase，不复用 PTO-ISA ST 二进制。golden 对两个 window 分别切片并转换
ND-to-NZ，两个输出独立比较。测试按第 3.2.1 节分成两组：

- full-store group：两个 destination 都是 full-valid；plain NZ 使用 canonical physical NZ
  GlobalTensor shape，经过 `TLOAD -> TEXTRACT -> two TSTORE`，debug build 必须保持
  assertion enabled，并比较两块完整 physical GM output。A5 `RowPlusOne` 只有在 adapter
  明确定义 gap 的写回值并通过设备 golden 后才允许比较 gap；否则该 compact mode 仍停在
  support gate，不能借 full-valid 名义扩大 coverage。
- partial-valid group：使用 `prepareAndDumpPartialNzForTest` 的 full-valid dump alias 路径；
  先以 `TLOAD` 清零与 partial descriptor 同地址的完整 physical UB footprint，再执行 TEXTRACT，
  以 full-valid alias 经过 generic TSTORE 写到 physical-size GM output。不能把 partial tile
  直接传给 generic TSTORE，也不依赖 simulator UB dump/raw-UB 私有接口。golden 只比较 valid
  logical coordinates，未定义 padding 不比较，allocation/GM guard bytes 单独检查是否越界。

A2/A3 最小集合：

- f16 aligned full-valid（完整 TSTORE）；
- f16 或 i8 unaligned index；
- i8 odd validCol；
- i32；
- `1x1`；
- 两路不同 valid shape。

A5 必选最小集合：

- f16 aligned full-valid（完整 TSTORE）；
- f32 sub-c0 unaligned；
- hif8 和至少一种 fp8 的 sub-c0 unaligned byte-exact case；其余 1-byte low-precision dtype
  至少覆盖 aligned byte-exact case；
- `1x1`；
- 两路不同 valid shape。

A5 support-gate 集合：

- FP4 packed dimension：必须同时覆盖 RowMajor ND source 和 ColMajor NZ destination 的
  packed axis、row stride 与 byte-exact golden；
- plain + RowPlusOne：先用 full-valid case 经过 `TLOAD -> TEXTRACT -> two TSTORE`，证明
  virtual rows、planner footprint 和 TSTORE stride 一致；partial-valid RowPlusOne 仍走
  UB-only group。

full-store group 必须经过完整链路，partial-valid group 必须保留明确的 UB-only 标记；测试
汇总分别报告 `TEXTRACT numerical coverage` 和 `full TSTORE coverage`，不能把后者的数量
扩大到 odd-valid/`1x1` case。

### 12.6 回归门槛

- `test/lit/pto/textract_*` 全部通过；
- `test/lit/vpto/*textract*` 全部通过；
- PTOAS unit/lit 全量通过；
- A3/A5 compile-only；
- capability validation 的正负向 lit 全部通过；CPU backend 存在时执行 CPU-sim 双输出
  数值测试，缺失时 manifest 明确标记 unsupported 并链接 upstream dependency，不伪造
  simulator coverage；cost-model 同理；
- A3/A5 至少执行必选 NPU ST，并在 PR 中记录设备、PTO-ISA revision 和命令；FP4、
  RowPlusOne 只有执行对应 support-gate ST 后才可在 verifier 放行。

## 13. 实现拆分

建议按以下顺序提交，保证每一步都可单独 review：

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| 0 | rebase、逐 target pin/backend 探测、CMake manifest 生成与 driver 注入 | NPU probe 生成正向 capability；CPU/cost-model 失败生成稳定负向 capability；manifest path/root/revision 校验可执行 |
| 1 | ODS、assembly、DPS、pipe、effects | parse/print/effects 基础 lit 通过 |
| 2 | shared emitted-dimension helper、IR verifier、runtime provenance 与 backend-boundary validation | 架构矩阵、bounds、TSTORE eligibility、provenance/capability gate lit 通过 |
| 3 | no-alias、level3 address gate 与 legacy/modern planner 回归 | 三组 alias 被拒绝，declared/tpop provenance 在 planner 前失败，动态 level3 地址失败，双输出 liveness 正确 |
| 4 | EmitC pattern | A3/A5 精确文本与 pin compile-only 通过 |
| 5 | A5 TileLib/VPTO template | aligned/unaligned/tail/enabled-lowp 展开通过；NZ+1/FP4 随 gate 开启 |
| 6 | A3/A5 NPU ST；可用时 CPU-sim | 必选组合两路 byte-exact；optional gate 有真实 backend 证据 |
| 7 | manual、SPEC、ReleaseNotes | 文档与实际 verifier/EmitC 一致 |

## 14. 兼容性与完成条件

现有 `pto.textract` 的文本、ODS class、builder、accessor、verifier 和生成 C++ 不发生变化。
新 op 没有历史 IR，因此不需要 bytecode migration 或 deprecated alias。

实现合入必须同时满足：

- PTO IR 能表达两个不同 shape 的 NZ destination 和两组 index；
- verifier 的合法集合不宽于目标 PTO-ISA，且不误拒绝其 unaligned/odd/1x1 路径；
- 两个 DPS init 在 effects、sync、fusion boundary 和两套 PlanMemory 中都不丢失；
- 三 tile 两两 no-alias；
- `DeclareTileOp`、`TAssignOp`、`TPopOp`/frontend pop 绑定及其 view chain 在两个 planner 前均被
  runtime-bound provenance gate 拒绝；正向 operand 必须来自 planner-owned allocation；
- level3 中该 op 的三个 local allocation 都有可静态证明的地址；
- EmitC 只生成一次、参数顺序精确的公开 `TEXTRACT`；
- 所有宣称支持的实际编译 target，其 PTO-ISA pin 同时包含公开 overload 和对应 backend
  implementation，且 CMake probe 生成的 manifest 已通过 path/include-root/revision/digest
  校验并由 driver 注入；CPU/cost-model backend 缺失时
  不得宣称对应模拟或性能模型支持；
- A3/A5 至少各有一条 full-valid 端到端双输出数值链路；partial-valid/odd/`1x1` 只计入
  通过 `prepareAndDumpPartialNzForTest` 观测的 UB-only TEXTRACT coverage，不能直接进入
  generic NZ TSTORE；没有该可编译 helper 的设备只能计 compile-only/simulator coverage；A5
  的 FP4/NZ+1 只有通过各自 support gate 后才进入 verifier 支持集合；
- 既有单输出 `pto.textract` 行为和测试零变化。
