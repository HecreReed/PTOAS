# VPTO Soft Post-Update 优化 Pass 设计文档

## 1. 概述

本文档设计一个新的 PTOAS pass——`VPTOSoftPostUpdate`，在 **MLIR 层**（LLVM lowering 之前）将非 Post-Update 形式的 VPTO 访存操作转换为 Post-Update 形式。首期目标指令为已支持 `updated_base` 的三个 op（`pto.vlds`、`pto.vsts`、`pto.vsstb`），后续扩展至更多访存指令（见第 2 节指令全景）。pass 覆盖两种场景：`scf.for` 循环内的固定步长访存模式（循环路径）和前端已展开的连续等差访存序列（顺序路径）。

循环路径示例：

```mlir
// 变换前：偏移由归纳变量计算
scf.for %iv = %c0 to %N step %c64 iter_args(...) {
  %vec = pto.vlds %base[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
}

// 变换后：指针通过 iter_args 循环携带
scf.for %iv = %c0 to %N step %c64 iter_args(..., %ptr = %base) {
  %vec, %new_ptr = pto.vlds %ptr[%c64] : !pto.ptr<f32, ub>
      -> !pto.vreg<64xf32>, !pto.ptr<f32, ub>
  scf.yield ..., %new_ptr
}
```

## 2. Post-Update 指令全景

下表列出 bisheng `hiipu-vf-soft-postupdate` 支持的所有指令与 PTOAS 现状的交叉对比。

bisheng 内部将候选指令分为两个处理分支：

- **Auto 分支**：指令有独立的 base 和 offset 操作数。pass 通过 SCEV 分析 `base + offset` 的地址递推，构造嵌套 PHI 来模拟多层循环的 base 递增。
- **Soft 分支**：指令无独立 offset 操作数（如 `vldus`），步长只能从 base 自身的 SCEV 演化中反推。

### 2.1 已实现 `updated_base` 的指令（Mechanism A）

这些指令的 `updated_base` 是 `Optional` 结果——有则为 Post-Update 形式，无则为普通形式。

| 分支 | PTOAS Op | 非 Post intrinsic | Post intrinsic |
|------|----------|-------------------|----------------|
| Auto | `pto.vlds` | `llvm.hivm.vldsx1.v{N}{ty}` | `llvm.hivm.vldsx1.post.v{N}{ty}` |
| Auto | `pto.vsts` | `llvm.hivm.vstsx1.v{N}{ty}` | `llvm.hivm.vstsx1.post.v{N}{ty}` |
| Auto | `pto.vsstb` | `llvm.hivm.vsstb` | `llvm.hivm.vsstb.post` |

LLVM lowering 时根据 op 是否有 `updated_base` 结果来选择生成 post 或非 post intrinsic。

### 2.2 PTOAS 有 Op 但尚未实现 Post variant 的指令

这些指令结构上可支持 `updated_base`，但当前 ODS 定义中没有该可选返回值。

| 分支 | PTOAS Op | 当前 Intrinsic | 备注 |
|------|----------|---------------|------|
| Auto | `pto.vldsx2` | `llvm.hivm.vldsx2.v{N}{ty}` | `vlds` 的双向量变体 |
| Auto | `pto.vsldb` | `llvm.hivm.vsldb` | `vsstb` 的加载对称体（块步长加载） |
| Auto | `pto.plds` | `llvm.hivm.plds.b8` | predicate mask 加载（strided） |
| Auto | `pto.pldi` | `llvm.hivm.pldi.b8` | predicate mask 加载（interleaved） |
| Auto | `pto.psts` | `llvm.hivm.psts.b8` | predicate mask 存储（strided） |
| Auto | `pto.psti` | `llvm.hivm.psti.b8` | predicate mask 存储（interleaved） |
| Auto | `pto.vstas` | `llvm.hivm.vstas` | align 存储（带 offset） |
| Auto | `pto.sprsts` | `llvm.hivm.sprsts` | 标量 predicate 寄存器存储（strided） |
| Auto | `pto.sprsti` | `llvm.hivm.sprsti` | 标量 predicate 寄存器存储（interleaved） |

### 2.3 Stateful Post-Update 指令（Mechanism B：align 状态穿针）

这些指令通过显式的 align 寄存器跟踪状态，**始终返回**更新后的 align，没有非 Post 形式。

| 分支 | PTOAS Op | Intrinsic | 输出状态 | 备注 |
|------|----------|-----------|---------|------|
| Soft | `pto.vldus` | `llvm.hivm.vldus.v{N}{ty}` | `updated_align`（+ hidden base ptr） | intrinsic 返回 3 个值，PTOAS 丢弃第 3 个 |

### 2.4 bisheng 支持但 PTOAS 尚无 Op 定义的指令

| 分支 | 指令 | 说明 |
|------|------|------|
| Auto | vldix1 | 向量 interleaved 加载 x1 |
| Auto | vldix2 | 向量 interleaved 加载 x2 |
| Auto | vstai | align 存储（interleaved） |
| Soft | vldui | 非对齐 interleaved 加载 |

## 3. 上移到 PTOAS 的收益分析

毕昇编译器（bisheng）已在 LLVM IR 层实现了 `hiipu-vf-soft-postupdate` pass。将此优化上移至 MLIR 层有以下具体优势。

### 3.1 相比 bisheng LLVM IR 层实现的优势

bisheng 的 pass 工作在 LLVM IR 上，此时高级循环结构和 PTO 类型信息已被擦除，导致以下问题：

**信息丢失：**

- **`scf.for` 结构消失。** 结构化循环变成 CFG 中的 phi 节点。bisheng 必须通过 `LoopInfo` 重建循环结构，并使用 `ScalarEvolution`（SCEV）分析地址递推模式。当 SCEV 无法将地址表达式展开为仿射 `AddRecExpr` 时，Auto 分支直接放弃变换（遗漏优化），Soft 分支则用 `VecLen`（向量长度）作为步长兜底（可能产生错误代码）。在 MLIR 中，`scf.for` 直接暴露归纳变量、上下界、步长和 `iter_args`，无需任何重建。

- **PTO 指针语义丢失。** LLVM lowering 后，`!pto.ptr<f32, ub>` 变成 `ptr addrspace(6)`，元素偏移被预乘为字节偏移（`%offset * 4`）。bisheng 必须从字节算术中逆向推导元素步长。在 VPTO 层面，偏移以元素为单位，元素类型是显式的。

**已知脆弱性（按分支）：**

- **Soft 分支**（影响 `vldus` 等 Mechanism B 指令）：PHI incoming 顺序敏感，pass 会交换 header PHI 的 incoming 值作为全局副作用；回边修复存在已知的 latch-write bug（当循环不被 `LoopInfo` 识别时，静态下标写到错误的 incoming 槽）；SCEV 失败时的 `VecLen` 兜底可能静默产生错误代码。

- **Auto 分支**（影响 `vlds`/`vsts`/`vsstb` 等首期目标指令）：显式排除 AIV 软件循环，使这些模式未被优化；为多层循环构造嵌套 PHI 链，代码复杂且依赖 incoming 的固定布局；SCEV 分析失败时直接放弃变换——不会出错，但遗漏优化。

**MLIR 层面的优势：** `scf.for` 保证了良好的循环结构，`iter_args` 提供显式的 SSA 语义。不需要 SCEV 分析，不需要构造 PHI，不存在 incoming 顺序问题，AIV 软件循环也使用同样的 `scf.for` 表示，从根源上消除了上述问题。

### 3.2 能覆盖 bisheng 遗漏的场景

| 模式 | bisheng | PTOAS（本方案） |
|------|---------|----------------|
| 简单 `scf.for` + IV 偏移 | ✓（通过 SCEV） | ✓（直接模式匹配） |
| 嵌套 `scf.for` 循环 | 部分支持（脆弱的 PHI 嵌套） | ✓（递归 `iter_args` 穿针） |
| 带 mask 的 `pto.vsts` | ✓ | ✓ |
| `pto.vsstb` 块步长存储 | 部分支持（独立开关） | ✓（统一框架） |
| AIV 软件循环 | ✗（排除，因为 AIV 软件循环不被 `LoopInfo` 识别为真实硬件循环，SCEV 无法分析） | ✓（`scf.for` 统一表示，无此限制） |
| 非循环顺序访问 | ✗（不支持） | ✓（顺序路径，见 4.3） |
| 显式 `arith.addi` 步长模式 | 可能 SCEV 失败 | ✓（直接匹配 `arith.addi`） |

## 4. 算法设计

### 4.1 整体流程

```
VPTOSoftPostUpdate pass:
  单次遍历 pto.vecscope 内所有指令，对每条指令查询
  PostUpdateSet（static DenseSet<OperationName>，维护第 2 章中的指令集合）:
    若命中:
      若在 scf.for 内 → 循环路径（按优先级依次尝试）：
        1. 累加器分析 —— offset 或 base 是否为 iter_arg 且 yield 中有显式递增？
        2. 若累加器分析未命中 → delta 分析 —— stride 是否为循环不变量？
        3. 合法性检查
        4. 改写
        5. 若产生了新的 iter_arg，向外层循环传播
      若不在 scf.for 内 → 顺序路径：
        从当前指令起向后扫描，收集连续的同类型、同 base、等差常量偏移的序列，
        一次性链式改写为 post-update，迭代器跳过已处理的指令
```

### 4.2 循环路径

循环路径有两条互斥的分析路径：**累加器分析**（优先）和 **delta 分析**（兜底）。前者处理 base 或 strideOperand 已通过 `iter_args` 显式累加的场景（stride 可以是任意已计算的值），后者处理从 IV 全新计算、无累加器的场景（stride 须为循环不变量）。

两类指令（vlds/vsts 与 vsstb/vsldb）的分析和改写通过统一的地址描述符抽象，共享同一套分析流程。

#### 4.2.1 地址描述符

每个候选 op 的有效地址可表示为 `base + weight * strideOperand`：

```
struct OpAddressDescriptor {
  Value base;            // dest / source
  Value strideOperand;   // offset（vlds/vsts）/ repeat_stride（vsstb/vsldb）
  int64_t weight;        // 1（vlds/vsts）/ 32（vsstb/vsldb）
};
```

| 指令 | base | strideOperand | weight | 有效地址 |
|------|------|---------------|--------|---------|
| vlds/vsts | source/destination | offset (Index) | 1 | base + offset |
| vsstb/vsldb | destination/source | repeat_stride (I16) | 32 | dest + 32*repeat_stride |

分析和改写的核心公式基于此描述符统一表达：

```
total_delta = delta(base) + weight * delta(strideOperand)
stride_new  = total_delta / weight    // 填入 post-update 的 strideOperand 槽
init_ptr    = base_0 + weight * strideOperand_0   // 通过 pto.addptr 创建
```

约束：`total_delta` 必须整除 `weight`（weight=1 时自动满足，weight=32 时要求 `delta(base) % 32 == 0`，因为 `weight * delta(strideOperand)` 天然整除 32）。

#### 4.2.2 累加器分析（优先）

对 base 和 strideOperand 分别检查是否为 `scf.for` 的 block argument（来自 `iter_args`），且对应的 `scf.yield` 值为该 block argument 加一个增量值：

```
getIterArgIncrement(Value v, ForOp forOp) -> std::optional<Value>:
  若 v 是 forOp 的 block argument，且对应的 yield 操作数形如 arith.addi(v, s)
  （或等价的指针加法），返回 s；否则返回 nullopt。
```

```
baseIncr = getIterArgIncrement(desc.base, forOp)
soIncr   = getIterArgIncrement(desc.strideOperand, forOp)

若两者均为 nullopt → 非累加器模式，走 delta 路径
否则 → total_delta = (baseIncr or 0) + weight * (soIncr or 0)
        stride_new = total_delta / weight
```

不要求 stride 是常量或循环不变量，只要值在 IR 中已计算出来即可。

**示例（vlds，weight=1）：**

```mlir
%base = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
scf.for %iv = %c0 to %c16 step %c1
    iter_args(%off = %c0) -> index {
  %s = arith.addi %iv, %c1 : index
  %vec = pto.vlds %base[%off] : ...
  %next_off = arith.addi %off, %s : index
  scf.yield %next_off
}
```
`baseIncr` = nullopt，`soIncr` = `%s`。`total_delta = %s`，`stride_new = %s / 1 = %s`。

#### 4.2.3 delta 分析（累加器未命中时）

当 base 和 strideOperand 都不是 iter_arg 时，回退到 delta 分析。

定义 `delta(v)` 为值 `v` 在 `scf.for` 每次迭代间的变化量：

| `v` 的类型 | `delta(v)` 的值 |
|---------|-------|
| `v` 是 IV | `step`（`scf.for` 的步长） |
| `v` 是常量或循环不变量 | `0` |
| `v = arith.addi(a, b)` | `delta(a) + delta(b)` |
| `v = arith.subi(a, b)` | `delta(a) - delta(b)` |
| `v = arith.muli(a, b)`，其中一个循环不变 | `invariant * delta(other)` |
| `v = arith.index_castui(a)` 或 `arith.index_cast(a)` | `delta(a)` |
| 其他 | `unknown`（放弃） |

```
total_delta = delta(base) + weight * delta(strideOperand)
stride_new  = total_delta / weight
```

`stride_new` 须为循环不变量，`total_delta` 须整除 `weight`。

**正确性：** delta 表中的操作构成仿射函数的封闭运算集合。定义链仅由这些操作构成时，delta 计算不会遗漏。遇到表外操作时保守放弃。

#### 4.2.4 合法性检查

无论由累加器分析还是 delta 分析产出 stride，都须满足以下条件：

1. **op 尚未处于 Post-Update 形式。** 检查 `op.getUpdatedBase()` 为空。

2. **op 直接位于 `scf.for` 循环体内**（不嵌套在循环内的 `scf.if` 或其他控制流中），避免部分迭代问题。

3. **`total_delta` 整除 `weight`。** weight=1 时自动满足；weight=32（vsstb/vsldb）时要求 `delta(base) % 32 == 0`。

4. **stride_new 为零时跳过。** 地址不前进，post-update 无收益。

#### 4.2.5 改写

改写步骤对所有指令统一：

1. 计算初始指针 `init_ptr = base_0 + weight * strideOperand_0`（通过 `pto.addptr`；若值为零则直接用 `base_0`）。
2. 新增指针类型的 `iter_arg`，初始值为 `init_ptr`。
3. 创建 Post-Update op：将 `strideOperand` 替换为 `stride_new`，base 替换为 iter_arg 的 block argument。其余操作数（block_stride、mask、dist 等）不变。
4. 将 `updated_base` 通过 `scf.yield` 传出。
5. 交由 DCE 清除死代码。

**vsstb/vsldb 的硬件语义补充：**

非 Post-Update：`dest + 32*repeat_stride + blk*32*block_stride`
Post-Update：`dest_p + blk*32*block_stride`（repeat_stride 不参与存储地址），返回 `dest_p + 32*repeat_stride_p`

Post-Update 模式下 `repeat_stride` 从地址偏移变为指针前进量，因此初始指针需吸收原始偏移 `32*rs_0`。

#### 4.2.6 同一循环中的多个 Op

按 `(base, stride_new)` 分组。同组的 op 共享一个 `iter_arg`，所有 op 使用同一个 pre-update 指针（block argument），不链式传递 `updated_base`。原因：同一迭代内同组 op 访问相同地址，链式传递会使后续 op 的地址偏移一个 stride。每组只需 yield 一个 `updated_base`。

#### 4.2.7 嵌套循环

对于嵌套 `scf.for`，在每一层循环添加 `iter_arg` 携带指针，内层的 init 值接外层的当前值。处理方式：自内向外遍历。`scf.for` 的 `iter_args` 天然保证 init/yield 的对应关系。

### 4.3 顺序路径

处理前端已展开循环（或非循环场景）中连续排列的同类访存指令。

#### 4.3.1 序列检测

同样使用地址描述符。遍历过程中命中一条不在 `scf.for` 内的候选 op 时，提取其描述符，然后向后扫描同一 block 内的后续指令，收集满足以下条件的最长序列：

- 同一 op 类型
- 同一 base 操作数
- 相邻 op 的有效地址 `base_i + weight * strideOperand_i` 构成等差数列
- 等差公差整除 `weight`

```mlir
// vlds 示例（weight=1）：offset 构成等差
%v0 = pto.vlds %base[%c0] : ...     // 有效地址 = base + 0
%v1 = pto.vlds %base[%c64] : ...    // 有效地址 = base + 64
%v2 = pto.vlds %base[%c128] : ...   // 有效地址 = base + 128
// 公差 = 64，stride_new = 64 / 1 = 64
```

序列长度至少为 2 才值得变换。

#### 4.3.2 合法性检查

1. **op 尚未处于 Post-Update 形式。**
2. **公差整除 `weight`。**
3. **stride_new 为编译期常量。**
4. **序列中相邻指令之间无对同一 base 的别名写入。** 若序列中间插入了对 base 指向内存的写操作，序列在该处截断。

#### 4.3.3 改写

计算 `init_ptr = base_0 + weight * strideOperand_0`（首条 op 的有效地址），`stride_new = 公差 / weight`。将序列中的每条 op 替换为 post-update 形式，前一条的 `updated_base` 作为后一条的 base，strideOperand 替换为 `stride_new`：

```mlir
// vlds 变换后
%v0, %ptr1 = pto.vlds %init_ptr[%c64] : ... -> ..., !pto.ptr<f32, ub>
%v1, %ptr2 = pto.vlds %ptr1[%c64] : ... -> ..., !pto.ptr<f32, ub>
%v2, %ptr3 = pto.vlds %ptr2[%c64] : ... -> ..., !pto.ptr<f32, ub>
```

改写完成后，迭代器跳过已处理的指令。

## 5. Pass 集成

### 5.1 在 Pipeline 中的位置

pass 应运行在 VPTO 后端 pipeline 中，位于 `VPTOExpandWrapperOps` 之后、`PrepareVPTOLLVMLoweringPass` 之前：

```
VPTOExpandWrapperOps
CSE
→ VPTOSoftPostUpdate（新增）        ← 在此处
PTOInferVPTOVecScope
...
PrepareVPTOLLVMLoweringPass
LowerVPTOOpsPass
```

该位置确保：
- Wrapper op 已展开（IR 干净）
- Vecscope 结构完整
- Post-Update 结果对 LLVM lowering 可见，后者根据 `getUpdatedBase()` 选择 `vldsx1` 或 `vldsx1.post`

### 5.2 Pass 注册

```tablegen
// 在 include/PTO/Transforms/Passes.td 中
def VPTOSoftPostUpdate : Pass<"vpto-soft-postupdate", "ModuleOp"> {
  let summary = "Convert fixed-stride VPTO memory ops to post-update form";
  let dependentDialects = ["pto::PTODialect", "scf::SCFDialect",
                           "arith::ArithDialect"];
}
```

## 6. 实施计划

### ~~Step 1：pass 框架与循环 delta 分析~~（已完成）

1. ~~搭建 pass 框架：PostUpdateSet、遍历逻辑、pipeline 集成、CLI 开关。~~
2. ~~实现 delta 递归分析（4.2.3）。~~
3. ~~实现 delta 路径的合法性检查和改写（新增 iter_arg）。~~
4. ~~支持同一循环中多个 op。~~
5. ~~支持 `pto.vlds`、`pto.vsts`、`pto.vsstb`。~~
6. ~~添加 delta 路径的 lit 测试和反向测试。~~

### Step 2：循环累加器分析

7. 实现 `getIterArgIncrement` helper。
8. 实现累加器分析（4.2.1）：base/offset 统一检测与 stride 合并。
9. 实现累加器路径的改写（4.2.4）。
10. 在遍历逻辑中将累加器分析置于 delta 分析之前。
11. 添加累加器路径的 lit 测试（含非常量 stride、base 和 offset 都是 iter_arg 等场景）。

### Step 3：顺序路径

12. 实现顺序路径的序列检测（4.3.1）和合法性检查（4.3.2）。
13. 实现链式改写（4.3.3）和迭代器跳过逻辑。
14. 添加顺序路径的 lit 测试。

### Step 4：扩展指令覆盖

15. 为 2.2 中的指令（`vldsx2`、`vsldb`、`plds`、`pldi` 等）添加 ODS `updated_base` 定义。
16. 扩展 PostUpdateSet，使 pass 覆盖新指令。
17. 补充对应的 LLVM lowering（post intrinsic callee）和 lit 测试。

### Step 5：验证与开启

18. 端到端验证：用 ptoas 编译，与 bisheng Post-Update 输出对比已知 kernel。
19. NPU 验证：在硬件上运行 Post-Update kernel（通过现有 `test/vpto/cases/micro-op/vector-load-store/` 框架）。
20. 将默认值切换为开启。
