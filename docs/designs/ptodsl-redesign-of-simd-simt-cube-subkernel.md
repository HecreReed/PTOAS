# PTODSL TileOp 单核纯 UB 设计

## 1. 目标

`@pto.tileop` 是可复用的片上计算 helper。调用者先完成 Tile 的分配和数据搬运，
再调用 tileop 对已经准备好的本地数据执行计算。tileop 本身不承担 DMA 调度，也不
承担跨 pipe 交接。

本设计的核心边界是：**一个 tileop 只在一个计算核上执行。**

- Vector tileop 只包含 SIMD 微指令，可发起已显式定义的 SIMT helper launch。
- Cube tileop 只包含 Cube 微指令。
- SIMT 归入 Vector 侧。
- 同一个 tileop 不能混用 Vector 与 Cube 微指令，即使二者位于互斥 branch 中。

这使 tileop 的 kind、section 与同步模型都保持确定：一个 helper 只有一个计算域，
调用者负责将不同计算域和 MTE 串成完整算法。

本文描述本分支实现的目标设计。TileOp 已使用 Tile/Scalar ABI、单 kind 推导和显式
SIMT launch，并拒绝 tileop 内 MTE、pipe sync 与直接 SIMT 微指令。

## 2. 基础概念和接口分工

### 2.1 Kernel、TileOp 与 SIMT helper

- **kernel** 是 `@pto.jit` 的可执行入口。它负责 Tile 生命周期、MTE 数据搬运、
  算法层的调度与跨 pipe 同步。
- **tileop helper** 是 `@pto.tileop` 定义的可复用计算函数。它只处理调用者已经
  准备好的本地 Tile，不单独分配 Tile 或发起数据搬运。
- **SIMT helper** 是 `@pto.simt` 定义的 launched-SIMT 函数。它保持独立 public
  ABI，并由用户定义资源参数与 launch dimensions。

| surface | public ABI | 执行和使用方式 |
|---|---|---|
| `@pto.tileop` | `Tile`、PTO scalar | 单核 Tile 计算；Vector 或 Cube 二选一 |
| `@pto.simt` | `ptr`、PTO scalar | 独立 SIMT helper；用户显式配置资源和 launch |

`@pto.tileop` 绝不接受 ptr 参数。若 Vector tileop 需要 SIMT，它从自身 Tile 派生
内部 ptr，调用已经定义的 `@pto.simt` helper。SIMT 微指令只写在该 simt helper 中，
不直接写在 tileop body。

### 2.2 Pipe、section 和 memory barrier

- **Vector/Cube** 是 tileop 的唯一计算域。SIMT launch 被视为 Vector 域操作。
- **section** 是 PTOAS 标示 Vector 或 Cube 代码的 IR 容器：
  `pto.section.vector` 或 `pto.section.cube`。它由 PTOAS 推导，用户不指定。
- **memory barrier**（`mem_bar`）只约束同一计算核内的内存访问顺序。它不是 MTE
  与计算核之间的同步，也不替代 `set_flag`/`wait_flag` 或 `pipe_barrier`。

## 3. Public ABI

### 3.1 `@pto.tileop`

tileop 参数只允许 `pto.Tile` 与 PTO scalar。Tile 参数可以是输入、输出或 scratch；
Tile 输出通过可写参数表达。tileop function 必须返回 `None`，不提供 SSA result。

不允许 `TensorView`、`PartitionTensorView`、`ptr`、`memref`、host tensor、
`TensorSpec`、vreg、mask 或 pipe handle 跨 tileop 边界。caller 持有全部 Tile 的
分配、别名和生命周期，因此 PTOAS 能把 helper 的 Tile 读写映射到 call operands。

### 3.2 `@pto.simt`

`@pto.simt` 保留 ptr ABI。用户必须在定义和调用处显式提供 SIMT 所需配置：

```python
@pto.simt(max_threads=256)
def simt_epilogue(dst: pto.ptr(pto.f32, "ub"), cols: pto.i32):
    # SIMT 微指令写在这里。
    ...

@pto.tileop
def vector_epilogue(dst: pto.Tile, cols: pto.i32):
    # Tileop 不接收 ptr；这里的 ptr 仅是本地派生值。
    simt_epilogue[256, 1, 1](dst.as_ptr(), cols)
```

`max_threads` 属于 `@pto.simt` 定义的资源配置；每个 work-item 的寄存器预算由
VPTO 后端按硬件资源分区自动推导。`[dim_x, dim_y, dim_z]` 属于每次 launch 的
显式线程维度。PTOAS 不从 Tile shape、循环或相邻指令猜测 launch dimensions，也不把
`store_vfsimt_info` 作为 tileop 内部 SIMT outline 的隐式协议。

## 4. TileOp body 规则

### 4.1 Vector tileop

Vector tileop 允许：

- SIMD 微指令；
- 对已定义 `@pto.simt` helper 的显式 launch；
- 标量计算、地址派生、cast、`arith`、`scf` 与 `mem_bar`。

### 4.2 Cube tileop

Cube tileop 允许：

- Cube 微指令；
- 标量计算、地址派生、cast、`arith`、`scf` 与 `mem_bar`。

### 4.3 两类 tileop 共同禁止的内容

- 所有 MTE 指令，包括 GM/UB、L1/L0、L0C/UB 等数据移动；
- `pipe_barrier`、`set_flag`、`wait_flag`、buffer acquire/release 等 pipe 同步；
- 直接出现的 SIMT 微指令。它们必须位于 `@pto.simt` helper；
- 高层 TileOps、Tile 分配、`alloc_tile`、`reserve_buffer`、`TAlloc`；
- 调用另一个 tileop helper；
- Vector 与 Cube 微指令混用。

“纯 UB”在这里指 tileop 不自行进行 MTE 数据搬运：调用者必须在进入 tileop 前把所需
本地 buffer 准备好，在 tileop 返回后再按需要搬运或复用。Cube 计算需要的本地存储
层级同样由调用者预先安排；tileop 只发出计算微指令，不发出 MTE 指令。

`mem_bar` 是唯一允许直接写在 tileop 内的同步类操作。它只能用于本核访问顺序，
不创建跨 pipe 事件，也不改变 tileop 的 Vector/Cube kind。

## 5. 调用者调度和同步

完整算法的 MTE 与跨 pipe 交接留在 kernel orchestration 中：

```text
caller kernel:
  MTE load / local-buffer preparation
  -> Vector or Cube tileop
  -> MTE store / next-stage preparation
```

PTOAS 的 `InsertSync` 对普通 Tile data hazard 自动建模：它分析 caller 中的 MTE
操作和 tileop call，对不同 pipe 的读写依赖插入所需同步。算法规定的 event id、
流水重叠、刻意别名和其他非普通 data hazard，仍由 kernel 在 tileop 外显式表达。

tileop 内没有 MTE 或 pipe sync，因此不再需要多 phase helper summary。其摘要只记录：

- 唯一计算 kind：Vector 或 Cube；
- 各 Tile 参数的 read/write effect；
- scalar 参数的类型信息。

PTOAS 将参数 effect 按函数参数顺序记录在 `pto.tileop.effects` 数组中，元素为
`none`、`read`、`write` 或 `readwrite`。Tile 派生的地址、subview 和
`pto.simt_launch` 参数会回溯到原 Tile 参数；无法解析的 memory effect 保守合并到
全部 Tile 参数。旧 IR 若没有摘要，`InsertSync` 继续按 readwrite 保守处理。

当 helper 读取动态 Tile 的 `valid_shape` 时，materialize pass 在验证用户编写的
Tile/scalar ABI 后，为对应 helper 参数追加内部 row/col `index` 参数，并同步扩展所有
`func.call` 调用链。这样 ViewToMemref 将 Tile 参数转换为 memref 后，动态 metadata
仍由独立 SSA 值保存。内部参数在 `pto.tileop.effects` 中记为 `none`，不改变用户可见
的 `@pto.tileop` 签名，也不引入额外 memory dependency。

每个 tileop call 在 `InsertSync` 中只形成一个计算节点：Vector tileop，包括其中的
SIMT launch，使用 Vector pipe；Cube tileop 使用 Cube pipe。`mem_bar` 不产生额外的
跨 pipe 同步节点。

## 6. PTODSL tracing 和 SIMT launch

PTODSL tracing 执行 Python 定义并记录 PTO IR。新设计中：

1. `@pto.tileop` tracing 验证 Tile/Scalar ABI，生成带 `pto.tileop.helper` marker 的
   `func.func`，并原样记录允许的计算和结构性操作。
2. `@pto.simt` tracing 生成独立的 `pto.simt_entry`，保留用户指定的
   `max_threads` 与 ptr/scalar 函数签名；VPTO 根据 `max_threads` 自动推导
   每个 work-item 的寄存器预算。
3. tileop 对 simt helper 的 `helper[dim_x, dim_y, dim_z](...)` 调用生成
   `pto.simt_launch`。launch 的 callee、资源参数和 dimensions 都来自用户源码。

tileop tracing 不预套 section，不填写 `primary_domain`，不构造多 phase graph，也不
自动 outline 一段裸 SIMT 微指令。调用 `@pto.simt` helper 是使用 SIMT 的唯一合法
tileop surface。

## 7. Section 推导和物化

### 7.1 Kind 推导算法

PTOAS 在 `PTOMaterializeTileOpSectionsPass` 之前运行 tileop kind inference。输入是
已经通过 ABI 与 body contract 验证的 helper body。pass 递归遍历 entry block 和所有
`scf` region，并按下表分类：

| 操作类别 | 分类结果 |
|---|---|
| SIMD 微指令 | Vector evidence |
| `pto.simt_launch` | Vector evidence |
| Cube 微指令 | Cube evidence |
| `arith`、地址派生、cast、`scf`、`mem_bar` | neutral，递归分析其 region |
| MTE、pipe sync、直接 SIMT 微指令、高层 TileOp | contract error |

遍历结果遵循固定规则：

1. 只发现 Vector evidence，helper kind 为 Vector。
2. 只发现 Cube evidence，helper kind 为 Cube。
3. 同时发现二者，报错并标出首个 Vector 与 Cube 操作。不同 branch、不同 loop 或
   不同 source span 也不能规避该规则。
4. 没有任何计算 evidence，只含 scalar、control-flow 或 `mem_bar` 时，报错。

kind inference 不再产生 `pto.tileop.primary_domain`。它产生单一 kind summary 和
Tile effect summary，供 section materialization、`InsertSync` 和 verifier 使用。

### 7.2 Section IR 边界

`pto.section.vector/cube` 保持原有的无 operands、无 results 单 block 容器。helper
的 Tile/scalar function 参数按 MLIR region 的原有规则被 section body 隐式捕获。
tileop 的计算结果通过可写 Tile 参数传递，section 内的 SSA 值不能作为 function
result 逃逸：

```text
pto.section.vector {
  // %tile0 and %n are function arguments captured from the enclosing scope.
  ... Vector micro-ops and scalar control writing mutable Tiles ...
}
func.return
```

- section ODS、parser/printer 与 Python constructor 均保持原有接口；
- section 没有 operands/results，也不使用 `pto.section.yield`；
- helper 的 Tile/scalar 参数通过 region 隐式 capture 使用；
- vreg、mask、ptr 与 pipe handle 不能从 helper 返回；
- `func.return` 保持在 section 外，并且必须无 operands。

### 7.3 物化算法

kind inference 成功后，materialize pass 对每个 tileop helper 创建**一个**对应 kind
的 section：

1. 选择 helper entry block 中除 `func.return` 外的全部操作。结构化控制流、标量
   操作、`mem_bar` 和 Vector tileop 中的 `pto.simt_launch` 随计算操作一起进入 section。
2. 创建原有的无 operands/results section，并将操作移动到 section body；函数参数
   继续通过 region 隐式 capture 使用。
3. 原 helper 的无值 `func.return` 保持在 section 外。
4. 不允许 section 嵌套。inline 后若形成同 kind 的嵌套 section，section normalization
   合并其 body；任何跨 kind 嵌套直接报错。

因为 helper 已经是单核、无 MTE 和无 pipe sync，section 不需要按 phase 切分，也不
需要 MTE owner 推导。一个 helper 恰好产生一个 Vector 或 Cube section。

## 8. `VPTOSplitCVModule` 和 `kernel_kind`

单个 tileop 只有一个 kind，但 kernel inline 多个 helper 后仍可包含多个 Vector 与
Cube section。`VPTOSplitCVModule` 必须支持这种情况：

- Vector clone 保留全部 Vector section，以及这些 section 调用的 `pto.simt_entry`；
- Cube clone 保留全部 Cube section；
- 同 kind section 在 clone 中展开为 body，另一 kind section 及其无引用 simt entry
  被删除；
- 不再限制每个函数只能有一个同 kind section；
- 显式指定单一 `pto.kernel_kind` 的 module 若包含另一 kind section，必须诊断，
  不得静默删除用户代码。

`pto.kernel_kind` 是 PTOAS 为 split 与 VPTO lowering 赋予目标 module 的属性，不是
`@pto.tileop` 的用户参数，也不能取代 helper kind inference。

section 不产生 scalar result。跨核数据交接应通过 caller 拥有的 Tile、MTE 和同步完成。

## 9. Verifier 诊断

tileop contract verifier 至少应拒绝：

- 非 Tile/Scalar tileop 参数 ABI，或任何 function result；
- 直接 ptr ABI、TensorView ABI、helper-local Tile 分配或高层 TileOps；
- MTE 指令、`pipe_barrier`、flag 同步或 buffer 同步；
- 直接 SIMT 微指令，以及未由 `@pto.simt` 定义产生的 SIMT launch；
- Vector 与 Cube evidence 混用；
- 没有任何 Vector/Cube evidence 的 helper。

诊断必须指出 helper、违规操作和推导出的冲突 kind，而不是推迟到 VPTO 或 LLVM
lowering 才报告“非法 section”。

## 10. 主干基线与实施路径

本设计从最新 `upstream/main` 开始实现，而不是在此前的多 phase tileop 原型上继续
修改。当前主干的相关基线如下：

- PTODSL 公开的 subkernel surface 是 legacy `@pto.simd`、`@pto.cube` 和独立的
  `@pto.simt`；尚不存在 `@pto.tileop`。
- `@pto.simt` 已具备 ptr/scalar ABI、资源参数和显式 launch dimensions。这部分应
  保留，并成为 Vector tileop 发起 SIMT launch 的基础。
- PTOAS 尚无 tileop marker、kind inference、Tile effect summary、tileop contract
  verifier 或 tileop section materialization pass。
- 现有 `pto.section.vector/cube` 是无 operands/results 的宏容器，
  `VPTOSplitCVModule` 仍限制每函数每 kind 最多一个 section。

实施按以下顺序进行：

1. 在 PTODSL 增加 `@pto.tileop`，仅接受 Tile/Scalar ABI；保留 `@pto.simt` 的 ptr
   ABI、资源参数和 launch syntax。
2. 让 tileop 可以调用已定义的 simt helper，同时在前端和后端拒绝 tileop 内 MTE、
   pipe sync、直接 SIMT 微指令和 C/V 混用。
3. 新增单 kind inference 与 Tile effect summary；它取代 legacy simd/cube 的角色
   推断，不引入 `primary_domain` 或多 phase 属性。
4. 复用原有无 operands/results 的 section ODS、parser/printer、EmitC 与 rewrite，
   实现每 helper 一个 section 的物化，不扩展 section ABI。
5. 让 `InsertSync` 把 tileop call 建模为一个 Vector 或 Cube 节点，并与 caller 中的
   MTE 操作建立普通 data-hazard 同步。
6. 改造 `VPTOSplitCVModule`，支持 inline 后的多个同 kind section 和 Vector 侧的
   SIMT entry。
7. 补齐前端、lit、VPTO 和 LLVM lowering 回归，并退役 legacy
   `@pto.simd`/`@pto.cube` decorator 及 `with pto.simd():`/`with pto.cube():`
   context manager：使用时立即诊断并引导迁移到 `@pto.tileop` 或
   `with pto.tileop():`。inline TileOp 捕获值遵守与 named TileOp 相同的
   Tile/Scalar、无返回值 ABI；MTE、pipe sync 和 orchestration 留在 caller。

## 11. 设计验收与回归测试

目标实现至少应覆盖：

- Tile/Scalar tileop 参数 ABI 正例，以及 TensorView、ptr、memref、vreg、mask 和任何
  function result 的负例；
- SIMD-only、Cube-only、SIMT-launch-only Vector tileop 的 kind 推导；
- 同一 helper 内 C/V 混用，包括 branch/loop 中混用的负例；
- tileop 内 MTE、pipe sync、直接 SIMT 微指令的负例，和 `mem_bar` 正例；
- `@pto.simt(max_threads)` 加显式 launch dimensions 的正例，以及缺少合法
  simt 配置的负例；
- helper 参数在原有 section 中的隐式 capture，以及任意 function result 的负例；
- caller MTE 与单节点 tileop 的 `InsertSync` 依赖；
- 多 helper inline 后的多 Vector/Cube section split、SIMT entry 仅保留在 Vector
  clone；
- 最终 VPTO 与 LLVM lowering 的端到端编译回归。
