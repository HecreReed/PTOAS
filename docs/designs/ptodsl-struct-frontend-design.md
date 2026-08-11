<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# PTODSL Struct 前端支持设计

**状态：** 已实现，待合并

**范围：** PTODSL Python 前端、Python binding、VPTO LLVM lowering、用户手册与回归测试

**依赖：** 已存在的 `!pto.struct`、`pto.declare_struct`、`pto.struct_get` 和
`pto.struct_set` PTO IR 契约

## 1. 背景

PTO IR 已定义一个栈上异构聚合类型 `!pto.struct<...>` 及其声明、读取和写入
operation。它支持标量字段和嵌套 struct，EmitC 已能将该类型 lower 到函数局部存储；
本实现补齐 VPTO LLVM 路径的等价 lowering。

在本实现之前，PTODSL 没有对应的公开类型构造器或 operation wrapper。Python 用户
不能从 `pto` 命名空间构造 struct，也不能在 `@pto.jit` 函数中读写其字段。本设计不
依赖或改变既有 pipe pass 与 `pto.pipe` API。

本设计将现有 PTO IR 能力以最薄的、类型安全的 PTODSL 表层暴露出来。它不重新
定义 struct 的存储模型、ABI 或后端行为。

## 2. 目标与非目标

### 2.1 目标

- 在 `pto` 公开命名空间中提供 struct 类型、声明、字段读取和字段写入 API。
- 保持用户 API 与 PTO IR 的类型、常量 path 和生命周期语义一一对应。
- 支持任意层级的嵌套 struct 标量叶子访问。
- 在生成 IR 前尽早报告常见 Python 使用错误，同时保留 PTO IR verifier 作为
  最终语义约束。
- 为 EmitC 与 A5 VPTO 路径生成相同的 PTO struct IR；前端不依赖某一个后端。
- 补齐用户手册、docs-as-tests、Python 单元测试和最小 JIT 编译覆盖。

### 2.2 非目标

- 不新增或修改 `!pto.struct` 的 ODS、标量字段/path verifier、PTO pass 或 EmitC
  lowering。为在全部 PTOAS 输出路径保留 stack-storage provenance，本实现新增
  `!pto.struct` function argument 的 C++ verifier 拒绝；它不改变可表示的字段或存储
  语义。VPTO LLVM lowering 是本实现补齐的现有 IR 后端契约。
- 不将 struct 用作 Tile、TensorView、pointer、local array 或其他非标量 handle 的
  容器；这些字段本来就由 PTO IR type verifier 拒绝。
- 不新增通用 `pto.f64` 标量 type surface。PTO IR 可以在 struct 字段中表示 `f64`，
  但当前 PTODSL 未公开 `pto.f64`，因此本阶段不将 `f64` 列为可直接从 PTODSL
  author 的字段类型。
- 不提供字段名、dataclass 映射、Python dict 初始化、自动零初始化或动态字段索引。
  PTO IR 的字段身份是位置而不是名字。
- 不放宽栈对象生命周期：struct 不能从函数返回、不能作为 function argument、
  `scf.if` / `scf.for` 结果或 `yield` 值逃逸。
- 不扩展任何 PTODSL 函数 ABI 以传递 struct，包括 `@pto.jit` 的 entry 和
  `entry=False` module、`@pto.tileop` 与 `@pto.simt`。用户只能在同一 traced
  function 中声明和使用它。
- 不修改既有 pipe pass 或 `pto.pipe` 的公开 API。

## 3. 用户 API

### 3.1 类型构造

```python
state_type = pto.struct_type(pto.i32, pto.f32)
nested_type = pto.struct_type(pto.i16, pto.struct_type(pto.i32, pto.f16))
```

```python
pto.struct_type(*field_types) -> StructTypeDescriptor
```

`struct_type` 返回与 `pto.ptr(...)`、`pto.vreg_type(...)` 相同类别的惰性类型
descriptor，而不是在 Python import 时立即创建 MLIR type。其字段可为 PTODSL
标量 dtype descriptor、已经 materialize 的 MLIR scalar type，或另一个
`StructTypeDescriptor`。真正的 `!pto.struct<...>` 仅在 active MLIR context 内
解析。

第一阶段公开承诺的字段类型是现有 PTODSL scalar surface 中、同时满足 PTO IR
struct 约束的类型：`i8` / `i16` / `i32` / `i64`（含 signed、unsigned 和
signless 变体）以及 `f16`、`bf16`、`f32`。PTO IR 额外允许 `f64`，但 PTODSL
尚无 `pto.f64` descriptor；它不属于本 API 的可移植公开输入。

选择 `struct_type` 而不是仅暴露 `_pto.StructType.get(...)` 有三个原因：

- API 与其它 PTODSL type constructor 一致，并且可以安全地在 decorator 执行前
  创建；
- 嵌套类型不需要用户触及私有 `_pto` binding；
- 前端可以在调用绑定前给出稳定、面向用户的类型错误。

### 3.2 声明与字段访问

```python
@pto.jit(target="a5", mode="explicit")
def accumulate(x: pto.i32, y: pto.f32):
    state_ty = pto.struct_type(pto.i32, pto.struct_type(pto.f32, pto.i16))
    state = pto.declare_struct(state_ty)

    pto.struct_set(state, 0, x)
    pto.struct_set(state, (1, 0), y)
    count = pto.struct_get(state, [0])
    value = pto.struct_get(state, (1, 0))
    return
```

公开签名如下：

```python
pto.declare_struct(struct_type) -> Value
pto.struct_get(struct, path) -> ScalarValue
pto.struct_set(struct, path, value) -> None
```

`path` 可以是一个 Python `int`，或由 Python `int` 组成的 `tuple` / `list`。这三种
形式分别规范化为一个非空的 `DenseI64ArrayAttr`；例如 `0`、`[0]` 与 `(0,)` 完全
等价。`path` 不是运行时 scalar，因此 `pto.i32(0)`、变量、表达式或含 bool 的
序列都必须在前端报错。

直接以 PTO IR 名称公开 `declare_struct`、`struct_get` 和 `struct_set`，而不提供
`Struct.get()`、`Struct.set()` 或 `__getitem__` 语法。这样能保留 operation 与错误
的可追踪性，且不会把读和写混成 Python 容器语义。后续若有充分的易用性需求，可在
不改变这些 canonical API 的前提下增加薄别名；本设计不预先承诺该别名。

### 3.3 赋值与读取语义

- `pto.declare_struct` 只声明栈存储，不初始化字段。用户必须在读取前自行写入；
  本阶段不引入前端的 definite-assignment 分析。
- `pto.struct_set` 不产生 IR result，返回 `None`。struct value 是指向其声明 scope
  栈存储的 handle；set 是原地、带 side effect 的写入，不需要也不得 rebind 原变量。
- `pto.struct_set` 接受 Python `int` / `float` 字面量并按目标叶子字段 materialize。
  `int` 可写入整型字段，也可 materialize 为浮点字段；`float` 仅可写入浮点字段。
  写入 `f16` / `bf16` / `f32` 时，数值通过该目标 MLIR float type 的 `FloatAttr`
  materialize，因此遵循其目标精度的舍入。`bool`、字符串和其它 Python 对象不是
  本 API 的字面量输入；`float` 写入整型字段必须报错。
- 已有 SSA value 不做隐式 cast，类型必须精确匹配叶子类型。特别是本 API 不会为
  `i32 -> i16`、`index -> i32` 或 `f32 -> f16` 偷偷插入 conversion op。
- `pto.struct_get` 返回一个新的 scalar SSA value，而不是字段引用。因此先读取、
  后续再写同一字段时，已读取的 SSA value 保持原值。
- `struct_get` 和 `struct_set` 只允许 path 到达标量叶子。访问嵌套 struct 时用户
  必须给出更长的 path；不返回整块嵌套 aggregate。

## 4. 类型和生命周期契约

前端必须复用而不能绕开现有 PTO IR 契约。

| 项目 | 前端规则 | PTO IR 对应规则 |
|---|---|---|
| 字段数量 | 至少一个 | `StructType::verify` 拒绝空 struct |
| 标量字段 | 公开 PTODSL surface 支持 8/16/32/64 位整型和 `f16`、`bf16`、`f32`；IR 另支持未公开的 `f64` | 确保生成的 C++ 字段有精确可表示类型 |
| 嵌套 | 仅 `struct_type(...)` 递归嵌套 | `!pto.struct` 可作为字段 |
| 不支持字段 | `i1`、`f8e4m3`、`f8e5m2`、`hif8`、`f4*` 等 storage-only 低精度类型、tile/view、ptr、local array 和其他 handle | 与 type verifier 一致 |
| path | Python 编译期常量、非空、每级索引合法 | `DenseI64ArrayAttr`，逐层验证 |
| 叶子 | 只能读写标量叶子 | aggregate leaf 会被 op verifier 拒绝 |
| 生命周期 | 不从函数或 region 返回，不参与 yield 或 function argument | struct 是指向声明 scope 栈存储的 value |

嵌套层级没有人为上限。前端 path resolver 以迭代方式逐层遍历，复杂度与用户给出的
path 长度线性相关；VPTO storage-type construction 也使用显式后序遍历而不是 C++
递归。descriptor 的 lazy `resolve()` 继续按 descriptor nesting materialize 字段，
不参与 field-path 或 backend storage traversal。完整的 type 与生命周期合法性仍由
PTO IR verifier 最终决定。

前端不尝试在 Python 层完整复制 scope escape 分析。对函数结果、region result、
derived struct result 等跨 operation 的情形，保留现有 PTO module provenance
validation 的诊断。这样可以避免前端和 IR verifier 产生两套不一致的生命周期规则。

## 5. 前端实现

### 5.1 类型层

在 `ptodsl/ptodsl/_types.py` 中新增私有 `_StructDescriptor(_DType)` 与公开
`struct_type(*field_types)`：

1. 保存原始字段 descriptor，允许在无 MLIR context 时创建。
2. 在 `resolve()` 中按 descriptor nesting materialize 所有字段。
3. 通过 `_pto.StructType.get(resolved_fields)` 创建 type。
4. 在调用 binding 前检查 binding 是否存在；若安装的 native extension 太旧，报出
   明确的“重新构建 PTO Python extension”错误，沿用 `vreg_type` / `mask_type`
   的兼容性模式。
5. 将 `struct_type` 导出到 `_types.__all__` 及 `ptodsl/ptodsl/pto.py`。

descriptor 只保存不可变的字段声明，不缓存 resolved MLIR type，也不创建或共享
MLIR context。因此它不新增 PTODSL 的 thread-safety / context-sharing 语义；现有
trace session 与 MLIR context 的并发约束原样适用。

Python binding 的 `StructType.get` 使用 checked construction，因而会执行 PTO IR
`StructType::verify`，后者仍是底层 IR 的最终权威。PTODSL 同时保留公开 surface 的
刻意 allowlist：只接受本设计承诺的 i8/i16/i32/i64、f16/bf16/f32 及嵌套 struct，
以维持 `pto.f64` 尚未公开时的 API 边界。前端会在空字段、非 type 参数及这个公开
subset 外的字段上给出稳定错误，binding 再负责发现底层 IR 约束的其余违规。

### 5.2 Operation 层

在 `ptodsl/ptodsl/_ops.py` 中增加三个 wrapper 并导出到 `pto.py`：

| PTODSL wrapper | 使用的 binding | 关键前端职责 |
|---|---|---|
| `declare_struct(type)` | `_pto.DeclareStructOp` | 解析 type，确认结果为 `StructType`，返回包装后的 SSA value |
| `struct_get(struct, path)` | `_pto.StructGetOp` | 解包 struct value，规范化/验证静态 path，按 type 走 path 推导叶子 result type |
| `struct_set(struct, path, value)` | `_pto.StructSetOp` | 解包并验证 struct，推导叶子类型，materialize 字面量并检查 SSA value 类型 |

`StructGetOp` 的 result 是 `AnyType`，Python binding 不会替前端推导结果类型。因此
`struct_get` / `struct_set` 共用一个私有
`_resolve_struct_path(struct_type, path, *, op_name)` helper：它解析对应 `StructType`
的字段列表，逐层检查范围和 nested descent，并返回最终 scalar leaf type。`op_name`
必须进入诊断，使错误能明确指向 `pto.struct_get` 或 `pto.struct_set` 以及失败的
`path[N]`。该 helper 只做类型/path 归一化，不取代底层 verifier。

所有 wrapper 使用现有 `unwrap_surface_value()` 和 `wrap_surface_value()`，从而与
scalar 运算、trace session 和其它 `_ops.py` API 保持一致。它们不直接向用户暴露
`ptoas.mlir.dialects.pto` 的私有 `_pto` namespace。

`pto.py` 采用现有 type 和 operation 的显式 named import 方式导出四个新 symbol；
import 只能创建惰性 descriptor，不能 resolve type 或创建 MLIR context。

### 5.3 诊断

前端应在发射 operation 前给出以下稳定诊断类别：

| 场景 | 诊断要求 |
|---|---|
| `struct_type()` | 至少要求一个字段 |
| 字段不是 dtype / MLIR type | 指出字段下标和期望的类型形式 |
| 传入非 struct value | 明确要求 `pto.declare_struct(...)` 的结果 |
| path 为空、包含 bool / 非 int | 明确要求非空的静态 Python 整数 path |
| path 越界或进入 scalar | 指出失败的 path 层级 |
| path 停在 nested struct | 要求延长 path 到达 scalar leaf |
| 写入 SSA type 不匹配 | 显示目标叶子类型和实际类型 |
| `struct_type(...)` 用作 `@pto.jit` 参数注解 | 明确拒绝，提示 struct 只能在函数体内由 `declare_struct` 创建 |
| `struct_type(...)` 用作 `@pto.tileop` / `@pto.simt` 参数注解 | 明确拒绝，不把 struct 当作 runtime scalar annotation |
| struct 作为 `pto.for_(...).carry(...)` state | 明确拒绝，提示在外层声明并在 loop body 内原地读写 |
| 未在 tracing context 中创建 op | 沿用当前 MLIR insertion-point 错误约定 |

函数返回、`yield`、function argument 和其他逃逸场景继续采用 PTO IR verifier 的
既有诊断，而不在 PTODSL 中形成不完整的静态分析。例外是公开的 JIT/subkernel
signature 和 `pto.for_(...).carry(...)`：它们有明确的 PTODSL 入口，前端必须在构造
非法 ABI 或 loop-carried IR 前拒绝 struct。

## 6. 后端和兼容性

这是前端适配，不改变生成的 IR 含义：

```text
pto.struct_type(...)         -> !pto.struct<...>
pto.declare_struct(type)     -> pto.declare_struct
pto.struct_get(struct, path) -> pto.struct_get
pto.struct_set(struct, path, value) -> pto.struct_set
```

EmitC 保持既有行为，生成具名 C++ struct 的函数局部变量及成员访问。VPTO LLVM
lowering 将 `!pto.struct<T...>` 映射为 opaque LLVM pointer，并为每个 declaration
生成 literal LLVM struct 的 `alloca`；嵌套 storage type 通过显式后序遍历构造，避免
深层合法 type 消耗 C++ call stack。get/set 使用常量 field path 构造 GEP 后执行 scalar
load/store，并按 scalar type 发射 natural byte alignment。为避免 loop 或嵌套 region 内
重复栈分配，该 `alloca` 统一插入父函数的 entry block；同一 declaration 在其原始
region 中复用该 slot。该物理分配位置不放宽 PTO IR 的词法生命周期和逃逸约束。两个
VPTO emitter（默认和 CANN 9）采用相同的 lowering。无需新增 backend pass 或命令行 flag。

本实现不修改既有 pipe pass。用户声明的 struct 不引入额外的 `pto.pipe` API 或
编译器实现依赖。

新增 API 不改变既有 PTODSL 名称。选择 `struct_type` 而不是 `struct`，避免与
Python `struct` 标准库、潜在 Python class 名称以及未来结构化 value helper 混淆。

## 7. 文档和测试

### 7.1 用户手册

`ptodsl/docs/user_guide/04-type-system-and-buffer.md` 已增加“Struct”小节，明确：

- 三个 operation 签名和 `struct_type` 的字段类型规则；
- 嵌套 path 访问示例；
- 所有函数 ABI 都拒绝 struct，以及栈生命周期、无默认初始化、不能返回/yield/
  function argument 的限制；
- 该 API 用于标量状态，不用于 Tile、TensorView 或 pipe descriptor。

示例已增加至 `ptodsl/tests/support/docs_fragment_fixtures.py`，并纳入现有
docs-as-tests 机制。

### 7.2 Python 前端回归

`ptodsl/tests/test_struct.py` 覆盖：

- 标量 struct 的 `struct_type`、声明、写入、读取及期望 MLIR；
- 嵌套 struct 的多级 path 和 leaf result type；
- `int`、`float` 字面量按字段类型 materialize，以及 `bool`、`float -> integer` 和
  typed-SSA mismatch 的拒绝；
- 空字段、非法字段、非法 path、越界 path、aggregate leaf、非 struct operand 和
  写入类型不匹配的早期错误；
- 获取后再写的 SSA 读值语义；
- `struct_type(...)` 被用于 `@pto.jit` 和 subkernel 参数注解时的早期拒绝；
- 外层声明的 struct 在普通 `pto.for_` body 中原地读写仍然合法，而
  `pto.for_(...).carry(state=struct)` 被前端拒绝；
- `@pto.jit` 生成的 module 可通过 `--emit-pto-ir` frontend verification。

IR verifier lit 拒绝 struct function argument，与 PTODSL decorator 的 ABI 拒绝保持
一致。PTODSL frontend test 额外把局部 `declare_struct` 实际 lower 至 EmitC C++，防止
只验证 `--emit-pto-ir` metadata。最小 A5 VPTO LLVM lit 覆盖默认与 CANN 9 emitter，
确认 declaration、三层嵌套 path 的 get/set 都在 backend pipeline 中降为 LLVM
alloca/GEP/load/store，并覆盖 i8/bf16/i64 的自然对齐；该测试不依赖 NPU runtime。

### 7.3 验收矩阵

| 层级 | 验收内容 |
|---|---|
| Python 单测 | API 参数、type/path 推导、字面量 materialization 和错误信息 |
| Docs-as-tests | 用户示例可在 fixture context 中编译 |
| PTOAS frontend | JIT 输出在 `ptoas --emit-pto-ir` 下验证通过 |
| PTO IR verifier | struct function argument 被拒绝，避免隐藏 stack-storage provenance |
| EmitC lowering | PTODSL 局部 `declare_struct` 输出具名 struct 与直接字段访问 |
| VPTO lit | struct declaration / nested get-set 成功 lower 到 LLVM，并检查自然对齐 |

## 8. 实现组成

1. `_types.py` 实现不可变 `_StructDescriptor` 和 `struct_type`；binding 使用 checked
   `StructType` construction，并由 type 单元测试覆盖。
2. `_ops.py` 实现 path normalizer、leaf type resolver 和三个 wrapper，并导出到
   `pto.py`。
3. `_kernel_signature.py` 与 `_subkernels.py` 的 runtime-scalar annotation 判定显式
   排除 `_StructDescriptor`；所有 `@pto.jit` 与 subkernel ABI 保持 struct 专用拒绝
   诊断；control-flow carry builder 拒绝 struct state。
4. `test_struct.py` 固定公开 API、ABI boundary 和 error contract。
5. 用户手册和 docs fixture 纳入 docs-as-tests。
6. 回归验证包括 PTODSL 单测、PTODSL 到 EmitC 输出检查、针对性 PTOAS frontend
   verification，以及 struct function-argument verifier lit。
7. 两个 VPTO LLVM emitter 将 struct lowering 为 alloca/GEP/load/store，使用迭代
   post-order storage-type construction 与显式 natural store alignment；最小 A5 VPTO
   LLVM lit 覆盖默认与 CANN 9 路径、三层嵌套字段、mixed-width alignment 及 loop 内
   declaration 的 entry-block storage。

## 9. 需要保持的决定

- 公开 canonical API 是显式函数而不是容器魔术方法。
- path 是 Python 编译期常量，字段访问不支持运行时索引。
- `struct_get` 只能产生标量值，嵌套 aggregate 只能通过更长 path 访问。
- 第一阶段只支持函数内局部 state，不建立 PTODSL 的 struct 参数 ABI。
- `_StructDescriptor` 不可作为 runtime scalar annotation 或 loop-carried state。
- 嵌套层级无人为上限；path resolver 与 VPTO storage-type construction 必须迭代实现，
  并且 descriptor 不缓存跨 context 的 MLIR type。
- 后端 IR/verifier 是 lifecycle 和完整 type 合法性的最终权威。

这些决定使 PTODSL 表层成为现有 PTO IR 的直接、可验证映射，避免引入第二套 struct
对象模型或对其他编译器实现的意外依赖。
