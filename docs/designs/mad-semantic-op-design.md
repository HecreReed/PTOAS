# `mad` 族语义化 op 设计

## 目标

把 `pto.mad*` / `pto.mad_mx*` 从“按 ISA 位域拼装”收敛成“语义自描述” op。

设计原则：

- op 直接表达计算语义
- 影响结果的因素必须可见
- 能从类型推导的，不再单独暴露
- 不能从类型推导的，必须显式成 clause
- target profile 先做闭包，不把 profile1 / reserved 字段混进来

这里讨论的是 `disa-cube.json` 对应的 target profile 语义。

## 1. 语义来源

### 1.1 从指针类型推导

`mad` / `mad_mx` 的矩阵类型应由指针元素类型推导，而不是再单独放一个 `type` 参数。

### 1.2 必须显式表达

- `unit_flag`
- `disable_gemv`
- `sat` / `nosat`
- `tf32_mode`
- `n_dir`
- `bias`

`C` 的初值语义不单独做成 clause，而是由 op 本身区分：

- `pto.mad`：zero-init
- `pto.mad_acc`：accumulate-init
- `pto.mad_bias`：bias-init

### 1.3 通过规则约束，不作为独立 operand

- `mad_mx` 的 scale 地址
- 对齐 / fractal / layout 约束
- GEMV 条件

## 2. `mad` 族完整 op 集

### 2.1 `pto.mad`

```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64
```

语义：

```text
dst = lhs * rhs
```

### 2.2 `pto.mad_acc`

```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64
```

语义：

```text
dst = dst + lhs * rhs
```

### 2.3 `pto.mad_bias`

```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>,
    !pto.ptr<..., bt>, i64, i64, i64
```

语义：

```text
dst = bias + lhs * rhs
```

### 2.4 `pto.mad_mx`

```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64
```

语义：

```text
dst = (ScaleA * lhs) * (ScaleB * rhs)
```

说明：

- `ScaleA` / `ScaleB` 不作为显式 operand
- 它们通过 `lhs` / `rhs` 的地址派生到 `L0A_MX / L0B_MX`
- `lhs` 与 `rhs` 的 MX scale 存储必须已被外部加载并与 data tile 对齐

### 2.5 `pto.mad_mx_acc`

```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64
```

语义：

```text
dst = dst + (ScaleA * lhs) * (ScaleB * rhs)
```

### 2.6 `pto.mad_mx_bias`

```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>,
    !pto.ptr<..., bt>, i64, i64, i64
```

语义：

```text
dst = bias + (ScaleA * lhs) * (ScaleB * rhs)
```

## 3. raw op 接口

semantic `mad` 族会展开成：

```text
CTRL update + raw MAD/MMAD op
```

raw op 只承载底层 MAD/MMAD 指令本身，不承载 `CTRL` 语义。

### 3.1 raw op 集合

为了保留 typed pointer 和 memory effect 信息，raw 层不直接做成全寄存器
`i64, i64, i64, i64` 形式，而是使用 typed pointer 加 packed `X_t`：

```mlir
pto.mad_raw %lhs, %rhs, %dst, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64

pto.mad_bias_raw %lhs, %rhs, %dst, %bias, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>,
    !pto.ptr<..., bt>, i64

pto.mad_mx_raw %lhs, %rhs, %dst, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64

pto.mad_mx_bias_raw %lhs, %rhs, %dst, %bias, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>,
    !pto.ptr<..., bt>, i64
```

`mad_acc` 和 `mad_mx_acc` 不需要单独 raw op；它们使用
`pto.mad_raw` / `pto.mad_mx_raw`，区别只在 `%xt` 里的 `c_init` 位。

### 3.2 raw operand 语义

- `%lhs`：底层 `[X_n]`，Matrix A，必须是 `left`
- `%rhs`：底层 `[X_m]`，Matrix B，必须是 `right`
- `%dst`：底层 `[X_d[31:0]]`，Matrix C in L0C，必须是 `acc`
- `%bias`：底层 `[X_d[63:32]]`，bias table buffer，必须是 `bias`
- `%xt`：已经 packed 的 `X_t`，类型必须是 `i64`

raw op 不再接收：

- `unit_flag(...)`
- `disable_gemv`
- `sat`
- `tf32_mode(...)`
- `n_dir`
- `m / n / k`

这些都必须在 semantic-to-raw 展开之前完成编码。

### 3.3 `%xt` packed bit 约定

`%xt` 是底层 `X_t`：

- `[11:0]`：M
- `[23:12]`：K
- `[35:24]`：N
- `[56:55]`：unit-flag control bits，合法值 `0 / 2 / 3`
- `[61]`：GEMV disable
- `[62]`：C source，`0` 表示 L0C，`1` 表示 bias table
- `[63]`：C init，`1` 表示 C 初值为 0，`0` 表示读取 C

semantic op 到 raw op 的 `%xt` 生成规则：

| semantic op | raw op | `X_t[62] / c_src` | `X_t[63] / c_init` |
|---|---|---:|---:|
| `pto.mad` | `pto.mad_raw` | `0` | `1` |
| `pto.mad_acc` | `pto.mad_raw` | `0` | `0` |
| `pto.mad_bias` | `pto.mad_bias_raw` | `1` | `0` |
| `pto.mad_mx` | `pto.mad_mx_raw` | `0` | `1` |
| `pto.mad_mx_acc` | `pto.mad_mx_raw` | `0` | `0` |
| `pto.mad_mx_bias` | `pto.mad_mx_bias_raw` | `1` | `0` |

### 3.4 raw op 不负责的内容

raw op 不配置 `SPR.CTRL`。下面这些语义必须由 semantic-to-raw 展开显式插入
`get_ctrl / sbitset0 / sbitset1 / set_ctrl`：

- `hif8` ptr element type -> `CTRL[45]`
- `tf32_mode(...)` -> `CTRL[46] / CTRL[47]`
- `sat` / `nosat` -> `CTRL[48]`
- `n_dir` -> `CTRL[51]`

其中 `hif8 / tf32_mode / n_dir` 有明确的 off/on 语义，所以 semantic-to-raw
不能只在开启时置 1，也要在关闭时置 0：

- 普通 `fp8_e4m3` -> `CTRL[45] = 0`
- `hif8` -> `CTRL[45] = 1`
- 普通 `f322f32` -> `CTRL[46] = 0`
- `tf32_mode(round_even)` -> `CTRL[46] = 1, CTRL[47] = 0`
- `tf32_mode(round_away)` -> `CTRL[46] = 1, CTRL[47] = 1`
- 不写 `n_dir` -> `CTRL[51] = 0`
- 写 `n_dir` -> `CTRL[51] = 1`

`sat` / `nosat` 当前仍是显式 flag：写 `sat` 生成饱和语义配置，写 `nosat`
生成非饱和语义配置；不写不覆盖 `CTRL[48]`。

raw op 也不负责 MX scale 地址组织；`mad_mx_raw` 仍然按 `lhs / rhs` 地址派生
`L0A_MX / L0B_MX`，并通过 verifier 约束 scale 布局。

### 3.5 raw verifier 规则

- raw op 不允许出现任何 semantic clause
- `%lhs / %rhs / %dst` 必须是 typed `!pto.ptr`
- `%lhs` 地址空间必须是 `left`
- `%rhs` 地址空间必须是 `right`
- `%dst` 地址空间必须是 `acc`
- bias raw op 的 `%bias` 地址空间必须是 `bias`
- bias raw op 的 `%bias` 元素类型必须和 `%dst` 元素类型一致
- `%xt` 必须是 `i64`
- 如果 `%xt` 是常量：
  - raw non-bias op 要求 `X_t[62] = 0`
  - raw bias op 要求 `X_t[62] = 1`
  - `X_t[56:55]` 只能是 `0 / 2 / 3`

## 4. Type 语义

### 4.1 `mad` 家族 target profile 可用类型

| Family | lhs/rhs | dst | 备注 |
|---|---|---|---|
| `s8` | `s8` | `s32` | 可由 ptr 元素类型推导 |
| `f162f32` | `f16` | `f32` | 可由 ptr 元素类型推导 |
| `bf162f32` | `bf16` | `f32` | 可由 ptr 元素类型推导 |
| `f322f32` | `f32` | `f32` | 普通 FP32 可由 ptr 元素类型推导；TF32 需要显式 `tf32_mode(...)` |
| `e4m3e4m3` | `fp8_e4m3` / `hif8` | `f32` | 普通 FP8 和 HiF8 由 ptr 元素类型区分 |
| `e4m3e5m2` | `fp8_e4m3` / `fp8_e5m2` | `f32` | 可由 ptr 元素类型推导 |
| `e5m2e4m3` | `fp8_e5m2` / `fp8_e4m3` | `f32` | 可由 ptr 元素类型推导 |
| `e5m2e5m2` | `fp8_e5m2` | `f32` | 可由 ptr 元素类型推导 |

`u8`、`s4`、`s16s8`、`f162f16`、`f16u2`、`u8s8`、`b8u2`、`MMAD_SP` 不纳入 target-profile 设计。

### 4.2 `mad_mx` 家族 target profile 可用类型

| Family | lhs/rhs | dst | 备注 |
|---|---|---|---|
| `e1m2e1m2` | `fp4_e1m2` | `f32` | 可由 ptr 元素类型推导 |
| `e1m2e2m1` | `fp4_e1m2` / `fp4_e2m1` | `f32` | 可由 ptr 元素类型推导 |
| `e2m1e1m2` | `fp4_e2m1` / `fp4_e1m2` | `f32` | 可由 ptr 元素类型推导 |
| `e2m1e2m1` | `fp4_e2m1` | `f32` | 可由 ptr 元素类型推导 |
| `e4m3e4m3` | `fp8_e4m3` | `f32` | 可由 ptr 元素类型推导 |
| `e4m3e5m2` | `fp8_e4m3` / `fp8_e5m2` | `f32` | 可由 ptr 元素类型推导 |
| `e5m2e4m3` | `fp8_e5m2` / `fp8_e4m3` | `f32` | 可由 ptr 元素类型推导 |
| `e5m2e5m2` | `fp8_e5m2` | `f32` | 可由 ptr 元素类型推导 |

## 5. Clause 语义

### 5.1 `unit_flag(...)`

这是 producer 侧的 L0C block 语义。

- 不写 `unit_flag(...)`：关闭
- `unit_flag(check_only)`：检查，不设置
- `unit_flag(check_and_set)`：检查并设置

`check_and_set` 是 `mad` 侧对应语义；consumer 侧 `acc_store` 才使用 `check_and_clear`。

### 5.2 `disable_gemv?`

- 不写：允许 GEMV
- 写：禁止 GEMV

### 5.3 `sat?` / `nosat?`

表示 CUBE 的饱和/传播语义。

- 不写：保留 target-profile 下的默认 numeric policy
- 写：显式请求 saturate 语义
- 写 `nosat`：显式请求 non-saturate 语义

### 5.4 `tf32_mode(...)`

只对不能从指针元素类型推导的执行模式出现：

- `tf32_mode(round_even | round_away)`：只对 `f322f32` 有意义

`hif8` 不放进 `tf32_mode(...)`。后续引入独立 HiF8 元素类型后，`hif8` 语义由 `lhs / rhs` 的 ptr 元素类型推导；普通 `fp8_e4m3` 仍表示普通 E4M3 解释。

其他 family 不应携带 `tf32_mode(...)`。

### 5.5 `n_dir?`

这是 `CTRL[51]` 的语义化表达，用来约束 CUBE 输出 L0C 的方向顺序。

- 不写：`CTRL[51] = 1'b0`，先 M 后 N
- 写 `n_dir`：`CTRL[51] = 1'b1`，先 N 后 M

这个 clause 主要和后续 `acc_store*` 的 layout transform / unit-flag 语义配合，不改变数学结果。

## 6. `mad_mx` 的 scale 规则

`mad_mx` 不提供 scale pointer operand。

scale 通过输入地址派生：

- `lhs` 对应 `L0A_MX`
- `rhs` 对应 `L0B_MX`
- scale 基址由 data tile 地址派生，形如 `addr / 16`

也就是说，`mad_mx` 只负责声明“我要做 MX 语义”，不负责再把 scale 地址作为独立数据流传进来。

### 6.1 约束

- scale dtype 固定为 `e8m0`
- MX-fp4 家族的 data tile 为 `K0 = 64`，对应 scale tile 为 `16 x 2`
- MX-fp8 家族的 data tile 为 `K0 = 32`，对应 scale tile 为 `16 x 2`
- 每 32 个 K 元素共享同一个 scale
- `L0A_MX / L0B_MX` 必须和 `L0A / L0B` 地址对齐
- MX-fp4 / MX-fp8 的 K0 和 fractal 布局必须满足 target-profile 约束

## 7. 设计约束

### 7.1 `mad_bias`

- `bias` 必须是 `BIAS` 地址空间
- `bias` 元素类型与 `dst` 一致

### 7.2 `mad_mx`

- 不能把 scale 当作独立 operand
- scale 必须通过派生规则和 verifier 约束表达

### 7.3 `tf32_mode`

- `f322f32` 不能只靠 ptr 类型表达
- 必须显式带 `tf32_mode(...)`

### 7.4 `hif8`

- `hif8` 由指针元素类型表达，不作为独立 clause
- `hif8` 只允许用于 `e4m3e4m3` family
- `lhs / rhs` 必须同时是普通 `fp8_e4m3` 或同时是 `hif8`；不允许一边普通 E4M3、一边 HiF8

### 7.5 `CTRL` 派生枚举

这部分是 verifier / lowering 需要固定住的关键词，不直接暴露 bit 编码：

- `unit_flag`
  - `check_only` -> `2'b10`
  - `check_and_set` -> `2'b11`
- `disable_gemv`
  - present -> `X_t[61] = 1'b1`
- `hif8` ptr element type
  - present on both `lhs / rhs` -> `CTRL[45] = 1'b1`
  - absent -> `CTRL[45] = 1'b0`
- `tf32_mode(round_even | round_away)`
  - `round_even` -> `CTRL[46]=1'b1, CTRL[47]=1'b0`
  - `round_away` -> `CTRL[46]=1'b1, CTRL[47]=1'b1`
- `sat` / `nosat`
  - `sat` -> `CTRL[48] = 1'b0`
  - `nosat` -> `CTRL[48] = 1'b1`
- `n_dir`
  - absent -> `CTRL[51] = 1'b0`
  - present -> `CTRL[51] = 1'b1`

### 7.6 `sat` / `nosat`

- `sat` 和 `nosat` 是互斥的显式语义开关
- 不写时保留 target-profile 默认行为
- 写时表示希望显式控制饱和语义，不要依赖隐式约定

### 7.7 `n_dir`

- `n_dir` 只表达输出方向
- 不改变数值含义
- 需要和 `acc_store*` 的 layout 设计一致

## 8. 推荐 verifier 规则

### 8.1 通用

- `lhs / rhs / dst` 必须是 typed `!pto.ptr`
- `m / n / k` 必须是可转成 i64 的整型值
- `unit_flag(...)` 只能是 `check_only` 或 `check_and_set`
- `disable_gemv` 只能作为 flag 出现
- `n_dir` 只能作为 flag 出现

### 8.2 `mad_bias`

- `bias` 必须是 `BIAS` 地址空间
- `bias` 元素类型和 `dst` 一致

### 8.3 `mad_mx`

- `lhs` / `rhs` 需满足 MX family 类型表
- `dst` 必须是 `f32`
- scale 派生地址必须与 data tile 地址匹配
- scale 布局和 K0 规则必须满足 MX family 约束

## 9. target profile 排除项

以下不纳入本版设计：

- `Feature Map Offset` / `fm_offset`
- `Weight Matrix Offset` / `wt_offset`
- `smask_addr`
- `sub_dtype`
- `right_shift_en`
- `MMAD_SP`
- 其他 reserved / profile1-only 字段

## 10. 最终接口形状

semantic op：

```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64

pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64

pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, !pto.ptr<..., bt>, i64, i64, i64

pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64

pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64, i64, i64

pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, !pto.ptr<..., bt>, i64, i64, i64
```

raw op：

```mlir
pto.mad_raw %lhs, %rhs, %dst, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64

pto.mad_bias_raw %lhs, %rhs, %dst, %bias, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, !pto.ptr<..., bt>, i64

pto.mad_mx_raw %lhs, %rhs, %dst, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, i64

pto.mad_mx_bias_raw %lhs, %rhs, %dst, %bias, %xt
  : !pto.ptr<..., l0a>, !pto.ptr<..., l0b>, !pto.ptr<..., l0c>, !pto.ptr<..., bt>, i64
```

这版设计的核心变化是：

- type 由指针推导
- `unit_flag` 改成 producer 语义 `check_only` / `check_and_set`，不再混入 `check_and_clear`
- `disable_gemv` 改成 flag
- 新增 raw op 层，semantic op 不再直接 lowered 到 HIVM intrinsic
- raw op 只消费 typed pointer 和 packed `%xt`
- `mad_mx` 不再把 scale 当成独立 operand
- `sat`、`tf32_mode(...)`、`n_dir` 作为显式语义 clause
- `hif8` 从指针元素类型推导，不作为独立 clause
