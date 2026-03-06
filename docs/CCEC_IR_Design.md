# PTOAS CCEC IR 设计说明（V1）

## 1. 这份文档讲什么

这份文档描述的是 PTOAS 里第一版可落地的 CCEC IR 方案。

目标不是一次性把整条 PTOAS 编译链全部改掉，而是先引入一层最小可用的、面向目标语义的 IR，让 OP-LIB 可以不用再直接写成一大坨 `scf.for + arith.*`，同时保留原来的 legacy 路线。

当前 V1 的范围很明确：

- 在 PTOAS 内新增一个 `ccec` dialect
- 先只定义一个核心算子：`ccec.vbin`
- 先只支持 6 个 binary element-wise 浮点算子：
  - `add`
  - `sub`
  - `mul`
  - `div`
  - `max`
  - `min`
- 让 OP-LIB 模板可以直接用 CCEC IR 来写
- 再把 CCEC IR 降回 `memref/scf/arith`，复用现在已经有的 loop fusion、PlanMemory、EmitPTOManual、EmitC 路线

当前 V1 明确不做的事情：

- `tload` / `tstore`
- `tcvt`
- sync 相关 CCEC op
- buffer acquire/release 相关 op
- 直接从 CCEC IR 打 EmitC
- 完整的 tile descriptor / gm descriptor 类型系统

## 2. 为什么要加 CCEC IR

现在的 OP-LIB 模板，本质上是“用通用 IR 去间接表达目标语义”：

- 先写两层循环
- 循环里写 `memref.load`
- 再写 `arith.addf` / `arith.mulf`
- 最后 `memref.store`

这样能工作，但问题也很明显：

- 库开发者写的是“怎么算”
- 不是“我想表达一个目标向量二元操作”

也就是说，模板作者真正想写的是：

- “这里是一个 tile 级别的 add”
- “这里是一个 tile 级别的 max”

而不是：

- “请你帮我在两层循环里每个点 load 两次，再做一次 scalar add”

CCEC IR 的作用，就是在中间补这一层语义：

- 对上，库开发者写模板时更直接
- 对下，PTOAS 还是可以先把它合法化回老的 loop IR，再接着走老路线

所以 CCEC IR 不是要替代整个编译器，它只是加一层“目标语义更清楚”的表示。

## 3. 为什么第一版不直接全用 vector dialect

这个问题是合理的。MLIR 自带 `vector` / `arith`，按理说应该优先复用，而不是重复造轮子。

但第一版这里先不全押在 `vector dialect` 上，原因是：

1. 现在 OP-LIB 的接口约束本身就是 buffer 风格，不是 register SSA 风格。
2. 当前 PTOAS 的 fusion 主线已经是在 `PTOViewToMemref` 之后按 memref 形态工作。
3. 我们第一版只想解决“binary element-wise OP 模板化”这一件事，不想一上来把完整 vector lowering 问题也背上。
4. 现在已经复用了大量现成 dialect：
   - 存储用 `memref`
   - 控制流用 `scf`
   - 标量算术用 `arith`
   真正新增的自定义部分只是一层很小的目标语义 op。

所以 V1 的策略不是“完全不复用”，而是：

- 周边尽量复用 MLIR 现有基础设施
- 只把“目标语义核心点”单独拉成一个最小 dialect

后面如果 CCEC 继续扩展，再回头评估哪些部分应该直接建在 `vector/arith` 扩展上，是更稳妥的路径。

## 4. 在 PTOAS 里的分层位置

V1 在 PTOAS 里的位置是这样的：

```text
PTO IR
  ->
PTOViewToMemref
  ->
OP-LIB 模板（可以是 legacy loop 形式，也可以是 CCEC 形式）
  ->
Instantiate + Inline
  ->
PTOLowerCCECToLoops
  ->
现有 low-level loop fusion / PlanMemory / EmitPTOManual / EmitC 路线
```

这个放法有两个关键点：

1. 原来的 backend 主线不需要推翻重写。
2. CCEC IR 先作为“模板层的新表达”，而不是一上来就变成新的最终 codegen 边界。

## 5. 原线路保留

这个分支有意保留了原路线。

现在仍然支持：

- 原来的 loop/arithmetic 风格 OP-LIB 模板
- 原来的 low-level loop fusion
- 原来的 `EmitPTOManual -> EmitC -> C++` 路线

新增的是：

- OP-LIB 模板可以改写成 direct CCEC form

这意味着迁移是增量的，不是替换式的：

- 老模板还能继续用
- 新模板可以试着用 CCEC
- 后面可以一类一类算子慢慢迁，不需要一次性全改

## 6. CCEC dialect 的最小定义

V1 新增一个 dialect：

```text
dialect name: ccec
namespace:    mlir::ccec
```

当前只定义一个 op：

- `ccec.vbin`

故意做小，是因为第一版的目标很窄：

- 先把 binary element-wise family 跑通

不要在第一版里同时引入太多 descriptor、buffer 生命周期、sync 语义。

## 7. `ccec.vbin` 表达的语义

`ccec.vbin` 的意思非常直接：

- 从 `src0` 按元素读
- 从 `src1` 按元素读
- 做一种二元向量语义操作
- 把结果按元素写到 `dst`

当前支持的 kind：

- `"add"`
- `"sub"`
- `"mul"`
- `"div"`
- `"max"`
- `"min"`

当前 V1 的 operand 模型：

- `src0`：rank-2 memref
- `src1`：rank-2 memref
- `dst`：rank-2 memref

当前支持的 dtype：

- `f16`
- `f32`

当前的文本格式：

```mlir
ccec.vbin kind = "add"
  ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
  outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
```

## 8. V1 的 verifier 规则

`ccec.vbin` 当前会检查：

1. `src0/src1/dst` 都必须是 memref
2. 三个操作数都必须是 rank-2
3. 三者 element type 必须一致
4. dtype 只能是 `f16` 或 `f32`
5. `kind` 只能是 `add/sub/mul/div/max/min`

这个约束是故意收紧的。第一版宁可范围小一点，也不要做成一个规则很松、后面难收敛的 IR。

## 9. OP-LIB 和 CCEC IR 的关系

CCEC IR 进入系统的第一落点，不是替代 PTO IR，而是给 OP-LIB 模板用。

也就是说，OP-LIB 的外部契约先不变：

- 还是原来的函数属性
- 还是原来的 3 个 memref 参数
- 还是 rank=2
- seed dtype 还是 `f32`

变的是模板函数体内部怎么写。

### 9.1 老写法

老写法是：

- `scf.for`
- `memref.load`
- `arith.addf/mulf/...`
- `memref.store`

### 9.2 新写法

新写法是：

- 直接一个 `ccec.vbin`

### 9.3 CCEC 模板示例

下面是一个 `tmul` 的 CCEC 模板示例：

```mlir
func.func @__pto_ccec_tmul_template(
    %src0: memref<?x?xf32, #pto.address_space<vec>>,
    %src1: memref<?x?xf32, #pto.address_space<vec>>,
    %dst: memref<?x?xf32, #pto.address_space<vec>>)
    attributes {
      pto.oplib.op = "tmul",
      pto.oplib.kind = "binary_elementwise_template",
      pto.oplib.rank = 2 : i64,
      pto.oplib.seed_dtype = "f32"
    } {
  ccec.vbin kind = "mul"
    ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
    outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
  return
}
```

### 9.4 `tmax` 示例

```mlir
func.func @__pto_ccec_tmax_template(
    %src0: memref<?x?xf32, #pto.address_space<vec>>,
    %src1: memref<?x?xf32, #pto.address_space<vec>>,
    %dst: memref<?x?xf32, #pto.address_space<vec>>)
    attributes {
      pto.oplib.op = "tmax",
      pto.oplib.kind = "binary_elementwise_template",
      pto.oplib.rank = 2 : i64,
      pto.oplib.seed_dtype = "f32"
    } {
  ccec.vbin kind = "max"
    ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
    outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
  return
}
```

## 10. 现在的 lowering 策略

V1 不是让 CCEC IR 直接去打 C++。

当前策略是：

1. 导入 OP-LIB 模板
2. 校验模板是否合法
3. 按具体 dtype 和签名实例化
4. inline 到 fused function 里
5. 用 `PTOLowerCCECToLoops` 把 `ccec.vbin` 降成普通 loop
6. 后面的 pass 继续走原路线

也就是说，V1 的本质是“先用 CCEC 写模板，再合法化回老世界”。

### 10.1 `ccec.vbin` 降低后的形状示例

例如一个 `kind = "mul"` 的 `ccec.vbin`，会被改写成这种东西：

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%rows = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
%cols = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
scf.for %i = %c0 to %rows step %c1 {
  scf.for %j = %c0 to %cols step %c1 {
    %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
    %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
    %v = arith.mulf %a, %b : f32
    memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
  }
}
```

这正好解释了为什么第一版风险比较低：

- 上层模板写法变了
- 下层 backend 还没被强行重写

## 11. 在 `ptoas` 里的 pass 位置

当前 OP fusion 主线是：

```text
LoweringSyncToPipe
-> PTOCreateFusionGroups
-> PTOViewToMemref
-> PTOCreateFusionGroups
-> PTOMaterializeFusionGroupsFromOpLib
-> PTOInstantiateAndInlineOpLib
-> PTOLowerCCECToLoops
-> Canonicalizer
-> CSE
-> PTOLowLevelLoopFusion
```

这样放的好处是：

- CCEC 模板已经 inline 完了，再统一合法化
- low-level loop fusion 看到的仍然是标准 loop nest
- 现有 codegen 不需要立刻认识 `ccec.vbin`

## 12. 当前已经实现的库范围

这一版 CCEC OP-LIB 库已经覆盖：

- `TADD`
- `TSUB`
- `TMUL`
- `TDIV`
- `TMAX`
- `TMIN`

每个模板都满足同一套 OP-LIB 契约：

- 函数属性不变
- 输入输出签名不变
- 模板体内部改成 direct `ccec.vbin`

当前库文件位置：

- `test/tile_fusion/oplib_ccec/binary_templates.mlir`

legacy 版本库仍然保留在：

- `test/tile_fusion/oplib`

## 13. 当前限制

V1 现在还很小，下面这些都还没做：

- `!ccec.tile_desc`
- `!ccec.gm_desc`
- `tcvt_mode`
- sync policy
- `ccec.tload`
- `ccec.tstore`
- 直接把 CCEC IR 发给 EmitC

所以你可以把这版理解成：

- 不是完整 CCEC 编译后端
- 只是一个先把 element-wise family 跑通的最小闭环

## 14. 为什么这个起步方式是合理的

这个起步方式的优点有三个：

1. 库开发者终于可以更直接地表达目标语义。
2. PTOAS 原有主线没有被一下子打烂。
3. 后面继续扩展时，可以按 family 一类一类往上加。

换句话说：

- 新东西足够新，真的解决了模板表达问题
- 又没有新到把整个 backend 一起拖进高风险重构

## 15. 当前这版离“最终目标”还有多远

还差得不少，但方向已经清楚了。

真正更完整的 CCEC 路线，后面至少还要继续做：

1. 增加更多目标语义 op，不只 `vbin`
2. 让 CCEC IR 在更晚的阶段才合法化，而不是这么早回退成 loops
3. 评估哪些语义应该继续保留自定义 dialect，哪些可以叠加在 `vector/arith` 扩展上
4. 逐步把 `tload/tstore/tcvt/sync` 这种更“硬件味”的语义补起来

## 16. 后续建议顺序

建议按下面顺序推进：

1. 先把 binary element-wise family 做稳
2. 增加一条“保留 CCEC IR 更久”的调试/观察路径
3. 下一步优先考虑 `tcvt`
4. `tload/tstore` 再往后放
5. 等 element-wise 这条线稳定后，再认真评估和 `vector dialect` 的关系

这样推进，风险是可控的，也比较符合现在这个代码库的现实状态。
