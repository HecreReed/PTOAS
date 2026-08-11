# VPTO 寄存器驻留场景下的 `VF_SIMT` code size 修复方案

## 1. 背景与结论

PTOAS 使用 `pto.keep`/`pto.resume` 在连续 SIMT VF 调用之间保留寄存器值。
当前 LLVM lowering 通过空 inline asm 和 fixed TPER constraint 表达固定物理
寄存器的 def/use。BiSheng 无法正确统计包含该 inline asm 的 SIMT function
大小，最终把 `VF_SIMT` 的 code size 写成 `0xffff`。

该错误会把远大于实际 SIMT function 的地址范围声明为有效 text。当两个连续
运行的 kernel 分别位于不同 `.so` 时，前一个 kernel 可能使 IFU 提前读取后一个
kernel 尚未加载的地址。后一个 kernel 加载后，IFU 仍可能使用此前缓存的全零
内容，最终触发 vector core exception。

修复分为三层：

| 层级 | 方案 | 定位 |
| --- | --- | --- |
| BiSheng 方案 A | 空 inline asm 按 0 字节计数，并拒绝无效 size | 优先的根因修复，改动最小 |
| BiSheng 方案 B | 支持 TPER named-register intrinsic | 长期方案 |
| PTOAS 临时方案 | 根据最终 ELF symbol size 修补 device object | 用于等待新 BiSheng 发布期间的兼容处理 |

PTOAS 临时方案必须满足以下原则：

- 只修补能够由 LLVM call 关系、ELF symbol 和目标指令三方唯一确认的 callsite；
- 已正确编码的 object 保持不变；
- 任一校验失败时终止编译，不生成部分修补的 object；
- 新 BiSheng 完成根因修复并通过回归后，删除 object patcher。

这些方案只改变 LLVM lowering、code size 计算或编译产物处理，不改变
`pto.keep`/`pto.resume` 的 PTO IR 语义和逻辑 slot 分配。

本文不展开 BiSheng 源码、目标汇编、机器码或指令字段布局。目标相关实现以
内部 ABI/ISA 定义和匹配版本的工具链为依据，不构成 PTOAS 的公开接口。

## 2. 问题分析

### 2.1 触发条件

`pto.keep`/`pto.resume` 用于在同一个 scalar kernel 发起的连续 SIMT VF 调用
之间保留按线程划分的寄存器值。当前 slot 与物理寄存器的映射为：

```text
slot 0 -> TPER4
slot 1 -> TPER5
...
64-bit slot 2 -> TPERL3 -> TPER6/TPER7
```

为了让 LLVM 看到固定物理寄存器的 def/use，lowering 使用空 inline asm 和
fixed output constraint。两个 `i32` 值的 keep/resume 形式如下：

```llvm
%keep = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5},0,1"(i32 %a, i32 %b)

%resume = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5}"()
```

asm 模板不生成设备指令，但 LLVM Machine IR 中仍包含 inline-asm 节点。
问题由 BiSheng 的 function-size 统计触发，与 SIMT body 中的普通指令无关。

### 2.2 错误传播

BiSheng 中的错误传播过程如下：

1. SIMT function 的 machine basic block 中出现空 inline asm；
2. BiSheng 无法计算该 block 的大小，返回失败值 `-1`；
3. `-1` 继续作为整个 SIMT function 的大小向后传递；
4. 更新 `VF_SIMT` 时，失败值未经检查便转换为无符号值
   `4294967295`（`0xffffffff`）；
5. 最终 object 中的 `VF_SIMT` code size 变为 `65535`（`0xffff`）。

`-1`、`0xffffffff` 和 `0xffff` 分别是内部失败值、错误的无符号转换结果和
最终产物中的异常字段，都不表示 SIMT function 的真实大小。

根因有两部分：

- 空 inline asm 本身不占字节，却使 function-size 统计失败；
- 失败值没有被拦截，反而被写成了可编码的 code size。

只要 `simt_entry` 中含有 BiSheng 无法确定长度的 inline asm，就可能触发同类
问题，并不限于 fixed TPER constraint。

### 2.3 正确值来源

PTOAS 不从 PTO IR、LLVM IR 或汇编文本估算 code size。修复 `0xffff` 时，
PTOAS 在 BiSheng 完成代码生成后读取目标 `simt_entry` 的 ELF `st_size`，再按
目标 ABI 规定的单位换算，作为覆盖整个 symbol 的安全替代值。

BiSheng 正常生成的有限 code size 不要求与 `st_size` 换算结果严格相等。
`st_size` 可能包含不属于 `VF_SIMT` 取指范围的函数结束或对齐指令，因此正常
编码可以小于 symbol size。patcher 对这类值只检查其非零且不超过 symbol
边界，并保持原值不变。

`st_size` 必须满足以下条件：

- 非零；
- 符合目标指令大小和对齐要求；
- symbol 完整位于 executable section 内；
- 换算结果能够由 `VF_SIMT` code size 字段表达。

不能用下一个 symbol 地址或 section 尾地址推算大小，因为函数之间可能包含
对齐 padding。

### 2.4 d5120 产物证据

d5120 scalar caller 包含两个 SIMT callsite，原始 object 中的 code size 都是
`0xffff`。未 strip object 的 symbol 大小和正确值如下：

| SIMT function | ELF `st_size` | 修补 code size | 原始值 |
| --- | ---: | ---: | ---: |
| `rmsnorm_d5120_kernel_simt_0_simt_entry` | 232 bytes | 58 | 65535 (`0xffff`) |
| `rmsnorm_d5120_kernel_simt_1_simt_entry` | 2320 bytes | 580 | 65535 (`0xffff`) |

A/B 实验只把这两个 callsite 的 code size 改为 58 和 580。修补后的
`.aicore_binary` 仅这两处字段发生变化，SIMT body、scalar caller 的其他字段、
PB 参数、dynamic UB、host launcher 和 launch 顺序均保持不变。

### 2.5 运行影响

当 `kernel1` 和 `kernel2` 分别位于 `libkernel1.so` 和 `libkernel2.so` 时，
问题按以下顺序发生：

1. runtime 加载 `libkernel1.so` 并执行 `kernel1`；
2. `kernel1` 依次调用 `simt_vf0` 和 `simt_vf1`，两个 `VF_SIMT` callsite 的
   code size 都是 `0xffff`；
3. 过大的有效范围越过 `libkernel1.so` 的实际 SIMT text，并覆盖后续
   `libkernel2.so` 将使用的 `simt_vf2` 和 `simt_vf3` 地址；
4. `libkernel2.so` 尚未加载时，IFU 根据该范围提前取指，读取到的内容为全零；
5. runtime 随后加载 `libkernel2.so`，但没有使此前建立的 IFU 状态重新取指；
6. `kernel2` 启动 `simt_vf2` 时执行了缓存的全零内容，而不是新加载的 binary。

模型中该场景表现为持续执行由全零内容解码出的错误指令并卡死；真机上表现为
vector core exception，host 通常在同步时收到 `507035`。根据错误指令的执行
路径，也可能报告 GM address 或 UB access 越界。这些都是错误取指的下游症状，
不是 RMSNorm 数据地址或 dynamic UB 配置错误。

异常出现在 `kernel2`，但污染来自先执行的 `kernel1`。d5120 -> d7168 场景中，
d5120 的两个真实 code size 只有 58 和 580，却都声明为 65535，导致 d7168 的
SIMT text 落入前序 program 声明的范围。

以下 A/B 结果进一步确认了这一点：

- 单独运行每个 kernel 可以通过；
- 多个 kernel 放在同一个 fatbin 中可以通过；
- kernel 分别位于不同 `.so` 时，连续加载和执行会失败；
- 只修正前序 d5120 的两个 code size 后，d5120 -> d7168 连续运行通过。

因此，`0xffff` 是运行失败的直接原因。错误范围究竟通过预取、I-cache/tag
还是 fetch mapping 影响后续 program，属于 runtime、firmware 和 IFU 的实现
细节，不影响编译器侧的根因和修复结论。

## 3. BiSheng 根因修复

### 3.1 方案 A：正确处理空 inline asm

BiSheng 的 function-size 统计需要满足以下要求：

- inline asm 模板严格为空时，该节点按 0 字节处理，继续统计 block 中的其他
  指令；
- 模板非空且大小未知时，保持失败状态，不估算其长度；
- function size 为负数、未知值或超出字段范围时，停止更新 `VF_SIMT` 并给出
  明确诊断；
- 禁止把失败值转换为无符号数后写入 object。

实现上只需在现有 inline-asm 分支中区分空模板和非空模板，并在写入
`VF_SIMT` 前统一检查 size。判断应使用严格的空字符串语义，不能把只含空格、
注释或 directive 的非空模板当作零字节。

方案 A 不改变 PTOAS IR、fixed TPER constraint、寄存器分配或 SIMT body，
改动范围最小，是当前优先采用的 BiSheng 修复。

### 3.2 方案 B：使用 named-register intrinsic

长期方案是用 LLVM `llvm.write_register`/`llvm.read_register` intrinsic
替代 fixed TPER inline asm。PTOAS 仍沿用现有 slot 和 payload pack/unpack
规则，只改变 LLVM 表达形式：

```llvm
declare void @llvm.write_register.i32(metadata, i32)
declare i32 @llvm.read_register.i32(metadata)
declare void @llvm.write_register.i64(metadata, i64)
declare i64 @llvm.read_register.i64(metadata)

!100 = !{!"TPER4"}
!101 = !{!"TPER5"}
!102 = !{!"TPERL3"}

call void @llvm.write_register.i32(metadata !100, i32 %a)
call void @llvm.write_register.i32(metadata !101, i32 %b)
%a.restored = call i32 @llvm.read_register.i32(metadata !100)
%wide = call i64 @llvm.read_register.i64(metadata !102)
```

寄存器映射保持不变：

```text
32-bit: physical register = TPER(4 + slot)
64-bit: base register = 4 + slot
        name = TPERL(base register / 2)
```

该方案要求 BiSheng：

- 支持 `TPER4...TPER126` 的 `i32` named-register 访问；
- 支持 `TPERL2...TPERL62` 的 `i64` named-register 访问；
- 正确描述 64-bit register pair 与 32-bit sub-register 的 alias 关系；
- 保留 intrinsic 引用的 TPER 及其 alias，避免普通寄存器分配占用；
- 将 write/read 建模为物理寄存器 def/use，防止删除或跨越 SIMT 边界调度；
- 正确处理 copy cycle、`i32`/`i64` swap、寄存器压力和部分 slot 读写；
- 对生成的物理寄存器 copy 正常统计 code size，并保留方案 A 的失败检查。

BiSheng 可扩展 named-register 查询和 TPER register model，在寄存器分配前收集
SIMT function 引用的 TPER 集合并加入 reserved set，再由 read/write-register
lowering 生成物理 copy。PTOAS 检测到工具链支持该能力后，再切换 keep/resume
lowering。

方案 B 能彻底移除 keep/resume 对 inline asm 的依赖，但涉及寄存器模型、
寄存器分配和 copy lowering，开发和发布周期长于方案 A。

## 4. PTOAS 临时 Object Patch

### 4.1 采用原因与边界

方案 A 和方案 B 都依赖新 BiSheng 的开发、验证和发布。当前 PTOAS 交付需要
兼容已经发布且仍会生成 `0xffff` 的 BiSheng，因此暂时使用以下流程：

```text
PTO keep/resume
  -> fixed TPER empty inline asm
  -> BiSheng 生成 raw vector device object
  -> PTOAS 根据 ELF symbol size 修补 VF_SIMT code size
  -> device merge / fatobj 打包
```

临时方案只修正 code size，不改变 fixed TPER 的寄存器语义，也不在 PTOAS 中
模拟 BiSheng 的寄存器分配。

patcher 的处理范围如下：

- 仅处理 VPTO vector relocatable device object；
- 仅处理 target 能与 LLVM direct call manifest 中 caller/callee 关系对应的
  `VF_SIMT` callsite；
- 仅把已确认的异常值 `0xffff` 改为由 symbol size 换算的正确值；
- 正确字段保持不变；
- 不修改 EmitC、cube object、ASC frontend 产物、SIMT body、PB 参数或同步逻辑；
- 不解析或重写最终 fatobj 中嵌套的 `.aicore_binary`；
- 不把 patcher 的目标相关逻辑定义为长期 `VF_SIMT` ABI。

### 4.2 流水线插入位置

当前 VPTO fatobj 流程为：

```text
VPTO LLVM module
  -> writeLLVMModule()
  -> BiSheng: LLVM IR -> relocatable vector device object
  -> merge cube/vector device objects
  -> compile host stub and embed device object
  -> fatobj
```

加入 patcher 后：

```text
VPTO LLVM module
  -> 收集 SIMT call manifest
  -> writeLLVMModule()
  -> BiSheng: LLVM IR -> raw vector device object
  -> 校验并生成 patched vector device object
  -> merge cube/vector device objects
  -> compile host stub and embed device object
  -> fatobj
```

调用位置在 `tools/ptoas/ObjectEmission.cpp` 的
`emitVPTOVectorDeviceObject()` 之后、`VPTOFatobjArtifacts` 调用
`mergeDeviceObjects()` 之前。

选择该位置有三个原因：

- object 尚未 strip，保留 `.symtab`；
- scalar kernel 和 `*_simt_entry` 的 `st_value`/`st_size` 已反映最终布局；
- 后续 linker 只搬运 section bytes，不会重新生成已修补的指令。

raw object 和 patched object 分开保存：

```text
ptoas-device-vector-raw-XXXX.o
ptoas-device-vector-patched-XXXX.o
```

没有修补项时，device merge 直接使用 raw object；存在修补项且全部校验通过时，
使用 patched object。raw object 不做原位修改。

### 4.3 编译前收集 SIMT 调用清单

PTOAS 在调用 BiSheng 前从内存中的 `llvm::Module` 收集：

```text
caller function
callsite ordinal
direct simt_entry callee
callee 是否包含 inline asm
```

manifest 只接受 direct call，callee 必须是有定义的 `simt_entry` function。
数据结构可定义为：

```cpp
struct SimtCallSite {
  std::string callerName;
  std::string calleeName;
  unsigned ordinalInCaller;
  bool calleeContainsInlineAsm;
};
```

收集发生在 `applyVPTOLLVMABINames()` 之后，使 manifest 名称与 ELF symbol 名称
一致。

manifest 不记录最终函数字节数，也不直接提供补丁偏移。它只限定允许修改的
caller/callee 集合，并与 object 解码出的调用关系交叉校验。`auto` 模式仅处理
`calleeContainsInlineAsm == true` 的 callsite。

### 4.4 读取原始 Device ELF

patcher 使用 LLVM `Object` API 读取 raw object，并检查：

- 输入是目标架构对应的 relocatable ELF device object；
- 存在唯一的 `.text` 和 `.symtab`；
- manifest 中的 caller/callee 都对应 executable section 内有定义的
  `STT_FUNC` symbol；
- 每个相关 symbol 的地址和大小完全落在所属 section 内；
- callee `st_size` 满足第 2.3 节的全部条件。

每个 SIMT callee 需要记录：

```text
section index
st_value
st_size
symbol name
```

### 4.5 定位 `VF_SIMT` 调用点

patcher 不得在整个 executable section 中搜索异常常量或固定字节模式。该方式
无法证明匹配项属于哪个 caller，也无法证明其 callee。

PTOAS 使用当前 CANN 随 BiSheng 提供的匹配版本 target decoder。decoder 必须
从已发现的 CANN toolchain 路径取得，不能使用 `PATH` 中的任意版本。工具版本、
输出格式或目标架构不受支持时，编译失败。

对每个 manifest caller，patcher 只解码该 caller 的 ELF symbol 范围，并完成
三项检查：

1. LLVM manifest 中存在 caller -> callee direct call 关系；
2. 每个 callsite 重建出的 target 等于 manifest 中某个 callee 的 ELF
   `st_value`；
3. manifest 中每个 callee 至少对应一个最终机器 callsite。

机器优化可能展开或复制 LLVM callsite，因此最终 `VF_SIMT` 数量可以多于 LLVM
direct call 数量。同一个 callee 的每个机器 callsite 都要独立校验。若新
BiSheng 生成了尚未支持的等价目标地址形式，patcher 明确失败，不能退化为按
callsite 顺序猜测 callee。

raw object 可能已解析局部 SIMT label，因此不能要求相关 relocation 必然存在。
若 object 保留了 relocation，可将其作为额外交叉检查。

### 4.6 生成补丁

目标相关 helper 负责读取和写入 `VF_SIMT` code size。通用流程只处理
caller/callee 映射、ELF symbol 和一致性检查，不包含指令编码细节。

每个 callsite 按以下规则处理：

```text
0 < 旧值 < 0xffff && 旧值 <= symbol size 换算结果
  -> BiSheng 已正常编码，保持原值

旧值 == 0xffff && callee contains inline asm
  -> 生成补丁记录

旧值 == 0 或旧值 > symbol size 换算结果
  -> 报错
```

`0xffff` 但 callee 不含 inline asm 时同样报错，不能假设它属于本方案处理的
BiSheng 问题。

补丁只能改变 code size 字段，不能改变指令的其他语义字段、section 大小、
symbol、relocation、alignment 或 ELF header。

### 4.7 两阶段写入

patcher 先完成计划，再生成输出。

计划阶段：

1. 解析完整 ELF、manifest 和 decoder 结果；
2. 为所有 callsite 建立 no-op 或补丁记录；
3. 检查所有文件偏移位于对应 caller 的 executable section 内；
4. 检查补丁范围互不重叠，且没有 relocation 覆盖目标字段；
5. 在内存副本中应用补丁并计算 raw/patched byte diff。

任一检查失败时不写文件。

输出阶段：

1. 将完整内存副本写入新的 patched object；
2. 重新打开 object，检查 ELF section 和 symbol table；
3. 重新读取每个 callsite，确认修补值等于 symbol size 换算结果，未修补的有限
   值保持不变且未越过 callee symbol；
4. 比较 raw/patched，确认所有变化都位于已登记的 code size 字段；
5. 校验通过后，将 patched object 交给 device merge。

## 5. 命令行接口

新增选项：

```text
--vpto-fix-vfsimt-size=auto|off|verify
```

| 模式 | 行为 |
| --- | --- |
| `auto` | 默认值。修补满足全部条件的 `0xffff`；正确字段保持不变 |
| `off` | 跳过检查和修补，用于复现原始 BiSheng 行为 |
| `verify` | 只检查不修补；发现 `0xffff` 时令编译失败 |

不提供跳过一致性检查的 `force` 模式。

每个实际修补输出一条稳定诊断：

```text
PTOAS: patched VF_SIMT size
  caller: rmsnorm_d5120_kernel_mix_aiv
  callee: rmsnorm_d5120_kernel_simt_0_simt_entry
  symbol size: 232 bytes
  result: replaced known invalid size with size derived from the callee symbol
```

所有字段都正确时输出：

```text
PTOAS: VF_SIMT size verification passed; no patch required
```

## 6. 代码改动

### 6.1 新增文件

```text
tools/ptoas/VFSIMTSizePatcher.h
tools/ptoas/VFSIMTSizePatcher.cpp
```

接口定义：

```cpp
enum class VFSIMTSizeFixMode {
  Auto,
  Off,
  Verify,
};

struct VFSIMTSizePatchResult {
  bool changed = false;
  unsigned verifiedCallSites = 0;
  unsigned patchedCallSites = 0;
  std::string objectPath;
};

FailureOr<VFSIMTSizePatchResult> verifyAndPatchVFSIMTSize(
    llvm::Module &module,
    llvm::StringRef rawObjectPath,
    llvm::StringRef patchedObjectPath,
    const CANNToolchain &toolchain,
    VFSIMTSizeFixMode mode,
    llvm::raw_ostream &diagOS);
```

manifest 收集和 object patch 可以拆为内部 helper，ELF 和目标指令细节不进入
`ObjectEmission.cpp`。

### 6.2 修改现有文件

```text
tools/ptoas/ObjectEmission.cpp
  - vector LLVM 编译完成后调用 patcher
  - device merge 选择 raw 或 patched object

tools/ptoas/ObjectEmission.h
  - 传递修补模式和结果类型

tools/ptoas/ptoas.cpp
  - 解析 --vpto-fix-vfsimt-size

tools/ptoas/CMakeLists.txt
  - 加入 VFSIMTSizePatcher.cpp
  - LLVM_LINK_COMPONENTS 加入 Object
```

修补模式通过显式编译配置传入 object emission。patcher 不读取环境变量，也不
依赖隐式全局状态。

## 7. 测试计划

### 7.1 host 单元测试

- 正常解析目标 relocatable ELF；
- 缺失 `.symtab`、`st_size == 0` 或 size 对齐非法时失败；
- code size 超出目标字段范围时失败；
- manifest caller/callee 不唯一时失败；
- 机器优化将一个 LLVM callsite 展开为多个同 target callsite 时通过；
- manifest callee 没有对应机器 callsite 时失败；
- call target 与 ELF callee 地址不一致时失败；
- relocation 覆盖目标字段时失败；
- 补丁范围重叠时失败；
- raw/patched byte diff 超出登记字段时失败。

### 7.2 VPTO 编译测试

至少覆盖：

1. 不含 SIMT call：no-op；
2. SIMT call 不含 inline asm，原始有限 size 不超过 symbol 边界：no-op；
3. fixed TPER keep/resume 产生 `0xffff`：修补为由 symbol size 换算的正确值；
4. 同一 caller 调用两个不同 SIMT callee；
5. 同一个 LLVM callsite 被机器优化展开为多个同 callee callsite；
6. module 包含多个 scalar kernel；
7. cube/vector 混合 module：只修改 vector object；
8. `off` 保留原始异常字段；
9. `verify` 在受影响 BiSheng 上明确失败；
10. 新 BiSheng 已生成正确字段：`auto` 不产生 byte diff。

测试保存并检查以下产物：

```text
input PTO IR
emitted VPTO LLVM IR
raw vector device object
patched vector device object
匹配版本的 target decoder 输出
raw/patched byte diff
最终 fatobj
```

### 7.3 RMSNorm 真机回归

sequence reproducer 已从长期分支测试中删除，真机回归按以下场景执行：

| 场景 | 预期 |
| --- | --- |
| `off`，d5120 -> d7168 | 保留已知失败，作为负对照 |
| `auto`，d5120 单独 | 通过且结果正确 |
| `auto`，d7168 单独 | 通过且结果正确 |
| `auto`，d5120 -> d7168 | 通过且结果正确 |
| `auto`，d4096 -> d5120 -> d7168 | 三个 kernel 均通过且结果正确 |
| `verify`，受影响 BiSheng | 编译失败，不生成错误 fatobj |

同时确认 raw/patched d5120 只变化两个已登记的 `VF_SIMT` code size 字段，
修补值分别为 58 和 580。

## 8. 风险与退出条件

### 8.1 目标 decoder 依赖

临时方案依赖与当前 CANN 匹配的 target decoder。patcher 必须检查工具版本并
拒绝未知格式。若 PTOAS 后续能够直接链接匹配版本的 HiIPU MC decoder，应改用
结构化解码接口，移除对外部工具输出格式的依赖。

### 8.2 异常值的适用范围

patcher 不把所有最大值或 sentinel 都视为错误。只有同时满足以下条件才修补：

- callee 含 inline asm；
- manifest、decoder target 和 ELF symbol 唯一匹配；
- callee `st_size` 合法；
- 原字段正好为本问题产生的 `0xffff`；
- callsite 形式受当前目标 decoder 支持。

如果 ISA 后续定义了合法使用特殊 sentinel 的 kernel 类型，应明确排除该类型，
不能放宽现有匹配条件。

### 8.3 patcher 移除条件

满足以下条件后删除 PTOAS object patcher：

1. 目标 BiSheng 完成方案 A；或者 BiSheng 完成方案 B，PTOAS 同时切换到
   named-register intrinsic lowering；
2. BiSheng 会拒绝负数、未知值和超出字段范围的 code size；
3. `verify` 模式在完整 VPTO SIMT 测试集上均为 no-op；
4. d4096 -> d5120 -> d7168 真机连续运行通过；
5. raw vector object 中不再出现本问题产生的 `0xffff`，正常有限 code size
   非零且不超过对应 `simt_entry` symbol size；
6. TileLang/PTOAS 完成新 CANN 工具链切换，旧 BiSheng 退出支持范围。

删除 patcher 后保留编译期或 CI 校验，防止 `VF_SIMT` 声明范围再次明显大于
callee symbol。
