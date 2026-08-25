# PTOAS (PTO Assembler & Optimizer)

## 版本
- 版本号：v0.51
- 发布日期：2026-02-14

## 变更摘要
- PTOAS 首次发布

## 概述
PTOAS（PTO Assembler & Optimizer）是面向 PTO Bytecode 的编译器工具链，基于 LLVM/MLIR LLVM19 VPTO 分支 `vpto-dev/llvm-project:feature-vpto` 构建。它提供 PTO Dialect 的定义、解析、验证、优化与代码生成能力，并输出可调用 `pto-isa` 的 C++ 代码。

PTOAS很快将集成到以下框架中，敬请期待
- PyPTO
- TileLang

## 本仓库的目标用户
PTOAS 主要面向：
- 编译器与框架后端开发者
- 高性能算子/内核开发者
- 需要进行 PTO Bytecode 生成、调试与落地的工程团队

## 主要能力
- PTO Dialect 全流程（定义、解析、验证、打印）
- 与 Tile 抽象/地址空间/同步模型配套的 IR 支撑
- PTO Bytecode → C++ 生成
- Python 端的 Dialect 构建与测试样例
- `pto.textract` ND-to-2xNZ 双输出 form：一个 ND source + 四项 index + 两个 NZ 布局 DPS destination（PTO-ISA `TEXTRACT` 七参数 overload）

### `pto.textract` ND-to-2xNZ 双输出 form（设计文档：docs/designs/textract-nd-to-2xnz-design.md）
- 同一 op 名 `pto.textract` 按完整的五段 operand schema 区分单输出与 ND-to-2xNZ 双输出 form；单输出 canonical 文本、C++/Python 调用面与 PTOBC v0 fixture 保持兼容。
- EmitC 生成一次公开七参数调用 `TEXTRACT(dst0, dst1, src, row0, col0, row1, col1)`；PTOBC v0 中双输出走 generic record，不复用四/五 operand 固定 opcode。
- Python 提供 `pto.TExtractOp.build_nd_to_2xnz(...)` 工厂；旧 `pto.textract(src, index_row, index_col, dst, ...)` 与 `.indexRow/.indexCol/.dst` 属性保持源兼容。
- 兼容性边界：`Properties::operandSegmentSizes` 从六段变为五段布局，`getODSOperands()` 与旧 adaptor 的 fixed-field accessor 不在兼容范围内；MLIR generic assembly 使用新的五段 schema。

## 平台与依赖最低配置
- **操作系统**：macOS (Darwin) 或 Linux (Ubuntu 20.04+)
- **编译器**：Clang >= 12 或 GCC >= 9（支持 C++17）
- **构建工具**：CMake >= 3.20，Ninja
- **Python**：Python 3.8+

## 如何使用PTOAS以及PTO IR的详细描述
- 构建与环境配置：`README.md`
- PTO Bytecode 定义：`docs/PTO_IR_manual.md`
