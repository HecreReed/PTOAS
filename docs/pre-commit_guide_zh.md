# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

# pre-commit 代码质量检查使用指南

本项目使用 [pre-commit](https://pre-commit.com/) 框架，在 git commit 前自动执行代码格式化、拼写检查和开源合规检查，确保提交的代码符合项目规范。

## 安装步骤

### 步骤 1: 安装 pre-commit 框架

```bash
# 使用 pip（推荐）
pip install pre-commit

# 验证安装
pre-commit --version
# 输出: pre-commit 4.x.x
```

> **Windows 用户**: 确保已安装 Python 和 pip。

### 步骤 2: 进入项目目录

```bash
cd /path/to/your/pto-as
# 例如 cd d:\complianceRepo\CANN\pto-as
```

### 步骤 3: 安装 Git Hooks

```bash
# 在项目根目录运行
pre-commit install
```

安装后，每次 `git commit` 会自动触发 pre-commit 检查。

### 步骤 4: 验证安装（可选）

```bash
# 测试 hook（不会真正提交）
git commit --allow-empty -m "test pre-commit"
```

## 检查项说明

本项目配置了 13 个 hook，分为以下几类：

### 基础检查（pre-commit-hooks）

| Hook | 说明 |
|------|------|
| trailing-whitespace | 清除行尾多余空格 |
| end-of-file-fixer | 确保文件末尾有且只有一个换行符 |
| check-yaml | 校验 YAML 文件语法（允许多文档） |
| check-added-large-files | 防止意外提交大文件 |
| check-merge-conflict | 检测未解决的合并冲突标记 |
| detect-private-key | 检测意外提交的私钥文件 |
| check-json | 校验 JSON 文件语法 |

### C/C++ 代码格式化（clang-format）

使用 clang-format v18.1.8 对 `.c/.h/.cpp/.hpp/.cc/.hh/.cxx/.hxx` 文件执行格式化。

格式化规则定义在 `.clang-format` 文件中，主要配置：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| BasedOnStyle | Google | 基于 Google 风格 |
| IndentWidth | 4 | 4 空格缩进 |
| ColumnLimit | 120 | 120 列宽 |
| SortIncludes | false | 不自动重排 include 顺序 |
| BreakBeforeBraces | Custom | 自定义大括号换行 |
| AfterFunction | true | 函数左大括号换行 |
| PointerAlignment | Left | 指针左对齐（`int* p`） |

> **注意**: `test/samples/` 和 `test/npu_validation/templates/` 目录已排除 clang-format，因为这些目录的模板文件包含 `@PLACEHOLDER@` 占位符语法，clang-format 会在 `@` 两侧插入空格导致占位符替换失败。

### Python 代码检查（ruff）

使用 ruff v0.14.14 执行 Python 代码检查和格式化：

- **ruff-check**: 代码规范检查（E402、F841 等），自动修复（`--fix`）
- **ruff-format**: 代码格式化

per-file-ignores 配置（`ruff.toml`）：

| 文件 | 忽略规则 | 原因 |
|------|----------|------|
| `test/samples/**/*.py` | E402, F841 | import 前需设置 sys.path；中间 MLIR op 变量用于副作用 |
| `**/pto.py` | F821 | 引用 TableGen 动态生成的 op 类 |
| `**/package.py` | F841 | 保留调试变量 |

### 拼写检查（codespell + typos）

| 工具 | 版本 | 说明 |
|------|------|------|
| codespell | v2.4.1 | 常见拼写错误检查 |
| typos | v1.32.0 | 拼写错误检查（使用 `typos.toml` 配置） |

项目专有名词和历史遗留拼写已加入白名单（`typos.toml`），例如：CANN、nd、excuted、compatiable 等。

### OAT 合规检查

[OAT](https://gitee.com/openeuler/oat-py)（Open Source Audit Tool）用于检查开源合规性：

- **版权头检查**: 校验源码文件是否包含 CANN Open Software License 头
- **屏蔽列表**: `OAT.xml` 配置了不需要检查的目录和文件类型（test/、.codex/、.claude/、\*.xml、\*.yaml 等）
- **并发安全**: `scripts/oat_check.sh` 使用带 PID 的临时目录，避免 pre-commit 按文件类型分组调用时报告互相覆盖

## 手动运行 pre-commit

```bash
# 对所有文件运行（全量扫描）
pre-commit run --all-files

# 对暂存区文件运行（默认行为）
pre-commit run

# 只运行指定 hook
pre-commit run clang-format --all-files
pre-commit run ruff-check --all-files
pre-commit run typos --all-files

# 跳过某个 hook（不推荐，仅紧急情况）
SKIP=typos git commit -m "your message"
```

## 常见问题

### Q: clang-format 修改了 include 顺序导致编译失败？

clang-format 配置了 `SortIncludes: false`，不会自动重排 include 顺序。如果你的 include 被重排了，请检查是否误传了 `--style` 命令行参数（会覆盖 `.clang-format` 文件）。

### Q: PTO.cpp 中 `#include "xxx.cpp"` 的顺序被改变了？

本项目部分文件使用 `#include "xxx.cpp"` 的方式引入实现（如 PTO.cpp、PTOInsertSync.cpp），这些 include 有严格的依赖顺序。`SortIncludes: false` 确保了顺序不被改变。

### Q: typos 报告某个拼写错误，但这是项目专有名词？

将专有名词加入 `typos.toml` 的 `[default.extend-words]` 或 codespell 的 `-L` 参数白名单。

### Q: OAT 检查报告缺少 License 头？

为源码文件补充 CANN Open Software License 头。XML/yaml/info/pto/golden 等文件类型已通过 `OAT.xml` 屏蔽列表排除，无需补充。

## 相关文件

| 文件 | 说明 |
|------|------|
| `.pre-commit-config.yaml` | pre-commit 主配置文件（13 个 hook） |
| `.clang-format` | C/C++ 格式化规则 |
| `ruff.toml` | Python 代码检查 per-file-ignores 配置 |
| `typos.toml` | typos 拼写检查白名单 |
| `OAT.xml` | OAT 合规检查配置 |
| `scripts/oat_check.sh` | OAT 检查脚本 |
