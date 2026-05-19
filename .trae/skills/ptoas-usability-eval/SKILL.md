---
name: ptoas-usability-eval
description: Evaluate PTOAS repository usability for operator reproduction/deployment and the PTOAS-supported subset of basic operator implementation. Use PTOAS repo docs, scripts, samples, and CI config as evidence; score scene 01 as the primary template, score only the build/run/validation subset of scene 04, and mark unsupported scene 02/03/05/06 items as N/A.
---

# PTOAS Usability Eval

当用户要评估 `hw-native-sys/PTOAS` 的易用性，或要按 `01 算子复现部署` / `04 算子基本功能实现` 给 PTOAS 打分时，使用这个 Skill。

## 默认范围

- 默认按 `01 算子复现部署` 评估。
- 只对 `04 算子基本功能实现` 中 PTOAS 直接支撑的子链路打分：文档获取、文档学习、示例定位、示例理解、编译配置、工程编译、示例运行、运行反馈/验证。
- 不要强行覆盖 `02 算子迁移部署`、`03 builtin 算子定制修改`、`05 特定 shape 性能优化`、`06 泛化 shape 性能优化`。
- 对 `04` 里超出 PTOAS 边界的项，例如完整需求分析、完整方案设计、通用算子编码开发、精度/性能优化全流程，默认标 `N/A`。

先读 [references/scope.md](references/scope.md) 确认映射边界。

## 证据来源

优先只用仓库内证据，不把仓外经验当成主证据。固定入口见 [references/evidence-checklist.md](references/evidence-checklist.md)。

高优先级证据：
- `README.md`
- `docs/no_npu_compile_only_guide_zh.md`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
- `test/samples/*/README.md`
- `.github/workflows/ci.yml`

## 工作流

1. 先判断用户要的是 `01`、`04-子集`，还是两者都要；未说明时默认 `01`。
2. 从仓库内收集证据，记录每次检索轮次、文档跳转次数、执行命令、耗时、成功/失败结果。
3. `01` 场景读 [references/metrics-01.md](references/metrics-01.md)。
4. `04` 场景读 [references/metrics-04.md](references/metrics-04.md)。
5. 对每个指标都输出：原始观测值、评分、证据路径、说明。没有实测的数据不要猜，记为 `未实测` 或 `N/A`。
6. 明确区分：
   - PTOAS 仓库已提供的能力
   - 外部前置条件，例如 LLVM、CANN、NPU、`pto-isa`、驱动/权限
7. 若文档描述与实际运行冲突，以实际命令结果为准，并指出冲突位置。

## 计量规则

- `检索轮次`：每次新的定向搜索或定位尝试算 1 轮。
- `文档跳转次数`：命中首个目标文档后，每跨一个文档/README/脚本入口算 1 次。
- `耗时`：尽量记录真实墙钟时间；拿不到就写 `未实测`，不要臆测。
- `成功率`：只基于当前任务里真实执行或真实定位到的结果计算。
- `N/A`：只用于超出 PTOAS 能力边界，或当前任务明确未执行且无法合理推断的项。

## 输出格式

按下面顺序输出：

1. `评估范围`
2. `分项评分`
3. `关键证据`
4. `主要短板`
5. `建议动作`

如果用户只要简版结论，也要至少保留：场景归类、总评、最低分项、证据路径。
