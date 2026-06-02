---
name: ptoas-usability-eval
description: Evaluate PTOAS repository usability across scene 01 as the primary template plus the PTOAS-supported subsets of scenes 02, 04, 05, and 06. Always classify the evaluation by environment layer first, use only repo-native docs/scripts/samples/CI as primary evidence, use the 30-touch-point 10-point scorecard as the source of truth, and mark unsupported or untested dimensions as 未实测 or N/A.
---

# PTOAS Usability Eval

当用户要评估 `hw-native-sys/PTOAS` 的易用性，或要按 `01/02/04/05/06` 给 PTOAS 打分时，使用这个 Skill。

## 默认范围

- `01 算子复现部署`：主评估场景，正常评分。
- `02 算子迁移部署`：纳入，但只评 PTOAS 仓库能直接支撑的迁移入口、样例、IR/脚本、编译验证链路。
- `04 算子基本功能实现`：纳入，但只评 PTOAS 直接覆盖的示例、编译、验证、反馈子链路。
- `05 特定 shape 性能优化`：纳入，但只评 PTOAS 的文档、样例、性能数据获取入口、编译验证、精度/性能验证支撑能力。
- `06 泛化 shape 性能优化`：纳入，但只评 PTOAS 的 dynamic/valid-shape、多 shape 样例与验证支撑能力。
- `03 builtin 算子定制修改`：默认不纳入量化，标 `N/A`；必要时只做差距说明。

先读 [references/touchpoint-selection.md](references/touchpoint-selection.md) 选定适用触点，再读 [references/scope.md](references/scope.md) 确认各场景的边界和 `未实测/N/A` 规则。

## 先判层级

开始评分前，必须先声明本次评估覆盖到哪一层。没有层级，不能直接混着打分。

可选层级：
- `L1 文档审阅层`：只看仓库文档、脚本、样例、CI，不做运行。
- `L2 本地最小运行层`：当前机器已有 `ptoas` / `ptobc` / Python 绑定，可做最小命令验证。
- `L3 Linux compile-only 层`：需要 Linux + CANN/bisheng + `PTO_ISA_ROOT`，不要求带卡。
- `L4 NPU 上板层`：需要带卡 Linux、驱动、权限、`/dev/davinci*` 与对应用户组。

约束：
- 没进入某层，就把该层指标记为 `未实测`，不能因为当前机器缺环境就给 PTOAS 低分。
- `bisheng` / CANN compile-only 一般属于 `L3`，不应在本地 Mac 上硬打低分。
- 带卡运行、设备权限、驱动、ACL、用户组属于 `L4`。

## 证据来源

优先只用仓库内证据，不把仓外经验当成主证据。固定入口见 [references/evidence-checklist.md](references/evidence-checklist.md)。

高优先级证据：
- `README.md`
- `docs/no_npu_compile_only_guide_zh.md`
- `docs/PTO_IR_manual.md`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
- `test/samples/PyPTOIRParser/README.md`
- `test/samples/FlashAttention/`, `test/samples/GQA/`, `test/samples/FFN/`
- `test/samples/SetValidShape/`, `test/samples/LayoutInference/`, `test/samples/Partition5D/`, `test/samples/planmemory/`
- `.github/workflows/ci.yml`
- `.github/ISSUE_TEMPLATE/performance_issue.yml`

## 工作流

1. 先判断用户要的是 `01`、`02`、`04`、`05`、`06` 中哪些场景；未说明时默认 `01`。
2. 再判断本次覆盖层级：`L1/L2/L3/L4`。输出中必须显式写出来。
3. 先读 [references/touchpoint-selection.md](references/touchpoint-selection.md)，按场景选定本次的 `Core / Conditional` 触点。
4. 从仓库内收集证据，记录每次检索轮次、文档跳转次数、执行命令、耗时、成功/失败结果。
5. `01` 场景读 [references/metrics-01.md](references/metrics-01.md)。
6. `02` 场景读 [references/metrics-02.md](references/metrics-02.md)。
7. `04` 场景读 [references/metrics-04.md](references/metrics-04.md)。
8. `05` 场景读 [references/metrics-05.md](references/metrics-05.md)。
9. `06` 场景读 [references/metrics-06.md](references/metrics-06.md)。
10. 先读 [references/ptoas-usability-scorecard-10pt.csv](references/ptoas-usability-scorecard-10pt.csv) 取得分项定义、量化指标、打分规则与 VOD 备注；需要汇总总分时，再读 [references/scoring.md](references/scoring.md)。
11. 对每个指标都输出：原始观测值、评分、证据路径、说明。没有实测的数据不要猜，记为 `未实测` 或 `N/A`。
12. 明确区分：
    - PTOAS 仓库已提供的能力
    - 外部前置条件，例如 LLVM、CANN、`pto-isa`、NPU、驱动/权限、业务 baseline
13. 若文档描述与实际运行冲突，以实际命令结果为准，并指出冲突位置。
14. 默认给两个总分：`总分（支撑）` 和 `总分（实测）`。如果用户只要分项，不强制输出总分。

## 计量规则

- 单项、场景分、总分统一使用 `10 分制`。
- 分项字段定义、适用层级、量化指标、打分规则、VOD 备注，以 [references/ptoas-usability-scorecard-10pt.csv](references/ptoas-usability-scorecard-10pt.csv) 为准。
- `检索轮次`：每次新的定向搜索或定位尝试算 1 轮。
- `文档跳转次数`：命中首个目标文档后，每跨一个文档/README/脚本入口算 1 次。
- `耗时`：尽量记录真实墙钟时间；拿不到就写 `未实测`，不要臆测。
- `成功率`：只基于当前任务里真实执行或真实定位到的结果计算。
- `未实测`：当前会话未覆盖到对应环境层级，或该层级前置条件不存在，或缺少前后对照 baseline。
- `N/A`：只用于超出 PTOAS 能力边界，或当前任务明确不纳入本次评估范围的项。

## 输出格式

按下面顺序输出：

1. `评估范围`
2. `触点选择`
3. `评估层级`
4. `总分（支撑）`
5. `总分（实测）`
6. `分场景评分`
7. `覆盖说明`
8. `关键证据`
9. `主要短板`
10. `建议动作`

如果用户只要简版结论，也要至少保留：场景归类、评估层级、总评、最低分项、证据路径。
