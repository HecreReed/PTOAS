# `04 算子基本功能实现` 指标子集

`04` 只评 PTOAS 直接覆盖到的“示例 / 编译 / 验证 / 反馈”链路。分项定义、量化指标、打分规则统一来自 `ptoas-usability-scorecard-10pt.csv`；本文件只负责说明在 `04` 里默认选哪些 `Touch-Point`。

## 本场景默认选用触点

### Core Touch-Points

- `资料/文档`：`Touch-Point001-011`
- `API / 接口`：`Touch-Point012-013`
- `源码 & 示例类`：`Touch-Point014-020`, `Touch-Point023`
- `工具`：`Touch-Point024-026`
- `运行反馈`：`Touch-Point028-030`

### Conditional Touch-Points

- `Touch-Point021-022`：只有当前任务确实涉及业务复用比例或性能 / 精度关键链路时才纳入
- `Touch-Point027`：如果当前问题与版本切换、branch/tag/release 直接相关，再把版本检索命中率纳入

## 先声明层级

`04` 子集同样先声明层级：

- `L1 文档审阅层`
- `L2 本地最小运行层`
- `L3 Linux compile-only 层`
- `L4 NPU 上板层`

未进入对应层级，只能标 `未实测`。

## 重点看什么

- `Touch-Point001-005`：能不能快速找到目标样例、CLI、验证脚本
- `Touch-Point008-013`：文档是否准确，接口文档是否够查、够懂
- `Touch-Point014-016`：样例能否跑通，样例目录是否覆盖输入 / 输出 / golden / compare / validation
- `Touch-Point018-020`：命令示例是否齐全，API 调用样例是否可运行，编译报错修复轮次是否可控
- `Touch-Point023`：要改通当前 sample / validation，需要额外理解多少宏 / 函数 / 脚本 / 配置项
- `Touch-Point024-026`：从安装到标准任务完成，工具链步骤数和场景覆盖度是否合理
- `Touch-Point028-030`：失败时是否给了足够可诊断的日志

## 推荐证据

- `README.md`
- `docs/PTO_IR_manual.md`
- `test/samples/MatMul/`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
- `.github/workflows/ci.yml`

## 默认不直接量化的部分

这些超出 PTOAS 仓库直接支撑范围，默认记 `N/A`：

- 算子设计-需求分析
- 算子设计-方案实现
- 完整业务工程联调
