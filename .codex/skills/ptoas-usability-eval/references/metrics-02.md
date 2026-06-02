# `02 算子迁移部署` 指标子集

`02` 只评 PTOAS 对“迁移到 PTO IR / PTOAS 工具链”的支撑能力。分项定义、量化指标、打分规则统一来自 `ptoas-usability-scorecard-10pt.csv`；本文件只负责说明在 `02` 里默认选哪些 `Touch-Point`。

## 本场景默认选用触点

### Core Touch-Points

- `资料/文档`：`Touch-Point001-011`
- `API / 接口`：`Touch-Point012-013`
- `源码 & 示例类`：`Touch-Point017-021`
- `工具`：`Touch-Point025-026`
- `版本`：`Touch-Point027`
- `运行反馈`：`Touch-Point028-030`

### Conditional Touch-Points

- `Touch-Point014-016`：只有当前任务直接把 PTO sample 当成迁移模板时才纳入
- `Touch-Point022-024`：只有当前任务同时涉及关键链路显性化、安装部署或性能/精度链路时才纳入

## 先声明层级

`02` 场景同样先声明层级：

- `L1 文档审阅层`
- `L2 本地最小运行层`
- `L3 Linux compile-only 层`
- `L4 NPU 上板层`

没有进入对应层，就标 `未实测`。

## 重点看什么

- `Touch-Point001-005`：能不能快速定位迁移入口、IR 手册、样例与验证链路
- `Touch-Point010-013`：版本配套、接口发现性、接口分层说明是否清楚
- `Touch-Point017`：目标功能是否已经被 PTO sample / Demo 覆盖
- `Touch-Point018-019`：命令示例、API 调用样例是否齐全
- `Touch-Point020`：迁移后 sample / validation 编译报错的修复轮次是否可控
- `Touch-Point021`：已有样例或代码能否被业务迁移直接复用
- `Touch-Point025-030`：迁移任务的操作步骤数、工具场景覆盖度和失败日志质量

## 推荐证据

- `docs/PTO_IR_manual.md`
- `test/samples/PyPTOIRParser/README.md`
- `test/samples/*`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`

## 特别说明

- `Touch-Point017` 和 `Touch-Point021` 默认要求“迁移前 / 迁移后”对照物；没有对照物时记 `未实测`
- 如果指标本身依赖仓外业务工程，且用户没有给材料，可记 `N/A`
