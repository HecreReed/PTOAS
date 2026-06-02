# `06 泛化 shape 性能优化` 指标子集

`06` 只评 PTOAS 对“多 shape / dynamic shape / valid-shape 性能优化”的支撑能力。分项定义、量化指标、打分规则统一来自 `ptoas-usability-scorecard-10pt.csv`；本文件只负责说明在 `06` 里默认选哪些 `Touch-Point`。

## 本场景默认选用触点

### Core Touch-Points

- `资料/文档`：`Touch-Point001-013`
- `源码 & 示例类`：`Touch-Point018-023`
- `工具`：`Touch-Point025-026`
- `版本`：`Touch-Point027`
- `运行反馈`：`Touch-Point028-030`

### Conditional Touch-Points

- `Touch-Point014-016`：只有当前任务把 sample 当成泛化 shape 优化起点时才纳入
- `Touch-Point017`：只有用户提供迁移前后对照物时才纳入
- `Touch-Point024`：只有本次真实做了从零安装部署，才把“首次安装部署一次性成功率”纳入

## 先声明层级

`06` 场景通常至少需要：

- `L1`：评文档、样例、入口
- `L3`：评 compile-only、工程编译、多 shape 验证链路准备
- `L4`：评真实多 shape 性能、精度一致性、收益

## 重点看什么

- `Touch-Point001-005`：能不能快速找到 dynamic shape、valid-shape、layout、性能相关文档和样例
- `Touch-Point012-013`：CLI / Python / IR 接口是否容易找到，是否有从简单到复杂的说明
- `Touch-Point018-019`：命令示例、API 调用样例是否足以覆盖多 shape / valid-shape 典型用法
- `Touch-Point021`：已有样例或代码能否直接复用到当前泛化 shape 优化任务
- `Touch-Point022`：性能数据采集、精度验证、PyPTO sample 关键链路是否清晰可见
- `Touch-Point023`：为了理解泛化 shape 链路，需要额外读多少脚本 / 宏 / 配置项
- `Touch-Point025-030`：标准任务步骤数、工具场景覆盖度与失败日志质量是否合理

## 推荐证据

- `docs/PTO_IR_manual.md`
- `test/samples/SetValidShape/`
- `test/samples/LayoutInference/`
- `test/samples/Partition5D/`
- `test/samples/planmemory/`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`

## 特别说明

- 没有多 shape 基线矩阵或没有上板实测时，不要硬造收益结论
- `Touch-Point021-023` 可以先形成“支撑分”；真实性能 / 精度收益只有到 `L4` 才能进“实测分”
