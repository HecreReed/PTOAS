# `02 算子迁移部署` 指标

## 本场景默认选用触点

- `资料/文档`: `TP001-005, TP008-009, TP011, TP013, TP017`
- `API / 接口`: `TP019-020, TP032`
- `源码 & 示例类`: `TP035-037, TP039, TP044, TP046-047`
- `工具`: `TP051-052`
- `运行反馈`: `TP062-064`
- `Conditional`: `TP040-041, TP033, TP061`

这里只评估 PTOAS 仓库对“迁移到 PTO IR / PTOAS 工具链”的支撑能力，不把 PTOAS 包装成完整业务算子迁移平台。

说明：保留用户原始口径，`10 分制` 与 `100 分制` 混用，不强行归一。

## 先声明层级

`02` 场景同样先声明层级：

- `L1 文档审阅层`
- `L2 本地最小运行层`
- `L3 Linux compile-only 层`
- `L4 NPU 上板层`

没有进入对应层，就标 `未实测`。

## 共享易学习项

`02` 的 `文档获取` / `文档学习` 默认复用 [metrics-01.md](metrics-01.md) 中对应规则；只是目标从“复现部署”改成“找到迁移入口、IR 语义、样例和验证链路”。

推荐优先证据：
- `docs/PTO_IR_manual.md`
- `test/samples/PyPTOIRParser/README.md`
- `README.md`
- `docs/no_npu_compile_only_guide_zh.md`

## 易迁移

### 算子迁移

| 指标 | 在 PTOAS 中怎么测 | 主要证据 | 可评分层级 | 打分规则 |
| --- | --- | --- | --- | --- |
| tiling 迁移完备度 | 统计迁移前定义的 tiling 结构中，已成功落到 PTO IR / sample / generated testcase 的比例 | `docs/PTO_IR_manual.md`, `test/samples/*`, 用户提供的前后对照物 | `L1-L4` | `100%=10`；`80%=8`；`60%=6`；`40%=4`；`20%=2` |
| kernel 迁移完备度 | 统计迁移前 kernel 逻辑中，已能由 PTO case / sample / generated testcase 覆盖的比例 | 同上 | `L1-L4` | `100%=10`；`80%=8`；`60%=6`；`40%=4`；`20%=2` |
| 算子功能失败率 | 迁移后 compile-only / validation / board case 失败比例 | `runop.sh`, `generate_testcase.py`, `run_remote_npu_validation.sh` | `L3-L4` | `0% 失败=10`；`10%=8`；`20%=6`；`30%=4`；`40%=2` |
| 算子性能劣化率 | 迁移后相对迁移前 baseline 的性能增幅 | 用户 baseline、`performance_issue.yml`、实测日志 | `L4` | 劣化 `<5%=10`；`10%=8`；`20%=6`；`30%=4`；`40%=2` |
| tiling 代码修改率 | 迁移时 tiling 相关代码修改行数 / 总行数 | 用户对照 case、PR diff | `L1-L4` | `0%-5%` 高分；`5%-15%` 次高；`15%-25%` 中；`25%-35%` 低；`>35%` 最低 |
| kernel 代码修改率 | 迁移时 kernel 相关代码修改行数 / 总行数 | 用户对照 case、PR diff | `L1-L4` | `0%-5%` 高分；`5%-15%` 次高；`15%-25%` 中；`25%-35%` 低；`>35%` 最低 |
| 算子迁移总耗时 | 从开始迁移到迁移完成，不含最终验证环节 | 真实执行记录 | `L2-L4` | `<=10 分钟` 高分；`10-20` 次之；`20-30` 中；`30-60` 低；`>60` 最低 |
| 算子迁移成功率 | 迁移任务成功率 | 真实执行记录 | `L2-L4` | `100%` 高分；`80%-99%` 次之；`60%-79%` 中；`40%-59%` 低；`<40%` 最低 |

## 在 PTOAS 中如何落地这些指标

### 可直接量化的前提

只有当前任务同时具备以下材料时，才量化迁移完备度/修改率/失败率：
- 迁移前输入：上游 IR、旧 kernel、旧 tiling，至少一种
- 迁移后输入：PTO `.pto`、sample、generated testcase，至少一种
- 对应验证链路：compile-only 或 board validation

### 没有对照物时怎么办

- 只有仓库文档和样例，没有迁移前材料：
  - 可以评“入口是否清晰”“样例/脚本是否齐全”
  - 不能给出完备度/修改率/性能劣化率，记 `未实测`
- 指标本身依赖仓外业务工程，且用户没给材料：
  - 记 `N/A`

## 推荐默认证据路径

- `docs/PTO_IR_manual.md`
- `test/samples/PyPTOIRParser/README.md`
- `test/samples/*`
- `docs/no_npu_compile_only_guide_zh.md`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
