# PTOAS 评估范围映射

## 六大场景映射

| 场景 | 结论 | 处理方式 | PTOAS 中可量化的部分 | 默认不直接量化的部分 |
| --- | --- | --- | --- | --- |
| `01 算子复现部署` | 主评估场景 | 正常评分 | 构建、CLI、Python 绑定、sample、compile-only、board-validation | 无 |
| `02 算子迁移部署` | 有限纳入 | 评“迁移支撑能力” | PTO IR 入口、样例/PyPTO 快照、compile-only/validation、迁移前后对照样例 | 业务框架集成、完整模型接线、仓外工程改造 |
| `03 builtin 算子定制修改` | 默认不纳入 | 标 `N/A`，必要时只做差距说明 | 无固定模板 | builtin/业务算子定制主流程 |
| `04 算子基本功能实现` | 次级支撑 | 只评 PTOAS 直接支撑的工程子链路 | 文档、样例、编译配置、工程编译、运行反馈、验证 | 完整需求分析、完整方案设计、通用算子研发全过程 |
| `05 特定 shape 性能优化` | 条件纳入 | 评“特定 shape 性能优化支撑能力” | 性能文档、固定 shape 样例、性能数据获取入口、编译验证、精度/性能验证 | 没有 baseline 时的性能提升幅度、仓外友商对标 |
| `06 泛化 shape 性能优化` | 条件纳入 | 评“多 shape / dynamic shape 支撑能力” | `valid_shape`/dynamic shape 文档、泛化样例、多 shape 验证链路 | 没有多 shape 基线矩阵时的平均/最大收益结论 |

## 评估层级

任何量化前都先声明层级：

- `L1 文档审阅层`：只看仓库内容，不跑命令。
- `L2 本地最小运行层`：当前机器已有 PTOAS 产物，可验证最小命令。
- `L3 Linux compile-only 层`：Linux + CANN/bisheng + `PTO_ISA_ROOT`，不要求带卡。
- `L4 NPU 上板层`：带卡 Linux、驱动、ACL、设备节点和用户权限齐全。

评分规则：
- 某层未覆盖，只能标 `未实测`。
- 不得因为本机不是 Linux / 没有 `bisheng` / 没有 NPU 就对 PTOAS 本身扣分。
- 文档是否清楚说明这些前置条件，才属于可评分项。

## `02` 的适用边界

PTOAS 可以支撑 `02`，但只能评“迁移支撑能力”，不能假装它已经覆盖完整业务迁移平台。

可以评分的证据：
- `docs/PTO_IR_manual.md`：IR 层级、tile/view/valid-shape 语义、Level-2/Level-3 入口
- `test/samples/PyPTOIRParser/README.md`：来自 pypto `ir_parser` 的 vendored `.pto` 快照
- `test/samples/*`：迁移后 PTO case、样例目录、`golden.py`/`compare.py`
- `docs/no_npu_compile_only_guide_zh.md`、`runop.sh`、remote validation 脚本：迁移后 compile/validate 链路

`02` 里这些指标只有在“有前后对照”时才能量化：
- `tiling 迁移完备度`
- `kernel 迁移完备度`
- `tiling/kernel 代码修改率`
- `功能失败率`
- `性能劣化率`

如果当前任务没有“迁移前/迁移后”对照 case：
- 记 `未实测`，不要臆造百分比。
- 若该项本质依赖仓外业务工程，且用户没有给材料，可记 `N/A` 并说明原因。

## `04` 的适用边界

`04` 只纳入这些与 PTOAS 仓库直接对应的子项：
- 文档获取
- 文档学习
- 获取示例代码
- 学习/理解示例代码
- 运行示例代码
- 编译配置
- 算子工程编译
- 功能调试
- 精度调试中 repo 已有 compare/golden/validation 的部分

默认排除：
- 算子设计-需求分析
- 算子设计-方案实现
- 通用算子代码开发全过程
- 脱离 PTOAS 仓库的业务工程联调

## `05` 的适用边界

PTOAS 可以支撑 `05`，但重点是“特定 shape 性能优化的仓内支撑能力”。

可直接使用的仓内证据：
- `docs/PTO_IR_manual.md`：Level-2/Level-3、layout、partition、reshape、valid-shape 等语义
- `.github/ISSUE_TEMPLATE/performance_issue.yml`：性能问题收集模板
- `test/samples/FlashAttention/`, `test/samples/GQA/`, `test/samples/FFN/`, `test/samples/MatMul/`：固定 shape 样例
- `runop.sh`、compile-only 文档、remote validation 脚本：生成、编译、上板验证链路

`05` 里这些项通常至少要到 `L4` 才能量化：
- 性能数据采集耗时
- 性能报告完备度
- 优化迭代次数/耗时/成功率
- 性能提升幅度
- 精度一致性验证

如果只有文档和样例，没有真实 benchmark/baseline：
- 文档和入口可评分
- 真实性能数字、友商对标、提升幅度一律 `未实测` 或 `N/A`

## `06` 的适用边界

PTOAS 可以支撑 `06`，但重点是“多 shape / dynamic shape / valid-shape 的泛化支撑能力”。

可直接使用的仓内证据：
- `docs/PTO_IR_manual.md`
- `test/samples/SetValidShape/`
- `test/samples/LayoutInference/`
- `test/samples/Partition5D/`
- `test/samples/planmemory/`
- 其他带动态 shape / layout / partition / subview / reshape 的样例

`06` 里这些项没有多 shape 基线矩阵时不能硬算：
- 平均优化迭代次数
- 平均优化耗时
- 平均性能提升幅度
- 最大性能提升幅度
- 多 shape 精度保持率

如果当前任务只看到 IR 设计与样例，没有多 shape 实测：
- 文档、样例覆盖度可评分
- 多 shape 性能/精度结果记 `未实测`

## `未实测` 与 `N/A` 的使用规则

记 `未实测`：
- 仓库里有入口，但当前会话没有进入对应层级
- 缺 Linux/CANN/NPU/baseline，导致只能停在文档或最小运行层
- 缺迁移前后对照物，无法给出完成度/收益率

记 `N/A`：
- 指标本身超出 PTOAS 仓库边界
- 用户要求的是仓外业务工程能力，当前仓库没有独立证据链
- 友商对标、完整业务迁移、完整 builtin 定制等不是本 Skill 的默认量化对象

## 使用原则

- 如果用户没有特别说明，直接按 `01` 做主评估。
- 如果用户要求顺带看 `02/04/05/06`，按本文件的边界补相应子集。
- 可以做“支持度评分”，但不要把仓内文档/样例的存在，夸大成业务侧已经打通。
- 能量化就量化；不能量化就明确说明为什么是 `未实测` 或 `N/A`。
