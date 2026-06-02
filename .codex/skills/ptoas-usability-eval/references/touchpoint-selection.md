# PTOAS Touch-Point 选型

本文件把 `ptoas-usability-scorecard-10pt.csv` 的 `30` 个 `Touch-Point`，映射成适合 `hw-native-sys/PTOAS` 的 repo 级评估子集。

## 1. 选型原则

- `ptoas-usability-scorecard-10pt.csv` 是分项定义与打分规则的唯一基线。
- PTOAS 是**编译器 / 工具链 / 样例仓库**，默认优先评估仓库内可自证的文档、接口、样例、工具、版本与运行反馈。
- 触点是否纳入，要同时满足两件事：
  - 该 `Touch-Point` 与当前场景直接相关
  - 当前任务能拿到仓库内证据，或者用户补充了前后对照物 / benchmark / 真实运行结果
- 没有证据的项记 `未实测`；超出 PTOAS 仓库边界的项记 `N/A`，不要硬打分。

## 2. 30 个 Touch-Point 的分组

### 2.1 资料 / 文档

- `Touch-Point001` 检索命中成功率
- `Touch-Point002` 文档跳转次数
- `Touch-Point003` 多入口可达率
- `Touch-Point004` 单次任务文档跳转浏览率
- `Touch-Point005` 知识渐进式发布
- `Touch-Point006` 文档结构风格一致率
- `Touch-Point007` 概念跨文档冲突数
- `Touch-Point008` 文档错误点位密度
- `Touch-Point009` 文档场景 / 内容覆盖缺失率
- `Touch-Point010` 版本配套关系准确性
- `Touch-Point011` 资料交付件完备率

### 2.2 API / 接口

- `Touch-Point012` 目标接口平均查找检索轮次
- `Touch-Point013` 渐进式复杂披露覆盖度

### 2.3 源码 & 示例类

- `Touch-Point014` 示例代码一键编译运行成功率
- `Touch-Point015` quick_start / sample 一次跑通率
- `Touch-Point016` 样例覆盖度
- `Touch-Point017` 最小功能实现 Demo 覆盖率
- `Touch-Point018` 命令示例覆盖度
- `Touch-Point019` API 调用样例覆盖率
- `Touch-Point020` 样例代码编译错误检出与修复效率
- `Touch-Point021` 业务代码直接复用改编比例
- `Touch-Point022` 关键链路显性化率
- `Touch-Point023` 认知理解步数

### 2.4 工具 / 版本 / 运行反馈

- `Touch-Point024` 首次安装部署一次性成功率
- `Touch-Point025` 标准任务平均操作步骤数
- `Touch-Point026` 功能 / 场景覆盖率
- `Touch-Point027` 版本检索命中成功率
- `Touch-Point028` 报错携带环境 / 版本 / 上下文信息完整率
- `Touch-Point029` 报错自带排障建议比例
- `Touch-Point030` 无效冗余信息占比

## 3. 场景到 Touch-Point 的默认映射

| 场景 | 默认 Core Touch-Points | 条件 Touch-Points |
| --- | --- | --- |
| `01 算子复现部署` | `Touch-Point001-011`, `Touch-Point014-016`, `Touch-Point018`, `Touch-Point020`, `Touch-Point024-030` | `Touch-Point017`, `Touch-Point021-023` |
| `02 算子迁移部署` | `Touch-Point001-013`, `Touch-Point017-021`, `Touch-Point025-030` | `Touch-Point014-016`, `Touch-Point022-024` |
| `04 算子基本功能实现` | `Touch-Point001-013`, `Touch-Point014-020`, `Touch-Point023-026`, `Touch-Point028-030` | `Touch-Point021-022`, `Touch-Point027` |
| `05 特定 shape 性能优化` | `Touch-Point001-013`, `Touch-Point018-023`, `Touch-Point025-030` | `Touch-Point014-017`, `Touch-Point024` |
| `06 泛化 shape 性能优化` | `Touch-Point001-013`, `Touch-Point018-023`, `Touch-Point025-030` | `Touch-Point014-017`, `Touch-Point024` |

## 4. 条件纳入规则

- `Touch-Point017` 需要迁移前后对照物；没有对照物时记 `未实测`。
- `Touch-Point020` 需要真实编译 / validation 过程；只读文档时记 `未实测`。
- `Touch-Point021` 需要 PR diff、迁移前后 case，或业务侧复用记录；没有材料时记 `未实测`。
- `Touch-Point022` 主要用于 `05/06`；如果当前任务没有性能 / 精度关键链路，不强行纳入。
- `Touch-Point028-030` 需要真实日志；只看 README 时不能给“实测分”。

## 5. 默认排除项

本 Skill 只覆盖 `ptoas-usability-scorecard-10pt.csv` 这 `30` 个 `Touch-Point`。任何不在这张表里的生态级、友商对标级、产品矩阵级指标，默认都不纳入 PTOAS repo 级总分。

## 6. 使用要求

- 每次正式评估前，先在输出里给出 `触点选择`。
- 默认只打 `Core Touch-Points`。
- `Conditional Touch-Points` 只有在证据存在时才纳入，并明确说明为什么本次纳入。
- 任何没有证据的项，必须明确标成 `未实测` 或 `N/A`，不能为了凑总分而补猜。
