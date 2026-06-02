# PTOAS Usability Eval Skill

这是 PTOAS 仓库内的通用 Skill 源目录。

支持的客户端入口：
- Codex: `.codex/skills/ptoas-usability-eval/`
- Cursor: `.cursor/skills/ptoas-usability-eval/`
- Trae: `.trae/skills/ptoas-usability-eval/`
- Claude Code: `.claude/skills/ptoas-usability-eval/`

当前覆盖的评估场景：
- `01 算子复现部署`
- `02 算子迁移部署` 的 PTOAS 支撑子集
- `04 算子基本功能实现` 的 PTOAS 工程子集
- `05 特定 shape 性能优化` 的 PTOAS 支撑子集
- `06 泛化 shape 性能优化` 的 PTOAS 支撑子集

当前附带的评分基线：
- `references/ptoas-usability-scorecard-10pt.csv` 直接对齐 `PTOAS-易用性评估指标.xlsx`
- 全表共 `30` 个 `Touch-Point`
- 单项、场景分、总分统一使用 `10 分制`
- `未实测/N/A` 不进入总分分母
- 默认输出 `总分（支撑）` 和 `总分（实测）`
- 默认权重仍为 `01:40%`，`02/04/05/06:15%`

当前评估逻辑：
- 先以 `references/ptoas-usability-scorecard-10pt.csv` 作为分项定义与打分规则的唯一基线
- 再按 `01/02/04/05/06` 场景选择适用的 `Core / Conditional Touch-Points`
- 默认只把适用的 `Core Touch-Points` 放进 repo 级总分

约定：
- `skills/ptoas-usability-eval/` 作为仓库内的通用主副本
- 各客户端目录提供可直接发现的副本，便于不同工具开箱即用
- 修改 Skill 内容时，应同步更新上述四个客户端目录
- 对 `L3/L4` 依赖 Linux/CANN/NPU 的指标，未实测时必须标 `未实测`，不能因为当前机器缺环境直接给 PTOAS 低分
