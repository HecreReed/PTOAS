# `01 算子复现部署` 指标子集

`01` 是 PTOAS 的主评估场景。分项定义、量化指标、打分规则统一来自 `ptoas-usability-scorecard-10pt.csv`；本文件只负责说明在 `01` 里默认选哪些 `Touch-Point`。

## 本场景默认选用触点

### Core Touch-Points

- `资料/文档`：`Touch-Point001-011`
- `源码 & 示例类`：`Touch-Point014-016`, `Touch-Point018`, `Touch-Point020`
- `工具`：`Touch-Point024-026`
- `版本`：`Touch-Point027`
- `运行反馈`：`Touch-Point028-030`

### Conditional Touch-Points

- `Touch-Point017`：只有用户给了“复现前后对照样例 / 迁移目标”时才纳入
- `Touch-Point021-023`：只有当前任务需要衡量复用比例、关键链路显性化或认知理解步数时才纳入

## 先声明层级

在 `01` 场景里，必须先声明本次评估覆盖的层级：

- `L1 文档审阅层`
- `L2 本地最小运行层`
- `L3 Linux compile-only 层`
- `L4 NPU 上板层`

没有进入对应层，就标 `未实测`，不要把环境缺失当成 PTOAS 负分。

## 默认任务模板

若用户没有指定具体 case，优先用 `test/samples/MatMul/` 作为复现模板：

```bash
python3 test/samples/MatMul/tmatmulk.py > /tmp/tmatmulk.pto
./build/tools/ptoas/ptoas /tmp/tmatmulk.pto -o /tmp/tmatmulk.cpp
```

需要无卡 compile-only 时，继续参考：

```bash
python3 test/npu_validation/scripts/generate_testcase.py \
  --input /tmp/tmatmulk.cpp \
  --run-mode npu \
  --soc-version Ascend910
```

## 重点看什么

- `Touch-Point001-005`：能不能快速找到 README、sample、compile-only、上板验证入口
- `Touch-Point008-011`：文档是否准确、版本关系是否清楚、交付件是否完整
- `Touch-Point014-016`：样例能否直接跑，sample 链路是否完整
- `Touch-Point018`：`ptoas` / `ptobc` / validation 命令是否给出了足够示例
- `Touch-Point020`：sample 或 validation 编译报错后，修复轮次是否可控
- `Touch-Point024-027`：从 `git clone` 到 `ptoas --version` 的安装与版本定位是否顺畅
- `Touch-Point028-030`：一旦失败，日志有没有足够上下文与排障建议

## 推荐证据

- `README.md`
- `docs/no_npu_compile_only_guide_zh.md`
- `test/samples/MatMul/`
- `test/samples/runop.sh`
- `test/npu_validation/scripts/generate_testcase.py`
- `test/npu_validation/scripts/run_remote_npu_validation.sh`
- `.github/workflows/ci.yml`

## 何时记 `未实测` / `N/A`

- `未实测`：当前任务没有进入对应环境层级，例如本地没有 Linux/CANN/bisheng，却要评价 compile-only 或上板
- `N/A`：该项超出 PTOAS 仓库边界，或用户本次明确不纳入
