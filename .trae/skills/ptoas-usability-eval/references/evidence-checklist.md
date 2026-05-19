# 证据清单

## 固定证据入口

| 路径 | 主要用途 | 对应场景 | 层级 |
| --- | --- | --- | --- |
| `README.md` | 官方构建、环境变量、CLI、Python 绑定、sample 运行、compile-only/上板验证主入口 | `01`, `02`, `04`, `05`, `06` | `L1-L4` |
| `docs/no_npu_compile_only_guide_zh.md` | 无卡 compile-only 流程、批量验证流程、`pto-isa`/CANN 依赖说明 | `01`, `02`, `04`, `05`, `06` | `L1`, `L3` |
| `docs/PTO_IR_manual.md` | IR 层级、tile/view/valid-shape、layout、dynamic shape、Level-2/3 语义 | `02`, `04`, `05`, `06` | `L1-L4` |
| `test/samples/runop.sh` | 批量样例生成、`ptoas`/`ptobc` 运行、A3/A5 默认参数策略 | `01`, `02`, `04`, `05`, `06` | `L1-L4` |
| `test/npu_validation/scripts/generate_testcase.py` | 从 `*-pto.cpp` 生成验证工程，观察 golden/compare/兼容层处理 | `01`, `02`, `04`, `05`, `06` | `L1`, `L3`, `L4` |
| `test/npu_validation/scripts/run_remote_npu_validation.sh` | compile-only / sim / npu 运行链路、日志格式、设备与 `pto-isa` 检查 | `01`, `02`, `04`, `05`, `06` | `L1`, `L3`, `L4` |
| `test/samples/PyPTOIRParser/README.md` | 来自 pypto `ir_parser` 的 vendored `.pto` 快照说明 | `02` | `L1` |
| `test/samples/MatMul/` | README 直接引用的基准样例，适合作为 `01` 默认复现模板 | `01`, `04`, `05` | `L1-L4` |
| `test/samples/FlashAttention/` | 特定 shape 性能样例 | `05` | `L1-L4` |
| `test/samples/GQA/` | 特定 shape / attention 相关样例 | `05` | `L1-L4` |
| `test/samples/FFN/` | 特定 shape / 算子组合样例 | `05` | `L1-L4` |
| `test/samples/SetValidShape/` | dynamic/valid-shape 相关样例 | `06` | `L1-L4` |
| `test/samples/LayoutInference/` | layout 推断相关样例 | `06` | `L1-L4` |
| `test/samples/Partition5D/` | 多维 partition / shape 泛化相关样例 | `02`, `06` | `L1-L4` |
| `test/samples/planmemory/` | alias/planmemory/shape 相关样例 | `06` | `L1-L4` |
| `.github/workflows/ci.yml` | CI 中的 LLVM/PTOAS 构建、lit、sample test、remote validation 参考配置 | `01`, `02`, `04`, `05`, `06` | `L1`, `L3`, `L4` |
| `.github/ISSUE_TEMPLATE/performance_issue.yml` | 性能问题受理模板，可用来评估性能数据/复现要求的完备性 | `05`, `06` | `L1` |

说明：
- 不要把当前分支不存在的样例 README 当成固定证据源。
- 对 `02/05/06`，没有“前后对照基线”时，不要硬算迁移完备度或性能提升幅度。

## 推荐检索顺序

1. `README.md`
2. `docs/no_npu_compile_only_guide_zh.md`
3. `docs/PTO_IR_manual.md`
4. `test/samples/MatMul/` 或用户指定样例目录
5. `test/samples/PyPTOIRParser/`, `FlashAttention/`, `GQA/`, `FFN/`, `SetValidShape/`, `LayoutInference/`, `Partition5D/`, `planmemory/`
6. `test/samples/runop.sh`
7. `test/npu_validation/scripts/*.py` / `*.sh`
8. `.github/workflows/ci.yml`
9. `.github/ISSUE_TEMPLATE/performance_issue.yml`

## 推荐检索命令

```bash
rg -n "构建|运行测试|compile-only|runop|generate_testcase|run_remote_npu_validation|level3" README.md docs test .github
rg -n "valid_shape|layout|partition|reshape|dynamic shape|Level-2|Level-3" docs/PTO_IR_manual.md docs test
rg -n "FlashAttention|GQA|FFN|MatMul|SetValidShape|LayoutInference|Partition5D|planmemory" test .github
rg --files test/samples
find test/samples -maxdepth 2 -type f \( -name '*.py' -o -name '*.pto' -o -name 'README.md' \)
```

## 迁移 / 性能场景的补充记录项

如果在评 `02/05/06`，额外记录：
- 是否存在迁移前/迁移后对照物
- 是否存在性能 baseline
- baseline 来源路径
- 是否需要 NPU 实机
- 当前停在哪一层
- 哪些分数是实测，哪些是文档侧支撑分

## 记录要求

每个评分项至少要落这些证据字段：

- `证据路径`
- `检索/执行命令`
- `检索轮次`
- `文档跳转次数`
- `评估层级`
- `耗时`
- `结果`
- `评分`
- `备注`

## 默认样例

若用户没有指定具体算子或样例，优先使用：

- `test/samples/MatMul/tmatmulk.py`
- `test/samples/MatMul/tmatmulk.pto`
- `test/samples/Addc/addc.py`
- `test/samples/PyPTOIRParser/`
- `test/samples/FlashAttention/`
- `test/samples/SetValidShape/`

理由：这些路径要么被 `README.md` 直接引用，要么与迁移/性能/shape 泛化评估强相关，且当前主线可稳定找到。
