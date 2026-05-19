# 证据清单

## 固定证据入口

| 路径 | 主要用途 | 对应场景 | 层级 |
| --- | --- | --- | --- |
| `README.md` | 官方构建、环境变量、CLI、Python 绑定、sample 运行、上板验证主入口 | `01`, `04-子集` | `L1-L4` |
| `docs/no_npu_compile_only_guide_zh.md` | 无卡 compile-only 流程、批量验证流程、`pto-isa`/CANN 依赖说明 | `01`, `04-子集` | `L1`, `L3` |
| `test/samples/runop.sh` | 批量样例生成、`ptoas`/`ptobc` 运行、A3/A5 默认参数策略 | `01`, `04-子集` | `L1-L4` |
| `test/npu_validation/scripts/generate_testcase.py` | 从 `*-pto.cpp` 生成验证工程，观察 golden/compare/兼容层处理 | `01`, `04-子集` | `L1`, `L3`, `L4` |
| `test/npu_validation/scripts/run_remote_npu_validation.sh` | compile-only / sim / npu 运行链路、日志格式、设备与 `pto-isa` 检查 | `01`, `04-子集` | `L1`, `L3`, `L4` |
| `test/samples/MatMul/` | README 直接引用的基准样例，适合作为 `01` 默认复现模板 | `01`, `04-子集` | `L1-L4` |
| `test/samples/PyPTOIRParser/README.md` | vendored `.pto` 样例来源与稳定性说明 | `04-子集` | `L1` |
| `.github/workflows/ci.yml` | CI 中的 LLVM/PTOAS 构建、lit、sample test、remote validation 参考配置 | `01`, `04-子集` | `L1`, `L3`, `L4` |

说明：不要把当前分支不存在的样例 README 当成固定证据源。若某个样例目录确实存在 README，再按需补读。

## 推荐检索顺序

1. `README.md`
2. `docs/no_npu_compile_only_guide_zh.md`
3. `test/samples/MatMul/`
4. `test/samples/runop.sh`
5. `test/npu_validation/scripts/*.py` / `*.sh`
6. `.github/workflows/ci.yml`
7. 按具体样例目录补读 `test/samples/*/README.md`

## 推荐检索命令

```bash
rg -n "构建|运行测试|上板验证|compile-only|generate_testcase|run_remote_npu_validation" README.md docs test .github
rg --files test/samples
find test/samples -maxdepth 2 -type f \( -name '*.py' -o -name '*.pto' -o -name 'README.md' \)
```

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

理由：这些路径要么被 `README.md` 直接引用，要么在当前主线可稳定找到。
