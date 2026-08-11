# PTODSL Local C2V TPush/Tpop Sample

This sample mirrors the local FIFO shape in `test/samples/TPushTPop/test1` with
PTODSL pipe surface APIs.

Compile-only:

```bash
python3 test/samples/TPushTPop/ptodsl/local_c2v/kernel.py --emit-mlir
```

PTOAS frontend verification:

```bash
python3 test/samples/TPushTPop/ptodsl/local_c2v/kernel.py --verify-ptoas
```

The sample generates separate Cube and Vector PTODSL kernels, merges the
generated functions into one MLIR module, and verifies that PTOAS lowers the
local FIFO path through `TPUSH` / `TPOP` / `TFREE`.
