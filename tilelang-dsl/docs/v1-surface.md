# TileLang DSL v1 Surface

## Scope

This document records the implemented v1 boundary for the standalone
`tilelang_dsl` package introduced by
`add-tilelang-dsl-core-foundation`.

It covers:
- package entrypoints
- supported `@vkernel` decorator metadata
- parameter typing rules
- Tile specialization requirements
- current frontend diagnostics boundary
- deferred features that belong to follow-up changes

It does not define:
- DSL to VPTO lowering details
- matcher and priority semantics
- advanced vector-family surface
- implicit vecscope inference

For implemented lowering details, examples, and `verify()` behavior, see
`tilelang-dsl/docs/v1-lowering.md`.
For migration from the original v1 core boundary to the current matcher and
advanced-surface package capabilities, see
`tilelang-dsl/docs/matcher-and-advanced-surface-migration.md`.

## Source Of Truth

TileLang DSL v1 source of truth lives under:
- `tilelang-dsl/python/tilelang_dsl/`
- `tilelang-dsl/tests/`
- `tilelang-dsl/examples/`
- `tilelang-dsl/docs/`

`python/pto/dialects/pto.py` is not the source of truth for TileLang DSL v1.
That file still exists for PTO dialect bindings and the legacy experimental VPTO
Python DSL surface. Root-level wiring into build, install, and test is allowed,
but new TileLang DSL core behavior must land under `tilelang-dsl/`.

## Package Entry

Examples and tests should import the standalone package:

```python
import tilelang_dsl as pto
```

The package currently exports:
- `vkernel`
- `ckernel`
- `VKernelDescriptor`
- `BoundKernelParameter`
- `MaterializedMLIRModule`
- `KernelRegistry`
- `select_kernel`
- `TileLangFrontendError`
- `TensorView`
- `PartitionTensorView`
- `Tile`
- `VRegType`
- `MaskType`
- scalar dtypes such as `f16`, `bf16`, `f32`, `i8`, `i16`, `i32`, `i64`
- type helpers such as `vreg(...)`, `ptr(...)`, `mask_b8`, `mask_b16`, `mask_b32`, `MemorySpace`, `TileConfig`, `TileSpecialization`, `ViewConfig`, `ViewLayout`

The package does not expose a DSL-level `pto.memref(...)` constructor. MemRef
only appears in generated/lowered IR, not in the public authoring type surface.

## v1 Decorator Surface

The supported v1 decorator surface is:

```python
@pto.vkernel(
    target="a5",
    op="some_op_name",
    dtypes=[(pto.f32, pto.f16, pto.i32)],
    name="optional_name",
    verify=True,
)
def kernel(...):
    ...
```

Current rules:
- `target` only accepts `"a5"`
- `op` is required and must be a non-empty string
- `dtypes` must contain exactly one monomorphic signature tuple
- `name` is optional and defaults to the Python function name
- `verify` is optional and must be a bool

The descriptor keeps these metadata fields:
- `target`
- `op`
- `dtypes`
- `name`
- `verify`

In current package head, descriptor selection also supports:
- `ops=[...]` multi-op matcher groups
- `constraints=[...]` callable selection predicates
- `priority`
- `templates={...}` slot substitution metadata

## Parameter Typing

v1 accepts these parameter categories:
- bare `TensorView`
- bare `Tile`
- scalar annotations such as `pto.i32`, `pto.f16`, `pto.f32`, `pto.AnyType`, or `pto.TypeVar("T")`

Binding rules:
- the single `dtypes` signature binds parameter element types positionally
- `TensorView` parameters get their element dtype from the same position in
  `dtypes`
- `Tile` parameters get their element dtype from the same position in `dtypes`
- scalar parameters must use a TileLang scalar-style annotation
- scalar annotations may be concrete scalar dtypes, wildcard dtypes, or
  `TypeVar(...)`
- concrete scalar annotations must exactly match the dtype at the same position
  in `dtypes`
- wildcard scalar annotations must accept the dtype at the same position in
  `dtypes`
- `TypeVar(...)` scalar annotations bind to the selected dtype at the same
  position in `dtypes`

Example:

```python
@pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.bf16, pto.i32)])
def kernel(inp: pto.TensorView, tmp: pto.Tile, scale: pto.i32):
    return None
```

In this example:
- `inp` binds to `f32`
- `tmp` binds to `bf16`
- `scale` binds to `i32`

## Tile Specialization

Bare `Tile` parameters are incomplete until descriptor-level specialization is
provided.

The only supported completion path is:

```python
specialized = descriptor.specialize(
    tmp=pto.TileSpecialization(
        shape=(16, 32),
        memory_space=pto.MemorySpace.UB,
        config=pto.TileConfig.from_mapping({"layout": "row_major"}),
    )
)
```

Current v1 Tile profile rules:
- Tile physical shape must be static
- Tile dimensions must be positive integers
- Tile rank must be 1D or 2D
- Tile memory space must be `MemorySpace.UB`
- `config` may be omitted, provided as `TileConfig`, or built from a dict

## View Config Surface

`TensorView` and `PartitionTensorView` expose a view-only config object during
matcher constraint evaluation.

Current query surface:
- `view.config.layout`

Current layout enum values:
- `pto.ViewLayout.ND`
- `pto.ViewLayout.DN`
- `pto.ViewLayout.NZ`
- `pto.ViewLayout.MX_A_ZZ`
- `pto.ViewLayout.MX_B_NN`

Rules:
- `view.config` is a `ViewConfig`, not a `TileConfig`
- `layout` is optional; when the source query has no explicit layout metadata,
  `view.config.layout` is `None`
- unknown or absent layout metadata must not be treated as implicit `ND`
- this surface is intended for descriptor selection constraints, especially
  when different templates should match different GM view layouts

Example:

```python
@pto.vkernel(
    target="a5",
    op="pto.tload",
    dtypes=[(pto.f32, pto.f32)],
    constraints=[lambda src: src.config.layout == pto.ViewLayout.DN],
)
def template_tload_dn(src: pto.TensorView, dst: pto.Tile):
    return None
```

Before all bare `Tile` parameters are specialized, the descriptor must reject:
- `mlir_text()`
- `mlir_module()`
- `verify()`
- `emit(path)`

## Materialization API

After all bare `Tile` parameters are specialized, the descriptor exposes:
- `mlir_text()`
- `mlir_module()`
- `verify()`
- `emit(path)`

At this stage of the workflow, these APIs provide a stable descriptor/materialization
surface for the new package. They do not yet define the final TileLang DSL to
VPTO lowering behavior; that work belongs to
`add-tilelang-dsl-authoring-vpto-lowering`.

## Frontend Diagnostics

The v1 frontend fails fast for:
- unsupported decorator matcher features
- unsupported Python syntax
- arbitrary external calls
- unsupported `pto.*` op surface
- missing Tile specialization
- dynamic physical Tile shape
- illegal Tile profile

Diagnostics are frontend errors, not deferred verifier failures. When source is
available, errors include file, line, and column information.

## Minimal Validation

The following commands are the minimal validation set for
`add-tilelang-dsl-core-foundation`:

```bash
cmake --build build --target TileLangDSLPackage
python3 -c "import sys; sys.path.insert(0, 'build/python'); import tilelang_dsl; print(tilelang_dsl.__file__)"
ctest --test-dir build -R tilelang_dsl_import --output-on-failure
ctest --test-dir build -R tilelang_dsl_unittest --output-on-failure
```

What these commands confirm:
- the standalone `tilelang_dsl` package is staged into `build/python/`
- Python can import the staged package directly
- the dedicated import smoke test passes
- the focused unittest suite passes for descriptor API, specialization, and
  diagnostics coverage

For a direct source-location diagnostics smoke, run:

```bash
tmp=$(mktemp /tmp/tilelang_dsl_diag_XXXX.py)
cat > "$tmp" <<'PY'
import tilelang_dsl as pto

try:
    @pto.vkernel(op="x", dtypes=[(pto.f32,)])
    def kernel(x: pto.TensorView):
        while True:
            return None
except pto.TileLangFrontendError as exc:
    print(exc)
PY
PYTHONPATH=build/python python3 "$tmp"
rm -f "$tmp"
```

Expected output shape:

```text
/tmp/tilelang_dsl_diag_XXXX.py:6:5: unsupported Python syntax `while` in TileLang DSL v1
```

This confirms diagnostics are emitted against the authored DSL source file
rather than an internal lowering location.

## Historical Deferred Features

The following were intentionally out of scope for the original v1 core boundary
and were assigned to follow-up changes:
- multiple `dtypes` signatures
- `constraints`
- `priority`
- `AnyFloat`, `AnyInt`, `AnyType`, `AnyMask`
- `TypeVar`
- matcher registry and deterministic selection
- implicit vecscope inference
- raw pointer authoring surface
- advanced vector-family support

Matcher-related extensions are deferred to
`extend-tilelang-dsl-matcher-and-advanced-surface`.
That follow-up capability is now implemented in the current package head; use
`tilelang-dsl/docs/matcher-and-advanced-surface-migration.md` for the updated
surface boundary instead of reading the list above as a statement about current
head behavior.
