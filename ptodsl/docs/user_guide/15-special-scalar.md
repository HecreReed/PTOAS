# 15. Special Scalar Memory Access

This chapter covers scalar memory operations that are useful when a kernel
needs one element at a time outside a tile transfer. It explains the two
forms of `pto.load_scalar` / `pto.store_scalar` and how they differ from the
SIMT-only `scalar.load` / `scalar.store` and `pto.ldg` / `pto.stg` operations.

## 15.1 Choosing a scalar access form

Use the operation that matches both the execution context and the memory
behavior you need:

| Operation | Execution context | Memory/type contract | Use it for |
|-----------|-------------------|----------------------|------------|
| `pto.load_scalar` / `pto.store_scalar` | Ordinary `@pto.jit` entry | One element through a typed pointer; the element type is inferred from the pointer | Scalar-pipeline access with the normal scalar-memory behavior |
| `pto.load_scalar(..., bypass_l1=True)` / `pto.store_scalar(..., bypass_l1=True)` | Ordinary `@pto.jit` entry | GM pointer with an `i8`, `i16`, `i32`, or `i64` element type | Integer GM metadata or control values that must bypass the local L1 data cache |
| `scalar.load` / `scalar.store` | `@pto.simt` helper or SIMT scope | One scalar per work-item; supports typed pointer and tile-element forms | Per-work-item scalar computation |
| `pto.ldg` / `pto.stg` | `@pto.simt` helper or SIMT scope | GM pointer, optional cache controls, and scalar or supported packed element types | SIMT GM access when cache policy or packed values matter |

`offset` is an element offset in all four surfaces. It is not a byte offset.
For example, offset `3` on an `i32` pointer addresses the fourth `i32`
element, not byte address `base + 3`.

## 15.2 Scalar-pipeline access

### `pto.load_scalar(ptr, offset=0, *, bypass_l1=False) -> ScalarType`

Loads one element from `ptr` and returns a runtime PTO scalar. The result type
is always the element type of `ptr`; no separate result-type argument is
needed.

With the default `bypass_l1=False`, `ptr` may refer to the memory space
appropriate for the normal scalar-pipeline access. With
`bypass_l1=True`, `ptr` must be a GM pointer whose element type is one of
`pto.i8`, `pto.i16`, `pto.i32`, or `pto.i64`.

### `pto.store_scalar(ptr, offset, value, *, bypass_l1=False) -> None`

Stores one scalar `value` to `ptr[offset]`. The value must be compatible with
the pointer element type. The `bypass_l1=True` form has the same GM and
integer-type restrictions as the corresponding load.

Both operations are ordinary AICore scalar-pipeline operations. They must not
be placed inside a `@pto.simt` helper or SIMT execution scope. The bypass form
only changes the local L1 data-cache behavior; it does not provide atomicity,
memory ordering, synchronization, or an L2 cache policy.

### Example: normal and L1-bypassing scalar access

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"special_scalar_access_probe","compile":{}} -->
```python
from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def special_scalar_access_probe(
    src: pto.ptr(pto.i32, "gm"),
    dst: pto.ptr(pto.i32, "gm"),
):
    # Normal scalar-pipeline access.
    value = pto.load_scalar(src, 1)
    pto.store_scalar(dst, 1, value)

    # Integer GM access that bypasses the local L1 data cache.
    metadata = pto.load_scalar(src, 3, bypass_l1=True)
    pto.store_scalar(dst, 3, metadata, bypass_l1=True)
```

The two calls in the second pair still use the PTODSL
`load_scalar`/`store_scalar` names. The `bypass_l1` flag selects their
L1-bypassing memory behavior while preserving the scalar-pipeline API.

## 15.3 Constraints and diagnostics

The following combinations are rejected during compilation:

- `bypass_l1` is not a Python `bool`.
- `bypass_l1=True` is used with a non-GM pointer.
- `bypass_l1=True` is used with a floating-point or unsupported pointer
  element type.
- Either bypass operation is placed in a SIMT execution scope.
- A store value does not match the destination pointer's element type.

Use `pto.castptr` only when you genuinely have an integer address and know its
memory space and element type. For ordinary GM tensors, pass the typed pointer
from the kernel entry directly.

## 15.4 SIMT scalar and GM access

`scalar.load` and `scalar.store` are intended for SIMT code. They execute one
logical access per work-item and can use a tile element such as
`scalar.load(tile[row, col])` or an explicit typed pointer and element offset.
They are the right choice when the surrounding computation is already inside
`@pto.simt`.

`pto.ldg` and `pto.stg` are the SIMT GM-specific forms. They provide explicit
L1/L2 cache-policy arguments and support the broader scalar and packed value
types accepted by the SIMT GM interface. They must remain inside a SIMT
execution scope and should be chosen when those cache controls or packed
values are part of the kernel contract.

For a plain integer GM metadata access in an ordinary AICore entry, prefer
`pto.load_scalar` / `pto.store_scalar` with `bypass_l1=True`.
