# 7. Partial Elementwise Operations

> **Category:** Tile-local VEC partial-shape compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib partial elementwise families. These ops combine two tiles whose valid regions may differ, but whose overlap starts at `[0, 0]`.

---

## Common Syntax

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics on the overlap region |
|----|----------------------------------|
| `pto.tpartadd` | `dst = src0 + src1` |
| `pto.tpartmul` | `dst = src0 * src1` |
| `pto.tpartmax` | `dst = max(src0, src1)` |
| `pto.tpartmin` | `dst = min(src0, src1)` |

**Constraints:**

- Let `big` ∈ {`src0`, `src1`} be the operand whose valid shape equals `dst.valid_shape`, and `small` be the other operand. Exactly one operand plays each role.
- `small.valid_shape` must be a prefix-aligned sub-rectangle of `dst.valid_shape` (i.e. starting at `[0, 0]`).
- For `pto.tpartadd` and `pto.tpartmul`: outside the overlap (where only `big` covers `dst`), `dst` takes `big`'s value.
- For `pto.tpartmax` and `pto.tpartmin`: A5 templates initialize `dst` with the dtype extremum before merging the operands, so uncovered regions follow the template's pad-extremum behavior.

**Example:**

```mlir
pto.tpartadd ins(%a, %b : !pto.tile_buf<vec, 32x32xf32>,
                          !pto.tile_buf<vec, 32x32xf32, valid=16x32>)
             outs(%dst : !pto.tile_buf<vec, 32x32xf32>)
```
