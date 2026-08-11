# 9. Type Conversion

> **Category:** Element-wise type conversion
> **Pipeline:** PIPE_V

This chapter documents the element-wise tile conversion instruction `pto.tcvt` and the rounding modes it uses.

---

## `RoundMode`

Rounding modes for `pto.tcvt`.

| Value | Int | Description |
|-------|-----|-------------|
| `NONE` | 0 | No rounding. |
| `RINT` | 1 | Round to nearest integer. |
| `ROUND` | 2 | Round `f16` away from zero. |
| `FLOOR` | 3 | Round toward negative infinity. |
| `CEIL` | 4 | Round toward positive infinity. |
| `TRUNC` | 5 | Truncate toward zero. |
| `ODD` | 6 | Round to odd. |
| `CAST_RINT` | 7 | Cast with round-to-nearest (default). |

**Attribute syntax:** `#pto<round_mode FLOOR>`

---

## `pto.tcvt`

- **syntax:**
```mlir
pto.tcvt ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
         {rmode = #pto<round_mode ...>}
```
- **semantics:** `dst[i, j] = cast(src[i, j], rmode)` element-wise.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Destination tile (different element type). |
| `rmode` | `RoundModeAttr` | Default `CAST_RINT`. |

**Constraints:**

- `src`/`dst` must be shape/valid-region compatible.
- This reference does not define extra legality rules for the `(src, dst)` type pair. **Undefined behavior** on conversion pairs not supported by the target hardware; consult the A2/A3 and A5 hardware specs for legal pairs.

**Example:**

```mlir
pto.tcvt ins(%src : !pto.tile_buf<vec, 16x16xf32>)
         outs(%dst : !pto.tile_buf<vec, 16x16xf16>)
         {rmode = #pto<round_mode FLOOR>}
```
