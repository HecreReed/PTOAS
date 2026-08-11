# 12. Fill and Padding Operations

> **Category:** Tile-local fill, pad, and expansion materialization
> **Pipeline:** PIPE_V

This chapter documents the TileLib fill / padding families. These ops preserve or materialize valid data and then synthesize the remaining destination region from the destination tile's padding policy.

The destination tile's `pad` / `pad_value` configuration determines which value is written into the synthesized padding or expansion region.

---

## 12.1 `pto.tfillpad`

- **syntax:**
```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>)
             outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** copy valid data from `src` into `dst`, then fill the remaining destination region according to `dst`'s pad policy.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Destination tile carrying the pad configuration. |

**Constraints:**

- Source and destination element types must be compatible.
- The destination tile must carry a meaningful pad configuration.
- This family is VEC-oriented.

**Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<vec, 8x64xf32, valid=?x?>)
             outs(%dst : !pto.tile_buf<vec, 8x64xf32, pad=1>)
```

---

## 12.2 `pto.tfillpad_expand`

- **syntax:**
```mlir
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>)
                    outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** copy valid data from `src` into `dst`, then fill row/column expansion according to `dst`'s pad policy when the destination valid region or backing shape is larger than the source.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Larger destination tile carrying the pad configuration. |

**Constraints:**

- `dst` may be larger than `src` in valid region or physical shape.
- The fill value is derived from `dst.pad_value`.
- A unified VEC-oriented template handles the supported element families.

**Example:**

```mlir
pto.tfillpad_expand ins(%src : !pto.tile_buf<vec, 4x32xf32>)
                    outs(%dst : !pto.tile_buf<vec, 8x64xf32, pad=1>)
```

---

## 12.3 `pto.tfillpad_inplace`

- **syntax:**
```mlir
pto.tfillpad_inplace ins(%src : !pto.tile_buf<...>)
                     outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** update the padding / expansion region of an already materialized tile without a separate copy-in phase.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer, typically aliasing the same physical tile. |

**Constraints:**

- PTOAS exposes `pto.tfillpad_inplace` as a dedicated Tile op.
- In typical use, `src` and `dst` refer to the same underlying tile buffer.
- The fill value is derived from `dst.pad_value`.

**Example:**

```mlir
pto.tfillpad_inplace ins(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
                     outs(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
```
