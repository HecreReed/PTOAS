# 8. Bitwise and Shift Operations

> **Category:** Tile-local integer VEC compute
> **Pipeline:** PIPE_V

This chapter documents the integer-only TileLib bitwise and shift families.

---

## 8.1 Unary Bitwise NOT: `pto.tnot`

- **syntax:**
```mlir
pto.tnot ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst = ~src`.

**Constraints:**

- Tile element types must be integer types.
- `src` and `dst` must have the same valid region.

**Example:**

```mlir
pto.tnot ins(%a : !pto.tile_buf<vec, 16x16xi32>)
         outs(%c : !pto.tile_buf<vec, 16x16xi32>)
```

---

## 8.2 Binary Tile-Tile Bitwise and Shift Families

Tile-tile bitwise and shift families:

| Op | Semantics |
|----|-----------|
| `pto.tand` | `dst = src0 & src1` |
| `pto.tor` | `dst = src0 \| src1` |
| `pto.txor` | `dst = src0 ^ src1` |
| `pto.tshl` | `dst = src0 << src1` |
| `pto.tshr` | `dst = src0 >> src1` |

### Common Forms

For `pto.tand`, `pto.tor`, `pto.tshl`, and `pto.tshr`:

```mlir
pto.<op> ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

`pto.txor` carries an extra scratch tile `%tmp` for A2/A3 interface compatibility (see [§1.7](01-tile-overview.md#17-scratch-operands-and-a2a3-compatibility)):

```mlir
pto.txor ins(%src0, %src1, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>,
             !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

**Constraints:**

- Tile element types must be integer types.
- `src0`, `src1`, and `dst` must have the same valid region.

**Example:**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<vec, 16x16xi32>, !pto.tile_buf<vec, 16x16xi32>)
         outs(%c : !pto.tile_buf<vec, 16x16xi32>)
```

---

## 8.3 Tile-Scalar Bitwise and Shift Families

Tile-scalar bitwise and shift families:

| Op | Semantics |
|----|-----------|
| `pto.tands` | `dst = src & scalar` |
| `pto.tors` | `dst = src \| scalar` |
| `pto.txors` | `dst = src ^ scalar` |
| `pto.tshls` | `dst = src << scalar` |
| `pto.tshrs` | `dst = src >> scalar` |

### Common Forms

For `pto.tands`, `pto.tors`, `pto.tshls`, and `pto.tshrs`:

```mlir
pto.<op> ins(%src, %scalar : !pto.tile_buf<...>, <integer_scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
```

`pto.txors` carries an extra scratch tile `%tmp` for A2/A3 interface compatibility:

```mlir
pto.txors ins(%src, %scalar, %tmp : !pto.tile_buf<...>, <integer_scalar_type>,
              !pto.tile_buf<...>)
          outs(%dst : !pto.tile_buf<...>)
```

**Constraints:**

- Tile element types must be integer types.
- `src` and `dst` must have the same valid region.
- The scalar operand must be an integer-compatible shift / bitwise scalar.

**Example:**

```mlir
pto.tands ins(%a, %s : !pto.tile_buf<vec, 16x16xi32>, i32)
          outs(%dst : !pto.tile_buf<vec, 16x16xi32>)
```
