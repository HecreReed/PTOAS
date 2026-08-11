# 16A. Cube Matmul Raw Ops (MAT)

> **Category:** Cube unit raw ops — low-level data movement and matrix-side configuration
> **Audience:** Backend / lowering / bridge-op implementers

---

## Scope

This document lists the raw cube matmul ops used by wrapper interfaces in
`16-cube-matmul.md`.

If you are writing user-facing VPTO kernels, prefer wrapper ops such as
`pto.mte_gm_l1`, `pto.mte_l1_l0a`, `pto.mte_l1_l0b`, `pto.mte_l0c_*`, and
`pto.mte_gm_l1_frac`.

---

## Raw Staging / Load Ops

### `pto.copy_gm_to_cbuf`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf %src, %dst, %n_burst, %len_burst, %src_stride, %dst_stride
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64
```
- **semantics:** Copy matrix tile data from GM to L1 (`cbuf`).

### `pto.load_cbuf_to_ca`

- **syntax:**
```mlir
pto.load_cbuf_to_ca %src, %dst, %m_start, %k_start, %m_step, %k_step, %src_stride, %dst_stride
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0A.

### `pto.load_cbuf_to_cb`

- **syntax:**
```mlir
pto.load_cbuf_to_cb %src, %dst, %m_start, %k_start, %m_step, %k_step, %src_stride, %dst_stride
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0B.

### `pto.load_cbuf_to_ca_mx`

- **syntax:**
```mlir
pto.load_cbuf_to_ca_mx %src, %dst, %m, %k
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0A using MX path.

### `pto.load_cbuf_to_cb_mx`

- **syntax:**
```mlir
pto.load_cbuf_to_cb_mx %src, %dst, %x_start_position, %y_start_position, %x_step, %y_step, %src_stride, %dst_stride
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64, i64, i64, i64, i64
```
- **semantics:** Load L1 (`cbuf`) tile to L0B using MX path with explicit hardware control fields.

---

## Raw L0C Writeback / Move Ops

### `pto.copy_matrix_cc_to_gm`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_gm %src, %dst, %xm, %xt
  : !pto.ptr<T, l0c>, !pto.ptr<T, gm>, i64, i64
```
- **semantics:** Write L0C (`acc`) tile back to GM.

### `pto.copy_matrix_cc_to_cbuf`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_cbuf %src, %dst, %config0, %config1
  : !pto.ptr<T, l0c>, !pto.ptr<T, l1>, i64, i64
```
- **semantics:** Move L0C (`acc`) tile to L1 (`cbuf`).

### `pto.copy_matrix_cc_to_ub`

- **syntax:**
```mlir
pto.copy_matrix_cc_to_ub %src, %dst, %config0, %config1
  : !pto.ptr<T, l0c>, !pto.ptr<T, ub>, i64, i64
```
- **semantics:** Move L0C (`acc`) tile to UB.

---

## Raw CBUF Outbound Ops

### `pto.copy_cbuf_to_bt`

- **syntax:**
```mlir
pto.copy_cbuf_to_bt %src, %dst, %len_burst, %n_burst, %src_gap, %dst_gap
  : !pto.ptr<T, l1>, !pto.ptr<U, bt>, i64, i64, i64, i64
```
- **semantics:** Move L1 (`cbuf`) data to BT buffer.

### `pto.copy_cbuf_to_fbuf`

- **syntax:**
```mlir
pto.copy_cbuf_to_fbuf %src, %dst, %n_burst, %len_burst, %src_gap, %dst_gap
  : !pto.ptr<T, l1>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Move L1 (`cbuf`) data to FB-related destination path.

### `pto.copy_gm_to_cbuf_multi_nd2nz`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf_multi_nd2nz %src, %dst, %sid, %loop1_src_stride, %l2_cache_ctrl, %n_value, %d_value, %loop4_src_stride, %smallc0_en
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64, i64, i64, i1
```
- **semantics:** Multi-fractal `ND2NZ` staging from GM to L1 (`cbuf`).

### `pto.copy_gm_to_cbuf_multi_dn2nz`

- **syntax:**
```mlir
pto.copy_gm_to_cbuf_multi_dn2nz %src, %dst, %sid, %loop1_src_stride, %l2_cache_ctrl, %n_value, %d_value, %loop4_src_stride, %smallc0_en
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64, i64, i64, i1
```
- **semantics:** Multi-fractal `DN2NZ` staging from GM to L1 (`cbuf`).

---

## Raw vs Wrapper Mapping

| Wrapper op | Typical raw op(s) |
|---|---|
| `pto.mte_gm_l1` | `pto.copy_gm_to_cbuf` + loop setup |
| `pto.mte_l1_l0a` | `pto.load_cbuf_to_ca` |
| `pto.mte_l1_l0b` | `pto.load_cbuf_to_cb` |
| `pto.mte_l1_l0a_mx` | `pto.load_cbuf_to_ca_mx` |
| `pto.mte_l1_l0b_mx` | `pto.load_cbuf_to_cb_mx` |
| `pto.mte_gm_l1_frac` | `pto.copy_gm_to_cbuf_multi_nd2nz` / `pto.copy_gm_to_cbuf_multi_dn2nz` + config setup |
| `pto.mte_l1_bt` | `pto.copy_cbuf_to_bt` |
| `pto.mte_l0c_l1` | `pto.copy_matrix_cc_to_cbuf` (+ related config) |
| `pto.mte_l0c_gm` | `pto.copy_matrix_cc_to_gm` (+ related config) |
| `pto.mte_l0c_ub` | `pto.copy_matrix_cc_to_ub` (+ related config) |
