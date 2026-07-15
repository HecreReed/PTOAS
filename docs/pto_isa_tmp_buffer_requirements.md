# PTO-ISA `tmp` Buffer Requirements by Target

This document is the allocation reference for PTO-ISA interfaces that expose a
`tmp` tile and are used or tracked by PTOAS. It records the minimum temporary
storage contract separately for A2/A3 and A5.

The source baseline is `cann/pto-isa` `master` commit
`7af803bc4056af8b39a55751ac2f4b75cdb47fbd` (2026-07-15). Follow the linked
PTO-ISA pages when a later runtime release changes a formula.

`Not used (compatibility only)` means that the target implementation does not
read or write scratch bytes through `tmp`. The PTO IR operation may still carry
the operand, and its generic tile type, memory-space, and layout checks still
apply. It does not mean that the operand can be omitted unless the operation's
syntax marks it optional.

This page covers on-chip tile scratch. `TPRINT` is not included because its
`tmp` argument for `mat`/`acc` printing is global-memory staging rather than a
tile scratch buffer.

## Notation

- `R`, `C`: valid row and column counts of the operation result unless stated
  otherwise.
- `E`: element size in bytes.
- `B = 32`: vector block size in bytes.
- `align_n(x) = ceil(x / n) * n`.
- `elemPerBlock = 32 / E` and `elemPerRepeat = 256 / E`.
- Tile allocation uses the physical tile extent. A `valid_shape` requirement
  alone does not reserve storage beyond `rows * cols * E` bytes.

## Target summary

| Interface | A2/A3 `tmp` requirement | A5 `tmp` requirement | PTO-ISA reference |
|---|---|---|---|
| `TTRANS` / `pto.ttrans` | Target/mode formula; see below | Not used (compatibility only) | [TTRANS](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TTRANS_zh.md#约束) |
| `TCVT` / `pto.tcvt` | Required only for non-saturating `f32 -> i16`, `f16 -> i16`, and `f16 -> i8`; see below | Not used (compatibility only) | [TCVT](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TCVT_zh.md#约束) |
| `TGATHER` index form / `pto.tgather` | Same index extent and type; see below | Not used (compatibility only) | [TGATHER](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TGATHER.md#temporary-tile) |
| `TGATHER` comparison form / `pto.tgather` | Three-region byte formula; see below | Not used (compatibility only) | [TGATHER](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TGATHER.md#temporary-tile) |
| `TCI` / `pto.tci` | 768 bytes for b32 output; 1792 bytes for b16 output | Not used (compatibility only) | [TCI](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TCI_zh.md#约束) |
| `TSORT32` / `pto.tsort32` four-operand form | Tail formula with an 8160-element threshold | Tail formula with an 8160-byte threshold | [TSORT32](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TSORT32.md#tmp-size-equation-4-arg) |
| INT8 `TQUANT` / `pto.tquant` | FP32 tile with the same physical extent as `src`: `4 * src.rows * src.cols` bytes | Not used (compatibility only) | [TQUANT](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TQUANT.md#int8) |
| `TROWEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}` / corresponding `pto.trowexpand*` tmp form | Row-broadcast formula; see below | Not used (compatibility only) | [TROWEXPANDADD](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDADD_zh.md#临时-tile) |
| A5 `TMOV` ND-to-ZZ form | Not applicable | Index-buffer formula; DN-to-ZZ accepts but does not use `tmp` | [TMOV](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TMOV.md#nd--zz-exponents--tmp-size-derivation) |
| `TXOR` / `pto.txor` | Same element type and valid extent as `dst` | Not used (compatibility only) | [TXOR](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TXOR_zh.md#临时空间) |
| `TXORS` / `pto.txors` | Same element type and valid extent as `dst` | Not used (compatibility only) | [TXORS](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TXORS_zh.md#临时空间) |
| `TPRELU` / `pto.tprelu` | Packed-predicate workspace plus one extra valid row; see below | Not used (compatibility only) | [TPRELU](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TPRELU_zh.md#临时空间) |
| `TMRGSORT` multi-list form / `pto.tmrgsort` format 2 | One row; columns equal the sum of input physical columns | Same as A2/A3 | [TMRGSORT](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TMRGSORT_zh.md#临时空间) |
| `TROWPROD` / `pto.trowprod` | One 32-byte block | Not used (compatibility only) | [TROWPROD](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWPROD_zh.md#临时空间) |
| `TCOLSUM` binary form / `pto.tcolsum` with `isBinary` | At least `ceil(R / 2)` rows and `C` columns | Same as A2/A3 | [TCOLSUM](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TCOLSUM_zh.md#临时空间) |
| `TCOLARGMAX`, `TCOLARGMIN` / corresponding `pto.*` | Type/mode-dependent one-row stride; see below | Not used (compatibility only) | [TCOLARGMAX](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TCOLARGMAX_zh.md), [TCOLARGMIN](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TCOLARGMIN_zh.md) |
| `TROWARGMAX`, `TROWARGMIN` / corresponding `pto.*` | Width- and mode-dependent per-row stride; see below | Not used (compatibility only) | [TROWARGMAX](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWARGMAX_zh.md), [TROWARGMIN](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWARGMIN_zh.md) |
| `TROWSUM`, `TROWMAX`, `TROWMIN` / corresponding `pto.*` | Integer path: one 32-byte block; floating path: source-shaped allocation is the safe bound | Not used (compatibility only) | [TROWSUM](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWSUM_zh.md#临时空间), [TROWMAX](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWMAX_zh.md#临时空间), [TROWMIN](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWMIN_zh.md#临时空间) |
| `TSEL` / `pto.tsel` | 16 logical bytes; reserve at least one 32-byte-aligned block | Not used (compatibility only) | [TSEL](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TSEL_zh.md#临时空间) |
| `TSELS` / `pto.tsels` | At least one source-typed element; reserve at least one 32-byte-aligned block | Not used (compatibility only) | [TSELS](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TSELS_zh.md#临时空间) |
| High-precision `TRSQRT` / `pto.trsqrt` | One 32-byte block | Not used (compatibility only) | [TRSQRT](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TRSQRT_zh.md#临时空间) |
| Floating `TPOW`, `TPOWS` / corresponding `pto.*` | Same type and valid extent as `dst`; integer paths do not use `tmp` | Not used (compatibility only) | [TPOW](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TPOW_zh.md#临时空间), [TPOWS](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TPOWS_zh.md#临时空间) |
| `TREM` / `pto.trem` | At least two rows and `C` columns | Not used (compatibility only) | [TREM](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TREM_zh.md#约束) |
| `TREMS` / `pto.trems` | At least one row and `C` columns | Not used (compatibility only) | [TREMS](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TREMS_zh.md#约束) |
| `TADDDEQRELU` (no PTO IR op currently) | INT32 tile with the same valid extent as `dst`: `4RC` bytes | Not used (compatibility only) | [TADDDEQRELU](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TADDDEQRELU_zh.md#临时空间) |

The current `TSORT32` four-operand implementation is an important exception to
the usual A5 compatibility rule: A5 accesses `tmp` when `C` has a partial
32-element tail. Do not size it as an unused placeholder.

## Detailed sizing rules

### `TTRANS`

For A2/A3 2-D transpose, let `rowStride = 32` for 8-bit data and `16` for
16-/32-bit data:

```text
tmpBytes = C * align_rowStride(R) * E
```

The temporary space is only accessed when the source and destination strides
meet the aligned transpose path's requirements. The linked PTO-ISA page also
defines the `NCHW <-> NC1HWC0`, `GNCHW <-> GNC1HWC0`, and
`NCDHW -> FRACTAL_Z_3D` formulas. A5 accepts the operand but its current
transpose implementation does not access it.

### `TCVT`

The A2/A3 tmp-backed paths are the non-saturating narrowing conversions. The
tile is sized by bytes and uses a 4-byte element type. Let `SS` be the source
physical row stride in source elements and
`H = min(C, 64)` for non-empty tiles.

For `f32 -> i16`:

```text
headBytes = 4 * 64 * min(floor(C / 64), 255)
tailBytes = 0                                                   if C mod 64 = 0
tailBytes = 32 * ((min(R, 255) - 1) * SS / 8
                  + ceil((C mod 64) / 8))                       otherwise
tmpBytes  = max(headBytes, tailBytes)
```

For `f16 -> i16`:

```text
tmpBytes = 32 * ceil(H / 8)       // at most 256 bytes
```

For `f16 -> i8`:

```text
tmpBytes = max(32 * ceil(H / 8), 128 + 32 * ceil(H / 16))
                                      // at most 256 bytes
```

Other conversions do not need tmp storage. A5 does not access `tmp` for these
forms.

### `TGATHER`

For A2/A3 index gather, `tmp` has the same `i32`/`ui32` type and extent as the
index tile. For comparison gather, the minimum byte size is:

```text
tmpBytes >= tmpRows * tmpCols
            + srcRows * srcCols * sizeof(dstType)
            + srcRows * sizeof(srcType)
```

The first term is the packed comparison region; the other two terms hold the
index and converted-key regions. A5 does not access either form's `tmp`.

### `TSORT32`

Let `G = 32`, `C` be the source valid columns, and `E` be the source element
size. The result is in source-typed elements:

```text
A2/A3:
  tmpElements = align_32(C)   when C <= 8160 elements
  tmpElements = 32            when C > 8160 elements

A5:
  tmpElements = align_32(C)   when C * E <= 8160 bytes
  tmpElements = 32            when C * E > 8160 bytes
```

When `C` is already a multiple of 32, the tail workspace is not accessed.

### Row-expand family

For the explicit-tmp, scalar-per-row form of
`TROWEXPAND{ADD,SUB,MUL,DIV,MAX,MIN,EXPDIF}` on A2/A3:

```text
tmpBytes = ceil(R / 8) * 256   when R < 256
tmpBytes = 7680                when R >= 256
```

An 8192-byte allocation is a shape-independent safe bound. A5 accepts but does
not access the operand. The same rule is documented for
[TROWEXPANDADD](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDADD_zh.md#临时-tile),
[TROWEXPANDSUB](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDSUB_zh.md#临时-tile),
[TROWEXPANDMUL](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDMUL_zh.md#临时-tile),
[TROWEXPANDDIV](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDDIV_zh.md#临时-tile),
[TROWEXPANDMAX](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDMAX_zh.md#临时-tile),
[TROWEXPANDMIN](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDMIN_zh.md#临时-tile), and
[TROWEXPANDEXPDIF](https://gitcode.com/cann/pto-isa/blob/master/docs/isa/TROWEXPANDEXPDIF_zh.md#临时-tile).

### A5 `TMOV` ND-to-ZZ

For an MX quantization input of `M x N` with group size 32, the A5 ND-to-ZZ
exponent transform needs:

```text
tmpBytes = 2 * (32 + ceil(M / 16) * (N / 64))
```

The DN-to-ZZ overload accepts `tmp` for interface parity but does not access it.
The current PTO IR `pto.tmov` surface does not yet expose either three-operand
X-to-ZZ form.

### Arg reductions

For A2/A3 column arg-reduction, define:

```text
gap = elemPerRepeat                         when C >= elemPerRepeat
gap = align_elemPerBlock(C)                 when C < elemPerRepeat
safeStride = 3 * gap
```

The maximum-use `f16` index-only mode needs all three regions. A one-row `tmp`
with `safeStride` elements is therefore safe for every supported type/mode
combination; use the linked mode table only when deliberately shrinking the
workspace.

For A2/A3 row arg-reduction:

```text
C <= elemPerRepeat:
  index-only: tmp is not accessed
  value+index: 2 elements per source row

elemPerRepeat < C <= elemPerRepeat^2:
  r1 = ceil(C / elemPerRepeat)
  stride = (ceil(2 * r1 / elemPerBlock)
            + ceil(r1 / elemPerBlock)) * elemPerBlock

C > elemPerRepeat^2:
  r1 = ceil(C / elemPerRepeat)
  r2 = ceil(r1 / elemPerRepeat)
  stage1 = align_elemPerBlock(2 * r1)
  stage2End = align_elemPerBlock(r1) + align_elemPerBlock(2 * r2)
  stride = max(stage1, stage2End) + 2
```

The latter two cases require one such stride per source row. A5 does not access
the arg-reduction `tmp` operands.

### Fixed and shape-coupled workspaces

- `TCI`: 768 bytes for `i32`/`ui32` output and 1792 bytes for
  `i16`/`ui16` output. A 2048-byte allocation is a convenient common bound.
- INT8 `TQUANT`: an `f32` `tmp` with the same physical extent as `src`, or
  `4 * src.rows * src.cols` bytes.
- `TXOR` / `TXORS`: same element type and valid extent as the destination.
- `TPRELU`: row-major `ui8`; physical and valid row counts must cover at least
  `R + 1` rows, with each valid row wide enough for `ceil(C / 8)` packed
  predicate bytes.
- Multi-list `TMRGSORT`: one row, destination element type, and
  `tmp.cols >= sum(src_i.cols)`.
- Binary `TCOLSUM`: destination element type, at least `ceil(R / 2)` rows and
  at least `C` columns on both A2/A3 and A5. The sequential form does not access
  `tmp`.
- `TROWPROD`: one 32-byte block.
- Integer `TROWSUM`, `TROWMAX`, and `TROWMIN`: one 32-byte block. For floating
  types, a source-shaped tile is the documented safe bound.
- `TSEL`: `ui32`, with 16 logical bytes for the predicate state; reserve one
  aligned 32-byte block.
- `TSELS`: at least one source-typed element; reserve one aligned 32-byte
  block.
- High-precision `TRSQRT`: one 32-byte block on A2/A3.
- Floating `TPOW` / `TPOWS`: same type and valid extent as the destination;
  integer paths do not access `tmp`.
- `TREM`: destination element type, at least two rows and `C` columns.
- `TREMS`: destination element type, at least one row and `C` columns.
- `TADDDEQRELU`: `i32`, row-major, and the same valid extent as `dst`.
  PTOAS does not currently expose a corresponding PTO IR operation.
