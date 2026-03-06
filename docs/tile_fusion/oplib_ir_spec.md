# PTOAS OP-Lib IR Spec (V1.1)

## 1. Goal

This document defines the OP-Lib template contract used by PTOAS tile fusion.

V1.1 keeps the original loop/arithmetic template path working and adds a new
CCEC-template path. Both paths share the same function signature and attributes.

Supported binary floating element-wise ops:

- `tadd`
- `tsub`
- `tmul`
- `tdiv`
- `tmax`
- `tmin`

The design keeps OP identity in attributes instead of symbol suffixes:

- function symbols are descriptive only
- shape uses dynamic rank-2 memrefs (`?x?`)
- seed template dtype stays `f32`
- PTOAS auto-instantiates `f16` / `f32` concrete instances on demand

## 2. Required Function Attributes

Each template function must carry these attributes:

- `pto.oplib.op` : `"tadd" | "tsub" | "tmul" | "tdiv" | "tmax" | "tmin"`
- `pto.oplib.kind` : `"binary_elementwise_template"`
- `pto.oplib.rank` : `2 : i64`
- `pto.oplib.seed_dtype` : `"f32"`

If any required attribute is missing or invalid, PTOAS fails hard.

## 3. Signature Contract

Template function signature must be:

```mlir
(memref<?x?xT, #pto.address_space<vec>>,
 memref<?x?xT, #pto.address_space<vec>>,
 memref<?x?xT, #pto.address_space<vec>>) -> ()
```

V1 seed template requires `T = f32`.

## 4. Allowed Template Body Forms

### 4.1 Legacy Loop/Arith Form

Allowed dialects/op families:

- `func`
- `memref`
- `scf`
- `arith`

Required structure:

- 2-level nested `scf.for`
- per-element `load/load/compute/store`
- one floating binary core op matching the declared `pto.oplib.op`

Core op mapping:

- `tadd` -> `arith.addf`
- `tsub` -> `arith.subf`
- `tmul` -> `arith.mulf`
- `tdiv` -> `arith.divf`
- `tmax` -> `arith.maximumf`
- `tmin` -> `arith.minimumf`

### 4.2 Direct CCEC Form

Allowed dialects/op families:

- `func`
- `ccec`

Required structure:

- no `scf.for`
- one `ccec.vbin`
- `ccec.vbin kind` must match the declared `pto.oplib.op`

Core op mapping:

- `tadd` -> `ccec.vbin kind = "add"`
- `tsub` -> `ccec.vbin kind = "sub"`
- `tmul` -> `ccec.vbin kind = "mul"`
- `tdiv` -> `ccec.vbin kind = "div"`
- `tmax` -> `ccec.vbin kind = "max"`
- `tmin` -> `ccec.vbin kind = "min"`

### 4.3 Forbidden Body Patterns

Forbidden in all V1.1 templates:

- `memref.alloc` / `memref.dealloc`
- `func.call`
- `scf.if` / `scf.while`
- mixing legacy arith core ops and CCEC core ops in the same template
- core ops whose semantic kind does not match `pto.oplib.op`

## 5. Runtime Behavior

When `--enable-op-fusion` is enabled:

1. PTOAS scans `--op-lib-dir` and imports template functions.
2. PTOAS validates signature, attributes, and body contract.
3. PTOAS instantiates a concrete function for each `(op, dtype, signature)` on demand.
4. PTOAS materializes fused group functions by calling the instantiated symbols.
5. PTOAS inlines instantiated OP-Lib bodies into the synthesized fused function.
6. If the template used CCEC ops, PTOAS lowers them to `memref/scf/arith` loops.
7. Existing low-level loop fusion and downstream codegen keep running unchanged.

Failure policy:

- missing template / invalid attrs / unsupported dtype => compile error
- no silent fallback

## 6. Example: Legacy Loop Template (`tmul`)

```mlir
func.func @__pto_oplib_tmul_template(
    %src0: memref<?x?xf32, #pto.address_space<vec>>,
    %src1: memref<?x?xf32, #pto.address_space<vec>>,
    %dst: memref<?x?xf32, #pto.address_space<vec>>)
    attributes {
      pto.oplib.op = "tmul",
      pto.oplib.kind = "binary_elementwise_template",
      pto.oplib.rank = 2 : i64,
      pto.oplib.seed_dtype = "f32"
    } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
  %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>

  scf.for %i = %c0 to %m step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      %v = arith.mulf %a, %b : f32
      memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
    }
  }
  return
}
```

## 7. Example: Direct CCEC Template (`tmax`)

```mlir
func.func @__pto_ccec_tmax_template(
    %src0: memref<?x?xf32, #pto.address_space<vec>>,
    %src1: memref<?x?xf32, #pto.address_space<vec>>,
    %dst: memref<?x?xf32, #pto.address_space<vec>>)
    attributes {
      pto.oplib.op = "tmax",
      pto.oplib.kind = "binary_elementwise_template",
      pto.oplib.rank = 2 : i64,
      pto.oplib.seed_dtype = "f32"
    } {
  ccec.vbin kind = "max"
    ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
    outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
  return
}
```

## 8. Library Layout

Current tree layout:

- legacy loop templates: `test/tile_fusion/oplib`
- CCEC templates: `test/tile_fusion/oplib_ccec`

This is intentional for V1.1:

- legacy route remains the default compatibility path
- CCEC route is additive and can be enabled by switching `--op-lib-dir`
