# VMI VMULL Pair-Result Lowering Design

## Status

Proposed design for `mouliangyu/PTOAS:feature-vmi`.

This document defines the implementation contract for adding executable
`pto.vmi.vmull` support. The initial pull request containing this document is
design-only. ODS, verifier, lowering, PTODSL, user documentation, and tests are
follow-up implementation work.

## 1. Motivation

The physical VPTO operation already represents a native 32-bit widening
multiply as a pair of 32-bit vector results:

```mlir
%low, %high = pto.vmull %lhs, %rhs, %mask
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>
      -> !pto.vreg<64xi32>, !pto.vreg<64xi32>
```

One physical operation processes 64 `i32` or `ui32` lanes. VMI must expose the
same logical operation for 64, 128, and 256 lanes and split a larger logical
operation into the required physical operations automatically.

The current VMI definition instead returns one `Lxi64` value:

```mlir
%result = pto.vmi.vmull %a, %b, %mask
    : !pto.vmi.vreg<Lxi32>, !pto.vmi.vreg<Lxi32>, !pto.vmi.mask<Lxpred>
      -> !pto.vmi.vreg<Lxi64>
```

That form does not match the existing physical op. It would require a separate
virtual width-axis representation to translate one logical `i64` result into
low/high `i32` physical registers. The current VMI type converter has no such
result-axis contract, and there is no `VMIVmullOp` conversion pattern.

The proposed surface therefore represents the widened product as two logical
`Lxi32` results. This makes every physical chunk a direct pair-result lowering
while retaining VMI's logical lane count.

## 2. Goals and non-goals

### 2.1 Goals

- Support logical lane counts `L in {64, 128, 256}`.
- Support signed `i32` and unsigned `ui32` inputs.
- Return logical `%low` and `%high` values, both with the same `LxT` type as
  the inputs.
- Lower a contiguous 64/128/256-lane operation to exactly 1/2/4 physical
  `pto.vmull` operations.
- Preserve one common layout relation across both inputs, the mask, and both
  results.
- Preserve the low-result and high-result grouping expected by MLIR 1:N type
  conversion.
- Expose the pair result through PTODSL as a Python tuple.
- Reject unsupported shapes and predicate modes before conversion instead of
  leaving residual VMI IR.

### 2.2 Non-goals

- Supporting lane counts other than 64, 128, and 256.
- Reconstructing a logical `Lxi64` value from the low/high result pair.
- Adding a new VPTO op or changing the existing VPTO emitter.
- Supporting merge predication without explicit low/high passthrough values.
- Adding partial-register or tail-specific VMULL behavior.
- Implementing the code in the design-only pull request.

## 3. Proposed VMI operation contract

### 3.1 Syntax

```mlir
%low, %high = pto.vmi.vmull %a, %b, %mask
    : !pto.vmi.vreg<LxT>, !pto.vmi.vreg<LxT>, !pto.vmi.mask<Lxpred>
      -> !pto.vmi.vreg<LxT>, !pto.vmi.vreg<LxT>
```

Examples:

```mlir
%low64, %high64 = pto.vmi.vmull %a64, %b64, %mask64
    : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>,
      !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>

%low256, %high256 = pto.vmi.vmull %a256, %b256, %mask256
    : !pto.vmi.vreg<256xui32>, !pto.vmi.vreg<256xui32>,
      !pto.vmi.mask<256xpred>
      -> !pto.vmi.vreg<256xui32>, !pto.vmi.vreg<256xui32>
```

### 3.2 Type constraints

The verifier requires all of the following:

```text
L is exactly 64, 128, or 256
T is exactly i32 or ui32
type(a) == type(b) == type(low) == type(high)
lanes(mask) == L
```

At the surface boundary, none of these types carries an assigned layout or a
concrete mask granularity. After assignment, the data values retain identical
element type, lane count, and layout. The mask has the same logical lane count
and layout and uses `b32` granularity.

Mixed signedness is not legal. In particular, `i32` inputs cannot return
`ui32` results, and `ui32` inputs cannot return `i32` results.

The old single `Lxi64` result form is removed rather than supported as a second
assembly form. Repository search shows no active VMI VMULL tests or kernel
consumers, so an atomic update of ODS, PTODSL, documentation, and tests is
preferred over maintaining two contracts.

### 3.3 Per-lane semantics

For active lane `i`, the operation computes one 64-bit product and returns its
two 32-bit halves:

```text
if T == i32:
  product = signed_64(a[i]) * signed_64(b[i])
else:
  product = unsigned_64(a[i]) * unsigned_64(b[i])

low[i]  = bits(product, 31, 0)
high[i] = bits(product, 63, 32)
```

The result values use `T` so that signedness continues to select the existing
signed or unsigned physical VMULL form. The low 32 bits have the same bit
pattern for signed and unsigned multiplication; the high 32 bits reflect the
selected signedness.

### 3.4 Predicate mode

The first implementation is zeroing-only:

```text
mask[i] == true:
  low[i], high[i] are the product halves

mask[i] == false:
  low[i] = 0
  high[i] = 0
```

For minimum assembly churn, `OptionalAttr<StrAttr>:$pmode` may remain in ODS,
but the verifier accepts only an omitted attribute or `pmode = "zero"`. An
omitted attribute means zeroing.

`pmode = "merge"` must be rejected by the verifier. The operation is pure and
has no old `%low` or `%high` passthrough operands, so there is no SSA value from
which inactive lanes could be preserved. Supporting merge later requires an
explicit API decision, such as adding two passthrough operands; it cannot be
implemented correctly by the conversion pattern alone.

## 4. Layout contract

VMULL is an elementwise pair producer. Logical lane `i` in `%a`, `%b`, and
`%mask` maps to logical lane `i` in both `%low` and `%high`.

The assigned layout relation is:

```text
layout(a) == layout(b) == layout(mask) == layout(low) == layout(high)
```

The initial implementation supports dense layouts with `lane_stride = 1`:

- contiguous;
- deinterleaved layouts already representable by the VMI type converter.

Group-slot layouts and dense lane-stride layouts greater than one are not part
of the initial contract. They may change the physical footprint or physical
element type in ways that do not match `pto.vmull`'s `64xi32` inputs.

Contiguous is the default layout and gives the required canonical expansion:

| Logical type | Physical input parts | Physical VMULL count | Physical results |
|---|---:|---:|---:|
| `64xi32` | 1 | 1 | 1 low + 1 high |
| `128xi32` | 2 | 2 | 2 low + 2 high |
| `256xi32` | 4 | 4 | 4 low + 4 high |

For another supported dense layout, lowering uses the physical arity computed
by `getVMIPhysicalArity`. Corresponding parts of all five logical values must
have the same arity and type. This keeps the conversion correct for a legal
deinterleaved value without hard-coding contiguous part order into the pattern.

### 4.1 Assignment and propagation integration

VMULL cannot be implemented only in the final conversion pattern. The earlier
passes must establish its mask and layout contract:

1. `VMIMaskGranularityAssignment` requests `b32` for the VMULL mask because
   the data element type is 32-bit.
2. `VMILayoutAssignment` unifies `%a`, `%b`, `%low`, and `%high` as one data
   layout equivalence class.
3. `VMILayoutPropagation` treats `VMIVmullOp` as a same-layout operation so a
   layout fact on any input, mask, or result propagates to the other ports.
4. Conflicting consumer layouts are handled through the existing explicit
   `ensure_layout` or `ensure_mask_layout` materialization mechanism.

No VMULL-specific layout transform is needed. The relation is identity across
all ports, unlike `vintlv`/`vdintlv`, whose input and result layouts may differ.

## 5. VMI-to-VPTO lowering

### 5.1 Canonical contiguous expansion

For contiguous `Lxi32`, define:

```text
K = L / 64
```

The type converter produces:

```text
a    -> [a_0,    ..., a_(K-1)]
b    -> [b_0,    ..., b_(K-1)]
mask -> [mask_0, ..., mask_(K-1)]
low  -> [low_0,  ..., low_(K-1)]
high -> [high_0, ..., high_(K-1)]
```

The conversion creates one physical operation for every `p` in `[0, K)`:

```text
low_p, high_p = pto.vmull(a_p, b_p, mask_p)

a_p, b_p, low_p, high_p : !pto.vreg<64xi32> or !pto.vreg<64xui32>
mask_p                  : !pto.mask<b32>
```

### 5.2 Conversion pattern algorithm

`OneToNVMIVmullOpPattern` performs the following checks and rewrite:

1. Read `aParts`, `bParts`, and `maskParts` from the 1:N adaptor.
2. Obtain the converted type lists for result index 0 (`low`) and result index
   1 (`high`).
3. Require all five arities to be equal and non-zero.
4. Require every input and result part to be `!pto.vreg<64xi32>` or
   `!pto.vreg<64xui32>` with matching signedness, and every mask part to be
   `!pto.mask<b32>`.
5. For physical part `p`, create
   `pto.vmull(aParts[p], bParts[p], maskParts[p])`.
6. Replace the two logical results with the flattened physical result list.

The flattened replacement order is critical:

```text
[low_0, low_1, ..., low_(K-1), high_0, high_1, ..., high_(K-1)]
```

It must not be interleaved as `[low_0, high_0, low_1, high_1, ...]`.
`replaceOpWithFlatConvertedValues` partitions the flat list by logical result
index, so the first complete segment belongs to `%low` and the second complete
segment belongs to `%high`.

### 5.3 Preflight validation

`verifySupportedVMIToVPTOOps` gains an explicit VMULL check before dialect
conversion. The check requires:

- assigned, equal, supported dense layouts on all ports;
- `lane_stride = 1`;
- `b32` mask granularity;
- legal 32-bit signedness and lane count;
- matching, computable physical arity for inputs, mask, and both results;
- physical arity in the supported set produced by the legal logical shapes.

This gives an actionable `VMI-UNSUPPORTED` diagnostic instead of a final
`VMI-RESIDUAL-OP` failure.

## 6. Pipeline integration

The complete implementation crosses the following layers:

| Layer | File | Required change |
|---|---|---|
| ODS | `include/PTO/IR/VMIOps.td` | Change one `Lxi64` result to `(low, high)` `Lxi32` results and update syntax/description |
| Verifier | `lib/PTO/IR/VMI.cpp` | Enforce legal lane counts, types, pair equality, mask shape, and zero-only pmode |
| Mask assignment | `lib/PTO/Transforms/VMIMaskGranularityAssignment.cpp` | Request `b32` for the mask use |
| Layout assignment | `lib/PTO/Transforms/VMILayoutAssignment.cpp` | Unite both inputs and both results |
| Layout propagation | `lib/PTO/Transforms/VMILayoutPropagation.cpp` | Register VMULL as a same-layout relation |
| Unified bridge | `lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp` | Keep VMULL on the direct-to-VPTO path; update comments only if needed |
| Physicalization | `lib/PTO/Transforms/VMIToVPTO.cpp` | Add preflight validation, pair-result 1:N pattern, and pattern registration |
| PTODSL | `ptodsl/ptodsl/_vmi_namespace.py` | Return two results and infer their types from the inputs |
| User docs | VMI ISA and PTODSL guide | Replace the single widened result with the pair-result contract |
| Tests | `test/lit/vmi_new`, `ptodsl/tests` | Add verifier, assignment, lowering, and frontend coverage |

The existing `PTO_VmullOp`, `VmullOp::verify`, LLVM emitters, and physical
VMULL tests remain unchanged.

## 7. PTODSL API

The Python surface follows the existing two-result `vintlv`/`vdintlv` style:

```python
low, high = pto.vmi.vmull(
    a,
    b,
    mask,
    result_types=None,
    pmode=None,
)
```

When `result_types` is omitted, both result types are inferred from `a` and
`b`. If provided, it must be a two-item sequence; the IR verifier still
requires both types to equal the input type.

The old required `result_type=` keyword is removed atomically with the ODS
change. Because there are no in-repository PTODSL VMULL call sites, a temporary
dual API is not required. The user guide must clearly document that the return
value is a tuple `(low, high)`.

## 8. Test plan

### 8.1 ODS and verifier tests

Add positive parse/verify coverage for:

- `64xi32`;
- `128xi32`;
- `256xi32`;
- `64xui32` or another unsigned shape;
- omitted pmode and explicit `pmode = "zero"`.

Add negative coverage for:

- lane count outside `{64, 128, 256}`;
- input element type other than `i32/ui32`;
- input type or signedness mismatch;
- low/high type, signedness, or lane-count mismatch;
- mask lane-count mismatch;
- the old single `Lxi64` result form;
- `pmode = "merge"` and unknown pmode strings.

### 8.2 Mask and layout tests

- Verify that `!pto.vmi.mask<Lxpred>` becomes `b32` for VMULL.
- Verify that `%a`, `%b`, `%mask`, `%low`, and `%high` receive the same layout.
- Verify that a conflicting consumer layout inserts an explicit
  materialization rather than silently changing one VMULL port.
- Cover at least one supported deinterleaved same-layout case if that support
  is enabled in the implementation patch.

### 8.3 VMI-to-VPTO tests

For contiguous layouts, check exact physical operation counts:

```text
64xi32  -> CHECK-COUNT-1: pto.vmull
128xi32 -> CHECK-COUNT-2: pto.vmull
256xi32 -> CHECK-COUNT-4: pto.vmull
```

The multi-chunk checks must also capture result grouping, not only operation
count. Uses of the logical `%low` value must receive all low parts, while uses
of `%high` must receive all high parts.

Add both signed and unsigned lowering coverage and verify `!pto.mask<b32>` on
every emitted physical operation.

### 8.4 PTODSL and end-to-end tests

- Verify `pto.vmi.vmull` returns a two-item tuple.
- Verify inferred and explicit `result_types` forms.
- Verify invalid explicit result types fail with a clear diagnostic.
- Compile at least one VMI-authored kernel through the complete VPTO pipeline
  and confirm no VMI op or type remains.

## 9. Compatibility and rollout

Changing one `Lxi64` result to two `Lxi32` results is an intentional breaking
change to an incomplete, currently non-lowerable VMI operation. All public
layers must change in one implementation pull request:

```text
ODS + verifier
mask/layout assignment
VMIToVPTO preflight + conversion
PTODSL
VMI ISA and PTODSL documentation
focused regression tests
```

Splitting the ODS change from the conversion pattern would temporarily create
another parseable but non-executable VMULL form and is therefore not
recommended.

## 10. Acceptance criteria

The implementation is complete when:

1. The only accepted VMI VMULL surface is pair-result `Lxi32/ui32`, with
   `L in {64, 128, 256}`.
2. Mask granularity and all five layouts are assigned consistently.
3. Contiguous 64/128/256-lane inputs produce exactly 1/2/4 physical
   `pto.vmull` operations.
4. Low and high result parts are associated with the correct logical result.
5. Signed and unsigned forms select their existing physical backend forms.
6. Merge predication is rejected before physical conversion.
7. The full pipeline contains no residual VMI op or VMI type.
8. ODS, verifier, lowering, PTODSL, user documentation, and tests land
   together in the follow-up implementation pull request.

## 11. Decisions requested from review

This proposal asks reviewers to confirm two API decisions before code is
implemented:

1. Use two logical `Lxi32/ui32` results instead of one logical `Lxi64/ui64`
   result.
2. Define the first implementation as zeroing-only and reject merge until the
   operation has explicit passthrough semantics.

Once those decisions are accepted, the implementation path is fully defined
by the contracts above.
