# A5 BufID Sync Design

Date: 2026-05-29
Status: proposed implementation

## Intent

`bufid_sync` is an A5 intra-core synchronization pass for PTOAS. It inserts
`pto.get_buf` and `pto.rls_buf` around operations that access overlapping local
buffers on different hardware pipes. The goal is to replace event-based
`set_flag` / `wait_flag` synchronization for this path with buffer-id based
program-order synchronization.

The pass is enabled explicitly:

```bash
ptoas input.pto --pto-arch=a5 --enable-bufid_sync -o output.cpp
```

Non-goals:

- It does not change PTO dialect semantics for existing `set_flag` /
  `wait_flag` synchronization.
- It does not modify the existing `InsertSync` analysis implementation.
- It does not handle global memory synchronization through bufid; GM dependency
  handling remains outside this pass.
- It should not emit bufid synchronization for A3.

## Sync Model

### Event Sync

`inject_sync` / `pto-insert-sync` models hazards as explicit events:

```cpp
MTE2(tile0)
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)
V(tile0)
```

The event pair directly encodes a relation between source pipe, destination
pipe, and event id. It is precise, but consumes event resources and requires
pairing `set` and `wait` around every dependency edge.

### BufID Sync

`bufid_sync` uses:

```cpp
get_buf(PIPE_MTE2, id, 0)
MTE2(tile0)
rls_buf(PIPE_MTE2, id, 0)

get_buf(PIPE_V, id, 0)
V(tile0)
rls_buf(PIPE_V, id, 0)
```

The hardware rule is based on program order: operations wrapped by the same
buffer id are ordered across participating pipes. The id represents a virtual
resource derived from local buffer aliasing. If several operations touch the
same logical local buffer region, they should use the same id.

Critical invariant:

```text
For the same physical buf id, get_buf(id) / rls_buf(id) pairs must not nest.
```

This invariant is stronger than ordinary interval coloring. Any id reuse pass
must prove that the final emitted sequence has no same-id nesting.

## Inputs And Outputs

Input:

- PTO MLIR after frontend lowering and memory planning where possible.
- `PTOIRTranslator` SyncIR built in `SyncAnalysisMode::NORMALSYNC`.
- `MemoryDependentAnalyzer` results reused from `InsertSync`.

Output:

- The original PTO MLIR with inserted `pto.get_buf` and `pto.rls_buf`.
- Later `PTOToEmitC` lowers those ops to C++ intrinsic calls:

```cpp
get_buf(PIPE_MTE2, 0, 0);
rls_buf(PIPE_MTE2, 0, 0);
```

## Pass Placement

Recommended `ptoas` pipeline placement:

```text
lowering-sync-to-pipe
infer-layout
normalize-a5-tmov
view-to-memref
plan-memory, if not level3
resolve-reserved-buffers
bufid_sync, if --enable-bufid_sync
materialize-tile-handles
CSE
PTOToEmitC
```

Rationale:

- `view-to-memref` and `plan-memory` expose local memory address information.
- Running after memory planning lets alias checks use physical local addresses.
- Running before EmitC keeps synchronization in PTO dialect form.

The pass should require effective target arch `a5`. If `--enable-bufid_sync` is
used with `--pto-arch=a3`, the driver should reject the configuration.

## High-Level Flow

```text
func.func
  |
  v
Build SyncIR with PTOIRTranslator
  |
  v
Collect cross-pipe local-buffer dependency pairs
  |
  v
Collect dependent tiles and classify by address space and overlap
  |
  v
Build intersection graph and allocate virtual buf ids
  |
  v
Map dependency pairs to virtual ids
  |
  v
Create per-operation get/rls sync plans
  |
  v
Optimize logic ids when same-pipe ordering allows it
  |
  v
Allocate physical buf ids, default capacity = 32
  |
  v
Validate no same-id nesting
  |
  v
Emit pto.get_buf / pto.rls_buf ops
```

## Data Model

### TileInfo

Represents one local tile or local memref participating in cross-pipe
dependencies.

Fields:

- `memInfo`: pointer to `BaseMemInfo` from `MemoryDependentAnalyzer`.
- `scope`: PTO address space, for example `MAT`, `LEFT`, `RIGHT`, `ACC`, `VEC`.
- `baseAddr`: planned local address when available.
- `size`: allocated byte range size.
- `tileValue`: SSA value directly used by the operation.
- `rootBuffer`: underlying allocation or alias root.

Design requirement:

- GM tiles must be excluded.
- Tiles in different address spaces must never be in the same virtual id.
- Unknown local addresses should be handled conservatively by alias analysis.

### DepPair

Represents one ordered pair of cross-pipe operations with at least one local
buffer dependency.

Fields:

- `srcElement`: earlier SyncIR compound element.
- `dstElement`: later SyncIR compound element.
- `depTiles`: all local tiles participating in RAW, WAR, or WAW dependency.
- `srcPipe`, `dstPipe`: hardware pipe classification from SyncIR.

Dependency collection rules:

- Only inspect two different pipes.
- Skip GM dependencies.
- Record RAW, WAR, and WAW dependencies.
- Same-pipe dependencies do not need bufid sync because hardware/program order
  already orders them on that pipe.

### VirtualBufId

Represents a logical synchronization id before physical id allocation.

Fields:

- `logicId`: unique logical id.
- `tiles`: maximal clique of mutually-overlapping tiles.
- `scope`: address space shared by all tiles in the clique.

A tile may appear in multiple virtual ids if it belongs to multiple maximal
cliques. When a dependency can map to several ids, choose the id whose clique
contains the most relevant tiles.

### BufSyncOperation

Represents one planned synchronization operation before MLIR codegen.

Fields:

- `type`: `GET_BUF` or `RLS_BUF`.
- `pipe`: concrete producer/consumer pipe.
- `logicId`: virtual buf id.
- `syncIRIndex`: SyncIR position of this operation.
- `depSyncIRIndex`: SyncIR position of the paired dependency operation.

Per original PTO operation, the pass builds:

- `pipeBefore`: `get_buf` operations inserted before the operation.
- `pipeAfter`: `rls_buf` operations inserted after the operation.

## Dependency Collection

For each ordered pair of compound SyncIR elements:

```text
for i in compounds:
  for j in compounds after i:
    if pipe(i) == pipe(j):
      continue
    deps = DepBetween(def(i), use(j))   // RAW
         + DepBetween(use(i), def(j))   // WAR
         + DepBetween(def(i), def(j))   // WAW
    filter out GM and cross-space pairs
    if deps not empty:
      create DepPair(i, j, deps)
```

The design intent is to collect all local dependent tiles for an operation pair,
not just the first successful dependency class. This matters for choosing the
best virtual id and for debug explainability.

## Virtual BufID Construction

### Step 1: Tile Collection

Collect all local tiles from all dependency pairs.

Deduplication is allowed only when it preserves alias semantics:

- Same SSA value can be deduped.
- Same root, same address space, same address range can be deduped.
- If two tiles may alias, retaining either is acceptable only if
  `findBestVirtualBufId` can still map dependency meminfo back through alias
  checks.

### Step 2: Connected Components

Partition tiles by address space first. Within each address space, construct a
union-find over `MemAlias(tileA, tileB)`:

```text
same address space + may alias => same connected component
```

Properties:

- Different components do not overlap.
- A component contains one or more tiles connected through overlap.

### Step 3: Maximal Cliques

For each connected component, build an undirected intersection graph:

```text
vertex = tile
edge(tileA, tileB) = MemAlias(tileA, tileB)
```

Run Bron-Kerbosch to enumerate maximal cliques. Each maximal clique becomes one
`VirtualBufId`.

Why maximal cliques:

- A clique means every tile in the id overlaps every other tile.
- A maximal clique groups the largest mutually-overlapping buffer set without
  adding a non-overlapping tile.
- One tile may be in several maximal cliques, which preserves ambiguous overlap
  relationships.

Implementation guardrails:

- Clique enumeration can be exponential. Keep debug counters and abort or fall
  back conservatively if clique count exceeds a configured threshold.
- The traversal order should be deterministic so generated IDs are stable.

## Mapping Dependencies To Sync Ops

For each `DepPair`, choose a virtual id:

```text
candidate ids = ids containing or aliasing any dep tile
score(id) = number of dep tiles covered, then clique size, then stable logic id
best id = max score
```

Then insert planned sync operations:

```text
src pipeBefore += get_buf(srcPipe, bestId)
src pipeAfter  += rls_buf(srcPipe, bestId)
dst pipeBefore += get_buf(dstPipe, bestId)
dst pipeAfter  += rls_buf(dstPipe, bestId)
```

Deduplication rule:

- If an operation already has `get_buf(pipe, logicId)`, do not add another.
- If an operation already has `rls_buf(pipe, logicId)`, do not add another.

Ordering rule:

- If multiple ids guard one operation, emit `get_buf` in deterministic order and
  `rls_buf` in reverse order. This keeps the textual nesting stack-like and
  makes later legality checks easier.

## Same-Pipe Merge Optimization

Motivation:

```text
MTE1_A(tileA) -> M(tileA, tileB)
MTE1_B(tileB) -> M(tileA, tileB)
```

If `MTE1_A` and `MTE1_B` are naturally ordered on the same pipe, two logic ids
can sometimes be reduced to one:

```cpp
get_buf(PIPE_MTE1, 1, 0)
TMOV(tileA)
rls_buf(PIPE_MTE1, 1, 0)

get_buf(PIPE_MTE1, 1, 0)
TMOV(tileB)
rls_buf(PIPE_MTE1, 1, 0)

get_buf(PIPE_M, 1, 0)
TMATMUL(tileA, tileB)
rls_buf(PIPE_M, 1, 0)
```

Eligibility:

- The same operation has multiple logic ids on the same pipe.
- For each candidate logic id, all other users are on a single identical peer
  pipe.
- Merging does not create same-id nesting.
- Merging does not cross a control-flow boundary in a way that changes ordering
  assumptions.

This optimization should be run before physical id allocation because it reduces
the number of logical intervals.

## Physical BufID Allocation

Default physical ID capacity is 32.

### Life Interval

For each logic id:

- Linear sequence: interval start is first `get_buf`, interval end is last
  `rls_buf`.
- Inside `scf.for`: interval covers the outermost loop body, because loop back
  edges make all iterations overlap in the hardware-visible sequence.

Example:

```text
0  get_buf(id0)
1  op
2  rls_buf(id0)
...
8  get_buf(id0)
9  op
10 rls_buf(id0)

life(id0) = [0, 10]
```

For loop:

```text
5  scf.for begin
6    get_buf(id0)
7    op
8    rls_buf(id0)
19 scf.for end

life(id0) includes [5, 19]
```

### Linear Scan

Sort logic intervals by start position. Maintain:

- `active`: intervals currently live, sorted by end.
- `freeIds`: reusable physical ids released from expired intervals.
- `nextPhysicalId`: next never-used physical id.

Algorithm:

```text
for interval in intervals sorted by start:
  expire all active intervals whose end < interval.start
  if freeIds not empty:
    assign smallest free id
  else:
    assign nextPhysicalId++
  insert interval into active sorted by end
```

If `maxPhysicalId < 32`, allocation is complete. Otherwise, enter reuse.

### Reuse

Reuse is a fallback when logical intervals cannot fit in 32 physical IDs.

Principles:

- Only reuse ids inside the same pipe signature group, for example
  `[MTE2, MTE1]` with `[MTE2, MTE1]`.
- Prefer reuse groups with lower performance risk. A simple score:

```text
score = pipeScore(signature) * idCount^2
```

Where MTE2-containing groups get lower reuse priority because MTE2 latency is
usually more expensive.

Reuse layout:

- If producer pipe IDs are consecutive, pair adjacent IDs:
  `(0,1), (2,3)`.
- Otherwise pair first half with second half:
  `(0,2), (1,3)`.

Mandatory legality check after reuse and get/rls merging:

```text
for every physical id:
  scan final get/rls sequence in program order
  depth must never exceed 1
  rls must match a currently-open get
```

If the final plan fails legality or still requires more than 32 physical ids,
the pass fails with a diagnostic instead of emitting unsafe synchronization.

## Get/Rls Merge Optimization

Adjacent same-pipe, same-id release/acquire pairs can be removed:

```cpp
rls_buf(PIPE_MTE1, 0, 0)
get_buf(PIPE_MTE1, 0, 0)
```

becomes an extended live region:

```cpp
// removed boundary; one longer get/rls region
```

This optimization is safe only when:

- The pair has the same physical id and same pipe.
- There is no intervening operation that requires the id to become available.
- The final same-id nesting validator still passes.

Because it can extend live regions, run the final legality validator after this
optimization too.

## MLIR Codegen

`BufidSyncCodegen` lowers the planned sync model to PTO dialect ops:

```mlir
pto.get_buf[#pto.pipe_event_type<TLOAD>, 0]
...
pto.rls_buf[#pto.pipe_event_type<TLOAD>, 0]
```

The PTO dialect models bufid endpoints with `pipe_event_type` /
`sync_op_type`, matching the existing `record_event` / `wait_event` style.
`PTOToEmitC` maps the endpoint to the final C++ intrinsic pipe token.

Mapping from `PipelineType` to emitted endpoint:

| PipelineType | PTO endpoint | EmitC pipe |
| --- | --- | --- |
| `PIPE_MTE2` | `#pto.pipe_event_type<TLOAD>` | `PIPE_MTE2` |
| `PIPE_MTE3` | `#pto.pipe_event_type<TSTORE_VEC>` | `PIPE_MTE3` |
| `PIPE_FIX` | `#pto.pipe_event_type<TSTORE_ACC>` | `PIPE_FIX` |
| `PIPE_MTE1` | `#pto.pipe_event_type<TMOV_M2L>` | `PIPE_MTE1` |
| `PIPE_V` | `#pto.pipe_event_type<TVEC>` | `PIPE_V` |
| `PIPE_M` | `#pto.pipe_event_type<TMATMUL>` | `PIPE_M` |

Unknown, scalar, or virtual pipes must not silently map to `PIPE_UNASSIGNED`.
The pass fails with a diagnostic if it cannot map a pipe to a supported A5 bufid
endpoint.

## Correctness Invariants

The pass is correct only if all of these hold:

- No GM dependency is converted to bufid sync.
- Same-pipe dependency does not emit redundant bufid sync.
- Cross-pipe RAW, WAR, and WAW local dependencies are ordered by at least one
  common physical buf id.
- A physical buf id never has nested `get_buf` / `rls_buf`.
- A physical buf id never has unmatched `get_buf` or unmatched `rls_buf`.
- Physical id is in `[0, 31]`.
- A5-only ops are emitted only for A5.
- ID assignment is deterministic for stable generated output.

## Implementation Layout

Expected files:

```text
include/PTO/Transforms/Passes.td
include/PTO/Transforms/Passes.h
lib/PTO/Transforms/CMakeLists.txt
lib/PTO/Transforms/BufidSync/
  BufidSyncPass.cpp
  BufidSyncAnalysis.h
  BufidSyncAnalysis.cpp
  BufidSyncIdAlloc.h
  BufidSyncIdAlloc.cpp
  BufidSyncCodegen.h
  BufidSyncCodegen.cpp
tools/ptoas/ptoas.cpp
```

Responsibilities:

| Component | Responsibility |
| --- | --- |
| `BufidSyncPass` | pass orchestration, SyncIR construction, phase ordering |
| `BufidSyncAnalysis` | dep pair collection, tile grouping, virtual id allocation, logic sync planning |
| `BufidSyncIdAlloc` | life intervals, physical id allocation, reuse policy |
| `BufidSyncCodegen` | `PipelineType` to `pipe_event_type`, emit `pto.get_buf` / `pto.rls_buf` |
| `Passes.td` / `Passes.h` | pass registration and options |
| `ptoas.cpp` | CLI flag and pipeline integration |

## Validation Plan

Minimum build gate:

```bash
ninja -C build ptoas
```

Smoke tests:

```bash
build/tools/ptoas/ptoas test/samples/Sync/test_a5_buf_sync.pto \
  --pto-arch=a5 --pto-level=level3 \
  -o /tmp/manual_get_buf.cpp

build/tools/ptoas/ptoas test/samples/Sync/matmul.pto \
  --pto-arch=a5 --enable-bufid_sync \
  -o /tmp/bufid_matmul.cpp
```

Recommended regression tests:

- Basic `MTE2 -> MTE1` RAW dependency emits one id.
- `MTE1 -> M` dependency emits shared id between pipes.
- `M -> MTE3/FIX` store dependency emits shared id.
- Same-pipe dependencies emit no bufid sync.
- GM-only dependencies emit no bufid sync.
- Overlapping local tiles share virtual ids.
- Non-overlapping local tiles do not share virtual ids unless same-pipe merge is
  explicitly legal.
- More than 32 logical ids triggers reuse and final no-nesting validation.
- `--enable-bufid_sync --pto-arch=a3` is rejected.

## Implemented Guardrails

- Linear scan maintains a persistent free-id pool before allocating new physical
  ids.
- Dependency-to-virtual-id selection scores all dependent tiles by coverage,
  clique size, and stable logic id.
- Reuse is followed by a final same-physical-id nesting validator.
- `get_buf` emission is deterministic by physical id and pipe; `rls_buf`
  emission uses the reverse order.
- `--enable-bufid_sync` is rejected unless the effective target arch is A5.
- Debug output is gated by `--enable-bufid-sync-debug` or LLVM debug flags.
