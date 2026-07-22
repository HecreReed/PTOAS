### Synchronization & Buffer Control

Operations for pipeline synchronization and buffer management.

#### Enum Types for Synchronization

The following enum types provide type-safe parameter specification for synchronization operations:

- **`BarrierType`**: Memory barrier types for `pto.mem_bar`
  - `VV_ALL`, `VST_VLD`, `VLD_VST`, `VST_VST`: vector→vector barriers
  - `VS_ALL`, `VST_LD`, `VLD_ST`, `VST_ST`: vector→scalar barriers
  - `SV_ALL`, `ST_VLD`, `LD_VST`, `ST_VST`: scalar→vector barriers

- **`Pipe`**: Hardware pipeline identifiers
  - `MTE2`: Memory Transfer Engine 2 pipeline
  - `V`: Vector pipeline
  - `MTE3`: Memory Transfer Engine 3 pipeline
  - `ALL`: All pipelines (for barrier operations)

- **`Event`**: Event identifiers for synchronization
  - `ID0`, `ID1`, `ID2`, `ID3`, ..., `ID31`: Event IDs 0-31 (A5 supports 32 event IDs, 0-15 for subblock 0, 16-31 for subblock 1)

#### `pto.set_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Sets a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.wait_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Waits for a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.pipe_barrier(pipes: PIPE) -> None`

**Description**: Executes a barrier across specified pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `PIPE` | Pipeline specification (e.g., `PIPE.ALL`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE

pto.pipe_barrier(PIPE.ALL)
```

#### `pto.get_buf(pipe: Pipe, buf_id: pto.i64, mode: pto.i64) -> None`

**Description**: Acquire buffer slot for inter-pipeline double-buffering coordination.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier (e.g., `Pipe.MTE2`, `Pipe.V`, `Pipe.MTE3`) |
| `buf_id` | `pto.i64` | Buffer identifier |
| `mode` | `pto.i64` | Acquisition mode |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Pipe

# Acquire buffer for MTE2 pipeline
pto.get_buf(Pipe.MTE2, 0, 0)
```

#### `pto.rls_buf(pipe: Pipe, buf_id: pto.i64, mode: pto.i64) -> None`

**Description**: Release buffer slot to allow other pipeline to proceed.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier (e.g., `Pipe.MTE2`, `Pipe.V`, `Pipe.MTE3`) |
| `buf_id` | `pto.i64` | Buffer identifier |
| `mode` | `pto.i64` | Release mode |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Pipe

# Release buffer for MTE2 pipeline
pto.rls_buf(Pipe.MTE2, 0, 0)
```

#### `pto.mem_bar(barrier_type: BarrierType) -> None`

**Description**: Memory barrier for pipeline synchronization within vector scope. Required when UB addresses alias between vector load/store operations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `barrier_type` | `BarrierType` | Barrier type controlling prior/subsequent instruction ordering. Supported values are `BarrierType.VV_ALL`, `BarrierType.VST_VLD`, `BarrierType.VLD_VST`, `BarrierType.VST_VST`, `BarrierType.VS_ALL`, `BarrierType.VST_LD`, `BarrierType.VLD_ST`, `BarrierType.VST_ST`, `BarrierType.SV_ALL`, `BarrierType.ST_VLD`, `BarrierType.LD_VST`, and `BarrierType.ST_VST`. |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import BarrierType

# Ensure stores are visible before loads to same UB region
pto.mem_bar(BarrierType.VST_VLD)
```

#### `pto.set_cross_core(core_id: pto.i64, event_id: Event) -> None`

**Description**: Signal event to another core (cross-core synchronization).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Target/source core identifier (platform-specific mapping) |
| `event_id` | `Event` | Cross-core event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Signal event ID0 to core 0
pto.set_cross_core(0, Event.ID0)
```

#### `pto.set_intra_block(block_id: pto.i64, event_id: Event) -> None`

**Description**: Signal event within a block (A5). Specifies trigger pipe. 1:1 per subblock.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block/pipeline identifier specifying trigger pipe |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Signal event ID0 on block/pipeline 0
pto.set_intra_block(0, Event.ID0)
```

#### `pto.set_intra_core(config: pto.i32) -> None`

**Description**: Configures intra-core synchronization settings.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `pto.i32` | Configuration value for intra-core synchronization |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_intra_core(3)
```

#### `pto.wait_flag_dev(core_id: pto.i64, event_id: Event) -> None`

**Description**: Wait for event from another core. SU-level blocking — entire core stalls.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Core identifier |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Wait for event ID0 from core 0
pto.wait_flag_dev(0, Event.ID0)
```

#### `pto.wait_intra_core(block_id: pto.i64, event_id: Event) -> None`

**Description**: Wait for event within block (A5). Specifies which pipeline should wait — only that pipe stalls, SU and other pipes continue.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block/pipeline identifier specifying which pipeline should wait |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Wait for event ID0 on block/pipeline 0
pto.wait_intra_core(0, Event.ID0)
```

### DMA Programming [Advanced Tier]

This section covers the canonical grouped DMA authoring surface for transfers
between Global Memory (GM), Unified Buffer (UB), and L1. The current public
MTE DMA names follow the VPTO manual directly:

- `pto.mte_gm_ub`
- `pto.mte_ub_gm`
- `pto.mte_ub_ub`
- `pto.mte_ub_l1`

These grouped DMA operations are the current authoring-facing MTE contract.
Older `copy_*` and `set_loop*_stride_*` / `set_loop_size_*` surfaces are
legacy compatibility or lowering-detail APIs; they are documented later in this
chapter as compatibility material, not as the canonical public entry points.

**Key Concepts:**
- **Grouped transfer model**: `nburst(...)` expresses the innermost burst
  pattern; optional `loop(...)` groups add outer repetition levels.
- **Stride units depend on the op family**:
  - `pto.mte_gm_ub` and `pto.mte_ub_gm` use byte strides
  - `pto.mte_ub_ub` and `pto.mte_ub_l1` use 32B burst units
- **GM→UB padding**: `pto.mte_gm_ub` optionally supports `pad(...)` for UB row
  padding.

**Usage Flow:**
1. Choose the canonical grouped DMA op for the source/destination address spaces.
2. Provide the required `nburst(...)` group.
3. Add optional `loop(...)` groups when the transfer needs outer repetition.
4. For GM→UB padding, add `pad(...)` directly on `pto.mte_gm_ub`.

**Note**: All grouped DMA operations in this section are part of the
**Advanced Tier** and require explicit pointer-form authoring.

#### Canonical Grouped DMA Example

```python
# GM -> UB grouped DMA
pto.mte_gm_ub(
    gm_ptr,
    ub_ptr,
    0,      # l2_cache_ctl
    128,    # len_burst in bytes
    nburst=(16, 128, 128),
)
```

#### Pad Fill Semantics

When using `pto.mte_gm_ub`, you can add a `pad(...)` clause to fill padded UB
row regions with a specified scalar value. This is useful when the source data
does not perfectly match the padded UB row shape, or when you need explicit
boundary fill behavior in grouped GM→UB DMA.

##### How Padding Works

1. **Select the GM→UB grouped DMA op**: Use `pto.mte_gm_ub(...)`.
2. **Add the `pad(...)` clause**: Provide the pad scalar, and optionally the
   left/right padding counts.
3. **Keep the grouped DMA structure explicit**: `pad(...)` is attached to the
   same op as `nburst(...)` and any `loop(...)` groups.

##### Example Workflow

Add the `pad(...)` clause directly to `pto.mte_gm_ub`:

```python
# Pad each UB row with 0.0 after the valid 200B payload
pto.mte_gm_ub(
    gm_ptr,
    ub_ptr,
    0,      # l2_cache_ctl
    200,    # len_burst
    nburst=(32, 200, 256),
    pad=(pto.f32(0.0), 0, 0),
)
```

##### Accessing Pad Values in Kernel Code

Tile `PadValue` descriptors can be used within kernel code for computation
purposes (for example, initializing vectors with a specific fill value).  
However, note that these descriptors are not automatically threaded into the
grouped DMA `pad(...)` clause; you still need to materialize the desired scalar
explicitly in the DMA call.

To access a pad value from a tile descriptor in kernel code:

```python
# Get the pad descriptor from the destination tile
pad_desc = dst.pad_value

# Check if a valid pad value is configured
if pto.constexpr(pad_desc != pto.PadValue.NULL):
    # Materialize the scalar value
    pad_scalar = pad_desc.eval()
    
    # Use the scalar value (e.g., for vector duplication)
    mask = pto.make_mask(pto.f32, PAT.ALL)
    pad_vector = pto.vdup(pad_scalar, mask)
```

##### Important Notes

- The `PadValue.NULL` descriptor indicates no pad value is configured. Attempting to call `.eval()` on `PadValue.NULL` will raise a frontend error.
- Custom pad values currently support only 32-bit float payloads (`PadValue.custom_f32(...)`).
- Padding only affects GM→UB transfers (`pto.mte_gm_ub`). UB→GM and UB→UB transfers do not support padding.
- The padded region is determined by the difference between the tile's `valid_shape` and its full `shape`. Ensure your tile is configured with appropriate dimensions.
- Tile `PadValue` descriptors are not automatically threaded into DMA padding; materialize the scalar explicitly in `pad(...)` when needed.

#### `pto.mte_gm_ub(gm_src, ub_dst, l2_cache_ctl, len_burst, *, nburst, loops=None, pad=None) -> None`  [Advanced Tier]

**Description**: Grouped GM→UB DMA transfer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `gm_src` | `GMPtr` | Source GM pointer |
| `ub_dst` | `UBPtr` | Destination UB pointer |
| `l2_cache_ctl` | `pto.i64` | L2 cache control operand |
| `len_burst` | `pto.i64` | Bytes transferred per burst row |
| `nburst` | `tuple[i64, i64, i64]` | Required burst triple `(count, src_stride, dst_stride)` |
| `loops` | `tuple[tuple[i64, i64, i64], ...] \| None` | Optional outer loop triples from inner to outer |
| `pad` | `tuple[ScalarType] \| tuple[ScalarType, i64, i64] \| None` | Optional pad payload and optional left/right counts |

**Notes**:
- `nburst(...)` is required.
- `src_stride` and `dst_stride` are byte strides.
- `pad(...)` is only part of `pto.mte_gm_ub`.

**Example**:
```python
pto.mte_gm_ub(
    gm_ptr,
    ub_ptr,
    0,
    128,
    nburst=(32, 128, 128),
)
```

**Padding Example**:
```python
pto.mte_gm_ub(
    gm_ptr,
    ub_ptr,
    0,
    200,
    nburst=(32, 200, 256),
    pad=(pto.f32(0.0), 0, 0),
)
```

#### `pto.mte_ub_gm(ub_src, gm_dst, len_burst, *, nburst, loops=None, l2_cache="nmfv") -> None`  [Advanced Tier]

**Description**: Grouped UB→GM DMA transfer.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `UBPtr` | Source UB pointer |
| `gm_dst` | `GMPtr` | Destination GM pointer |
| `len_burst` | `pto.i64` | Bytes transferred per burst row |
| `nburst` | `tuple[i64, i64, i64]` | Required burst triple `(count, src_stride, dst_stride)` |
| `loops` | `tuple[tuple[i64, i64, i64], ...] \| None` | Optional outer loop triples from inner to outer |
| `l2_cache` | `str` | Store-side L2 cache policy token. Defaults to `"nmfv"` (raw control `0`); supported tokens are `"nmfv"`, `"nmlv"`, `"nmprs"`, `"nmred"`, `"naci"`, `"napw"`, `"napi"`, `"nared"`, `"wbhfv"`, `"wbhlv"`, `"wbhprs"`, `"wbhred"`, `"wtsfv"`, `"wtslv"`, `"wtsprs"`, and `"wtsred"` |

**Example**:
```python
pto.mte_ub_gm(
    ub_ptr,
    gm_ptr,
    128,
    nburst=(32, 128, 128),
    l2_cache="nmfv",
)
```

#### `pto.mte_ub_ub(ub_src, ub_dst, len_burst, *, nburst) -> None`  [Advanced Tier]

**Description**: Grouped UB→UB copy.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `UBPtr` | Source UB pointer |
| `ub_dst` | `UBPtr` | Destination UB pointer |
| `len_burst` | `pto.i64` | Burst length in units of 32 bytes |
| `nburst` | `tuple[i64, i64, i64]` | Required burst triple `(count, src_gap, dst_gap)` in 32B units |

**Example**:
```python
pto.mte_ub_ub(
    ub_src,
    ub_dst,
    4,
    nburst=(8, 0, 0),
)
```

#### `pto.mte_ub_l1(ub_src, l1_dst, len_burst, *, nburst) -> None`  [Advanced Tier]

**Description**: Grouped UB→L1 copy.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `UBPtr` | Source UB pointer |
| `l1_dst` | `L1Ptr` | Destination L1 pointer |
| `len_burst` | `pto.i64` | Burst length in units of 32 bytes |
| `nburst` | `tuple[i64, i64, i64]` | Required burst triple `(count, src_gap, dst_gap)` in 32B units |

**Example**:
```python
pto.mte_ub_l1(
    ub_src,
    l1_dst,
    4,
    nburst=(8, 0, 0),
)
```

#### Legacy Compatibility Notes

Older `copy_gm_to_ubuf`, `copy_ubuf_to_gm`, `copy_ubuf_to_ubuf`,
`set_loop*_stride_*`, `set_loop_size_*`, and `set_mov_pad_val` surfaces may
still exist in parts of the implementation as compatibility or lowering-detail
APIs. They are intentionally kept out of the grouped DMA public examples in
this guide: canonical grouped DMA authoring here uses `pto.mte_*`, while
`copy_*` remains low-level compatibility/programming detail rather than the
grouped DMA public contract.
