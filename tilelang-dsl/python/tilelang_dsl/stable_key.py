# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Stable key computation for TileLang DSL instance caching.

This module provides StableKey computation logic for the daemon instance cache.
The stable key must include all fields that affect kernel selection, not just
those that affect code generation.

Key design principle (from expert review):
The cache key must include all fields that affect kernel selection,
not just those that affect code generation. Any field that affects
the selection result must be included in the key.

Therefore, for View operands, shape/strides/memory_space must all enter the key,
even though they may not affect the generated MLIR code directly. They can
affect constraint evaluation which determines which template is selected.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StableKey:
    """Stable key for instance caching.

    This key is used to identify a unique instance in the daemon's cache.
    It must be stable: same parameters always produce the same key.
    """
    target: str
    op: str
    operand_keys: tuple[OperandStableKey, ...]
    context_key: frozenset[tuple[str, str]]

    def to_hash(self) -> str:
        """Compute hash for internal indexing.

        Returns a 16-character hex string derived from SHA256 hash.
        """
        parts = [
            self.target,
            self.op,
            *[ok.to_hash_input() for ok in self.operand_keys],
            *[f"{k}={v}" for k, v in sorted(self.context_key)],
        ]
        input_str = "|".join(parts)
        return hashlib.sha256(input_str.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class OperandStableKey:
    """Stable key for a single operand.

    Supports three kinds of operands:
    - Tile: from TileBufType, all fields enter key
    - View: from MemRefType, enhanced with shape/strides/ms
    - Scalar: from scalar type, only dtype enters key
    """
    kind: str  # "tile", "view", "scalar"
    dtype: str  # "f32", "f16", "bf16", "i8", ...

    # === Tile fields (all enter key) ===
    shape: tuple[int, ...] | None = None
    valid_shape: tuple[int | None, ...] | None = None  # None means dynamic
    memory_space: str | None = None  # "ub", "gm"
    blayout: str | None = None  # "row_major", "col_major"
    slayout: str | None = None  # "row_major", "col_major", "none_box"
    fractal: int | None = None
    pad: int | None = None

    # === View fields (ENHANCED: all enter key) ===
    # Reason: may affect constraint evaluation -> kernel selection
    view_shape: tuple[int | None, ...] | None = None
    view_strides: tuple[int | None, ...] | None = None
    view_memory_space: str | None = None
    view_layout: str | None = None

    def to_hash_input(self) -> str:
        """Generate hash input string for this operand."""
        if self.kind == "tile":
            # Tile: all fields
            vs_str = ",".join(
                "?" if v is None else str(v) for v in self.valid_shape
            ) if self.valid_shape else ""
            s_str = ",".join(str(s) for s in self.shape) if self.shape else ""
            return (
                f"tile:{self.dtype}:{s_str}:{vs_str}:"
                f"{self.memory_space}:{self.blayout}:{self.slayout}:"
                f"{self.fractal}:{self.pad}"
            )
        elif self.kind == "view":
            # View: ENHANCED with shape/strides/ms
            s_str = ",".join(
                "?" if s is None else str(s) for s in self.view_shape
            ) if self.view_shape else ""
            st_str = ",".join(
                "?" if st is None else str(st) for st in self.view_strides
            ) if self.view_strides else ""
            return (
                f"view:{self.dtype}:{s_str}:{st_str}:"
                f"{self.view_memory_space}:{self.view_layout}"
            )
        else:  # scalar
            return f"scalar:{self.dtype}"


def normalize_dtype(dtype: Any) -> str:
    """Normalize dtype to string.

    Args:
        dtype: Can be ScalarType object, string, or other type marker

    Returns:
        Normalized dtype string (e.g., "f32", "f16", "i8")
    """
    if isinstance(dtype, str):
        return dtype.lower()
    
    # Handle ScalarType from types.py
    if hasattr(dtype, "name"):
        return dtype.name.lower()
    
    # Fallback: convert to string
    return str(dtype).lower()


def normalize_blayout(blayout: Any) -> str:
    """Normalize blayout to string.

    Args:
        blayout: Can be enum value, string, or int

    Returns:
        Normalized blayout string ("row_major" or "col_major")
    """
    if blayout is None:
        return "row_major"  # default
    
    if isinstance(blayout, str):
        return blayout.lower()
    
    # Handle enum
    if hasattr(blayout, "value"):
        val = blayout.value
        if isinstance(val, str):
            return val.lower()
        # Int value: 0 = row_major, 1 = col_major
        return "row_major" if val == 0 else "col_major"
    
    # Handle int directly
    if isinstance(blayout, int):
        return "row_major" if blayout == 0 else "col_major"
    
    return str(blayout).lower()


def normalize_slayout(slayout: Any) -> str:
    """Normalize slayout to string.

    Args:
        slayout: Can be enum value, string, or int

    Returns:
        Normalized slayout string ("row_major", "col_major", or "none_box")
    """
    if slayout is None:
        return "none_box"  # default
    
    if isinstance(slayout, str):
        return slayout.lower()
    
    # Handle enum
    if hasattr(slayout, "value"):
        val = slayout.value
        if isinstance(val, str):
            return val.lower()
        # Int value: 0 = none_box, 1 = row_major, 2 = col_major
        if val == 0:
            return "none_box"
        elif val == 1:
            return "row_major"
        else:
            return "col_major"
    
    # Handle int directly
    if isinstance(slayout, int):
        if slayout == 0:
            return "none_box"
        elif slayout == 1:
            return "row_major"
        else:
            return "col_major"
    
    return str(slayout).lower()


def normalize_memory_space(ms: Any) -> str:
    """Normalize memory_space to string.

    Args:
        ms: Can be enum value, string, or MemorySpace object

    Returns:
        Normalized memory_space string ("ub", "gm", etc.)
    """
    if ms is None:
        return "ub"  # default for tile
    
    if isinstance(ms, str):
        return ms.lower()
    
    # Handle MemorySpace enum from types.py
    if hasattr(ms, "value"):
        return ms.value.lower()

    return str(ms).lower()


def normalize_view_layout(layout: Any) -> str | None:
    """Normalize view layout to string while keeping unknown as None."""
    if layout is None:
        return None
    if isinstance(layout, str):
        return layout.lower()
    if hasattr(layout, "value"):
        value = layout.value
        if value is None:
            return None
        return str(value).lower()
    return str(layout).lower()


def compute_stable_key(
    target: str,
    op: str,
    operand_specs: list[dict],
    context_attrs: dict[str, Any] | None = None,
) -> StableKey:
    """Compute stable key from operand_specs.

    Args:
        target: Target architecture ("a5", "a3")
        op: Operator name ("tadd", "tmul", etc.)
        operand_specs: List of operand spec dicts from PTOAS
        context_attrs: Additional context attributes (round_mode, cmp_mode, etc.)

    Returns:
        StableKey object for caching

    Note:
        operand_specs format (from ExpandTileOp.cpp):
        [
            {
                "kind": "tile",
                "dtype": "f32",
                "shape": [16, 64],
                "valid_shape": [16, 64],
                "memory_space": "ub",
                "config": {
                    "b_layout": "row_major",
                    "s_layout": "none_box",
                    "s_fractal_size": 512,
                    "pad_value": 0
                }
            },
            {
                "kind": "view",
                "dtype": "f32",
                "shape": [16, 64],
                "strides": [64, 1],
                "memory_space": "gm"
            },
            {
                "kind": "scalar",
                "dtype": "f32"
            }
        ]
    """
    operand_keys = []
    
    for spec in operand_specs:
        kind = spec.get("kind", "tile")
        dtype = normalize_dtype(spec.get("dtype"))
        
        if kind == "tile":
            # Tile: all fields enter key
            shape = tuple(spec.get("shape", []))
            valid_shape_raw = spec.get("valid_shape", [])
            valid_shape = tuple(
                None if v is None else int(v) for v in valid_shape_raw
            ) if valid_shape_raw else None
            
            memory_space = normalize_memory_space(spec.get("memory_space"))
            
            config = spec.get("config", {})
            blayout = normalize_blayout(config.get("b_layout"))
            slayout = normalize_slayout(config.get("s_layout"))
            fractal = config.get("s_fractal_size", 0)
            if fractal is None:
                fractal = 0
            
            # Handle pad_value: can be int, string like "0x0", or None
            pad = config.get("pad_value", 0)
            if pad is None:
                pad = 0
            elif isinstance(pad, str):
                # Handle hex string like "0x0"
                if pad.startswith("0x") or pad.startswith("0X"):
                    pad = int(pad, 16)
                else:
                    pad = int(pad)
            
            operand_keys.append(OperandStableKey(
                kind="tile",
                dtype=dtype,
                shape=shape,
                valid_shape=valid_shape,
                memory_space=memory_space,
                blayout=blayout,
                slayout=slayout,
                fractal=int(fractal),
                pad=int(pad),
            ))
        
        elif kind == "view":
            # View: ENHANCED - shape/strides/ms all enter key
            view_shape_raw = spec.get("shape", [])
            view_shape = tuple(
                None if s is None else int(s) for s in view_shape_raw
            ) if view_shape_raw else None
            
            view_strides_raw = spec.get("strides", [])
            view_strides = tuple(
                None if st is None else int(st) for st in view_strides_raw
            ) if view_strides_raw else None
            
            view_memory_space = normalize_memory_space(spec.get("memory_space", "gm"))
            config = spec.get("config") or {}
            if hasattr(config, "fields"):
                config = dict(config.fields)
            view_layout = normalize_view_layout(config.get("layout"))
            
            operand_keys.append(OperandStableKey(
                kind="view",
                dtype=dtype,
                view_shape=view_shape,
                view_strides=view_strides,
                view_memory_space=view_memory_space,
                view_layout=view_layout,
            ))
        
        else:  # scalar
            operand_keys.append(OperandStableKey(
                kind="scalar",
                dtype=dtype,
            ))
    
    # Context attrs: normalize to frozenset of (key, str(value))
    context_key = frozenset(
        (k, str(v)) for k, v in (context_attrs or {}).items()
    )
    
    return StableKey(
        target=target,
        op=op,
        operand_keys=tuple(operand_keys),
        context_key=context_key,
    )
