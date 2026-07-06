#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    COLS,
    ROWS,
    default_buffers,
    float_values,
    load_case_meta,
    matrix32,
    pack_predicate_mask_for_buffer,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    mask_name, src_name = meta.inputs
    generator = rng()
    mask_bits = generator.integers(0, 2, size=(ROWS, COLS), dtype=np.uint8).astype(np.bool_)
    mask = pack_predicate_mask_for_buffer(
        mask_bits,
        elem_count=meta.elem_counts[mask_name],
        dtype=meta.np_types[mask_name],
        rows=ROWS,
    )
    src = float_values(generator, meta.elem_counts[src_name], style='signed')
    buffers = default_buffers(meta)
    buffers[mask_name] = mask
    buffers[src_name] = src
    write_buffers(meta, buffers)
    out = np.where(mask_bits, matrix32(src), np.float32(64.0))
    write_golden(meta, {single_output(meta): out.astype(np.float32).reshape(-1)})


if __name__ == '__main__':
    main()
