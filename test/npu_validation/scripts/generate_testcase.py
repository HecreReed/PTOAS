#!/usr/bin/env python3
# coding=utf-8

import argparse
import re
from pathlib import Path
from typing import Optional

INCLUDE_REPLACEMENT = (
    "// ---------------------------------------------------------------------------\n"
    "// PTOAS compatibility layer\n"
    "//\n"
    "// The upstream pto-isa headers reference some FP8/FP4 types and the\n"
    "// __VEC_SCOPE__ marker that are not available on every AICore arch/toolchain\n"
    "// combination (e.g. __NPU_ARCH__==2201).\n"
    "//\n"
    "// For our PTOAS-generated kernels we don't rely on these types today, but the\n"
    "// headers still mention them in templates/static_asserts. Provide minimal\n"
    "// fallbacks to keep compilation working on dav-c220.\n"
    "// ---------------------------------------------------------------------------\n"
    "#ifndef __VEC_SCOPE__\n"
    "#define __VEC_SCOPE__\n"
    "#endif\n"
    "\n"
    "#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)\n"
    "typedef struct { unsigned char v; } hifloat8_t;\n"
    "typedef struct { unsigned char v; } float8_e4m3_t;\n"
    "typedef struct { unsigned char v; } float8_e5m2_t;\n"
    "typedef struct { unsigned char v; } float8_e8m0_t;\n"
    "typedef struct { unsigned char v; } float4_e1m2x2_t;\n"
    "typedef struct { unsigned char v; } float4_e2m1x2_t;\n"
    "#endif\n"
    "\n"
    "#include <pto/pto-inst.hpp>\n"
    "#include <pto/common/constants.hpp>\n"
    "#include \"acl/acl.h\"\n"
)


def _parse_shape(text: str):
    match = re.search(r"Shape<(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"Shape<\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*,\s*(\d+)>", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 32, 32


def _is_gm_pointer_param(param: str) -> bool:
    return "__gm__" in param and "*" in param


def _extract_cpp_type(param: str) -> str:
    match = re.search(r"__gm__\s+([A-Za-z_]\w*)", param)
    if match:
        return match.group(1)

    tokens = param.replace("*", " ").strip().split()
    if not tokens:
        return "float"
    if len(tokens) == 1:
        return tokens[0]
    qualifiers = {"const", "volatile", "restrict", "__restrict", "__restrict__"}
    type_tokens = [t for t in tokens[:-1] if t not in qualifiers]
    return " ".join(type_tokens) if type_tokens else tokens[0]


def _extract_cpp_name(param: str) -> str:
    parts = param.strip().split()
    if not parts:
        return "arg"
    name = parts[-1].replace("*", "").strip()
    if name.startswith("__"):
        return "arg"
    return name


def _strip_param_name(raw: str, name: str) -> str:
    """
    Return the type part of a parameter declaration, keeping qualifiers and the
    pointer '*' but removing the trailing variable name.
    Example: "__gm__ float* v1" -> "__gm__ float*"
    """
    pattern = rf"\b{re.escape(name)}\b\s*$"
    stripped = re.sub(pattern, "", raw.strip())
    return stripped.strip()


def _infer_void_gm_pointee_type(text: str, param_name: str) -> Optional[str]:
    # Common patterns in PTOAS-generated kernels:
    #   __gm__ int16_t* v16 = (__gm__ int16_t*) v1;
    #   __gm__ half*   v16 = (__gm__ half*) v1;
    name = re.escape(param_name)
    patterns = [
        # Direct assignment after implicit conversion (some kernels keep the
        # ABI as `void*` and only materialize the real type for arithmetic).
        rf"__gm__\s+([A-Za-z_]\w*)\s*\*\s*\w+\s*=\s*{name}\b",
        rf"\(__gm__\s+([A-Za-z_]\w*)\s*\*\)\s*{name}\b",
        rf"reinterpret_cast<__gm__\s+([A-Za-z_]\w*)\s*\*\s*>\(\s*{name}\s*\)",
        rf"static_cast<__gm__\s+([A-Za-z_]\w*)\s*\*\s*>\(\s*{name}\s*\)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            ty = match.group(1)
            if ty and ty != "void":
                return ty
    return None


def _detect_output_pointer_param(text: str, pointer_param_names):
    if not pointer_param_names:
        return None

    tstore_gts = re.findall(r"\bTSTORE\s*\(\s*(\w+)\s*,", text)
    if not tstore_gts:
        return None

    gt_to_ptr = {}
    for m in re.finditer(r"\b(\w+)\s*=\s*[\w:<>]+\s*\(\s*(\w+)\s*[,)]", text):
        gt_to_ptr[m.group(1)] = m.group(2)

    ptr_to_base = {}
    for m in re.finditer(r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*\+", text):
        ptr_to_base[m.group(1)] = m.group(2)

    ptr_to_param = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b",
        text,
    ):
        ptr_to_param[m.group(1)] = m.group(2)

    def resolve_param(ptr: Optional[str]) -> Optional[str]:
        if not ptr:
            return None
        cur = ptr
        seen = set()
        for _ in range(8):
            if cur in seen:
                break
            seen.add(cur)
            if cur in pointer_param_names:
                return cur
            mapped = ptr_to_param.get(cur)
            if mapped in pointer_param_names:
                return mapped
            cur = ptr_to_base.get(cur)
            if cur is None:
                break
        return None

    for gt in tstore_gts:
        ptr = gt_to_ptr.get(gt)
        if not ptr:
            continue
        resolved = resolve_param(ptr)
        if resolved:
            return resolved
    return None


def _parse_kernel_params(text: str):
    match = re.search(r"__global__\s+(?:\w+\s+)*void\s+\w+\s*\(([^)]*)\)", text, re.S)
    if not match:
        return []
    params_blob = match.group(1).strip()
    if not params_blob:
        return []
    params = []
    depth = 0
    start = 0
    for idx, ch in enumerate(params_blob):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            params.append(params_blob[start:idx].strip())
            start = idx + 1
    last = params_blob[start:].strip()
    if last:
        params.append(last)
    return params


def _parse_kernel_name(text: str) -> str:
    match = re.search(r"__global__\s+(?:\w+\s+)*void\s+(\w+)\s*\(", text, re.S)
    return match.group(1) if match else "kernel"


def _np_dtype_for_cpp(cpp_type: str) -> str:
    mapping = {
        "float": "np.float32",
        "half": "np.float16",
        "aclFloat16": "np.float16",
        "__bf16": "np.uint16",
        "bfloat16_t": "np.uint16",
        "int8_t": "np.int8",
        "uint8_t": "np.uint8",
        "int16_t": "np.int16",
        "uint16_t": "np.uint16",
        "int32_t": "np.int32",
        "uint32_t": "np.uint32",
    }
    return mapping.get(cpp_type, "np.float32")


def _cpp_host_type(cpp_type: str) -> str:
    if cpp_type == "half":
        return "aclFloat16"
    if cpp_type in {"__bf16", "bfloat16_t"}:
        return "uint16_t"
    return cpp_type


def _derive_testcase_name(input_cpp: Path) -> str:
    name = input_cpp.stem
    if name.endswith("-pto"):
        name = name[:-4]
    if name.endswith("_pto"):
        name = name[:-4]
    return name


def _replace_includes(text: str) -> str:
    if "#include \"common/pto_instr.hpp\"" in text:
        return text.replace("#include \"common/pto_instr.hpp\"", INCLUDE_REPLACEMENT.rstrip())
    if "#include <pto/pto-inst.hpp>" in text:
        return text
    return INCLUDE_REPLACEMENT + "\n" + text


def _infer_aicore_arch(kernel_text: str, soc_version: str) -> str:
    # Heuristic: kernels that touch cube/L0/L1 tile types or cbuf memories need
    # the "cube" arch; pure vector kernels can use the vector arch.
    #
    # IMPORTANT: the default arch depends on the Ascend SoC.
    cube_markers = (
        "TileType::Left",
        "TileType::Right",
        "TileType::Acc",
        "TileType::Mat",
        "__cbuf__",
        "__ca__",
        "__cb__",
        "__cc__",
        "copy_gm_to_cbuf",
        "copy_cbuf_to_gm",
        "mad(",
        "mmad(",
        "TMMAD",
    )
    needs_cube = any(m in kernel_text for m in cube_markers)

    sv = (soc_version or "").lower()
    if "910b" in sv:
        # Ascend910B* (e.g. Ascend910B1) uses dav-c310 toolchain arch.
        return "dav-c310-cube" if needs_cube else "dav-c310-vec"

    # Default to Ascend910 (dav-c220) when SoC is unknown.
    return "dav-c220-cube" if needs_cube else "dav-c220-vec"


def _parse_int_list(blob: str):
    items = []
    for part in blob.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p, 0))
        except ValueError:
            return None
    return items


def _infer_mrgsort_block_len(kernel_text: str) -> Optional[int]:
    """
    Try to infer the compile-time blockLen argument passed to:
        TMRGSORT(dst, src, blockLen)

    Most PTOAS-generated kernels use a constant like:
        int32_t v3 = 64;
        TMRGSORT(v22, v21, v3);
    """
    call = re.search(r"\bTMRGSORT\s*\(\s*\w+\s*,\s*\w+\s*,\s*([^)]+?)\s*\)", kernel_text)
    if not call:
        return None
    arg = call.group(1).strip()
    # Direct literal.
    if re.fullmatch(r"(?:0x[0-9A-Fa-f]+|\d+)", arg):
        try:
            return int(arg, 0)
        except ValueError:
            return None

    # Identifier that is defined as a constant earlier in the kernel.
    if not re.fullmatch(r"[A-Za-z_]\w*", arg):
        return None
    match = re.search(rf"\b(?:int32_t|uint32_t|int|unsigned)\s+{re.escape(arg)}\s*=\s*(0x[0-9A-Fa-f]+|\d+)\s*;", kernel_text)
    if not match:
        return None
    try:
        return int(match.group(1), 0)
    except ValueError:
        return None


def _required_elements_for_shape_stride(shape_dims, stride_dims) -> Optional[int]:
    if not shape_dims or not stride_dims:
        return None
    n = min(len(shape_dims), len(stride_dims))
    req = 1
    for i in range(n):
        dim = shape_dims[i]
        stride = stride_dims[i]
        if not isinstance(dim, int) or not isinstance(stride, int):
            return None
        if dim <= 0:
            continue
        req += (dim - 1) * stride
    return max(req, 1)


def _infer_gm_pointer_elem_counts(kernel_text: str, pointer_param_names):
    """
    Infer minimum element counts for each __gm__ pointer param from GlobalTensor
    shape/stride metadata found in PTOAS-generated kernels.

    This fixes cases where the logical shape is small (e.g. 32x32) but the GM
    tensor uses padded strides (e.g. row stride 256), so the kernel accesses a
    much larger linear range.
    """
    if not pointer_param_names:
        return {}

    pointer_params = set(pointer_param_names)

    ptr_to_base = {}
    for m in re.finditer(r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*(\w+)\s*\+", kernel_text):
        ptr_to_base[m.group(1)] = m.group(2)

    ptr_to_param = {}
    for m in re.finditer(
        r"__gm__\s+[\w:<>]+\s*\*\s*(\w+)\s*=\s*\(__gm__\s+[\w:<>]+\s*\*\)\s*(\w+)\b",
        kernel_text,
    ):
        ptr_to_param[m.group(1)] = m.group(2)

    def resolve_param(ptr: str) -> Optional[str]:
        cur = ptr
        seen = set()
        for _ in range(16):
            if cur in pointer_params:
                return cur
            if cur in seen:
                break
            seen.add(cur)
            mapped = ptr_to_param.get(cur)
            if mapped:
                cur = mapped
                continue
            base = ptr_to_base.get(cur)
            if base:
                cur = base
                continue
            break
        return None

    # Parse aliases: GTShape_*=pto::Shape<...>; GTStride_*=pto::Stride<...>;
    shape_aliases = {}
    for m in re.finditer(r"using\s+(\w+)\s*=\s*pto::Shape<([^>]*)>;", kernel_text):
        dims = _parse_int_list(m.group(2))
        if dims:
            shape_aliases[m.group(1)] = dims

    stride_aliases = {}
    for m in re.finditer(r"using\s+(\w+)\s*=\s*pto::Stride<([^>]*)>;", kernel_text):
        dims = _parse_int_list(m.group(2))
        if dims:
            stride_aliases[m.group(1)] = dims

    # Map GT_* alias -> (shape_alias, stride_alias)
    gt_alias_to_shape_stride = {}
    for m in re.finditer(
        # Matches both:
        #   using GT = GlobalTensor<T, ShapeAlias, StrideAlias>;
        # and the 4-param layout form:
        #   using GT = GlobalTensor<T, ShapeAlias, StrideAlias, LayoutAlias>;
        r"using\s+(\w+)\s*=\s*GlobalTensor<\s*[^,>]+\s*,\s*(\w+)\s*,\s*(\w+)\s*(?:,\s*[^>]+)?\s*>;",
        kernel_text,
    ):
        gt_alias = m.group(1)
        shape_alias = m.group(2)
        stride_alias = m.group(3)
        gt_alias_to_shape_stride[gt_alias] = (shape_alias, stride_alias)

    # Find instantiations: GT_xxx v = GT_xxx(ptr, ...)
    param_elem_counts = {}
    for m in re.finditer(r"\b(\w+)\s+\w+\s*=\s*\1\s*\(\s*(\w+)\s*,", kernel_text):
        gt_alias = m.group(1)
        base_ptr = m.group(2)
        shape_stride = gt_alias_to_shape_stride.get(gt_alias)
        if not shape_stride:
            continue
        shape_dims = shape_aliases.get(shape_stride[0])
        stride_dims = stride_aliases.get(shape_stride[1])
        req = _required_elements_for_shape_stride(shape_dims, stride_dims)
        if not req:
            continue
        param = resolve_param(base_ptr)
        if not param:
            continue
        param_elem_counts[param] = max(param_elem_counts.get(param, 0), req)

    return param_elem_counts


def generate_testcase(
    input_cpp: Path,
    output_root: Optional[Path],
    testcase: str,
    run_mode: str,
    soc_version: str,
    aicore_arch: Optional[str] = None,
):
    sample_dir = input_cpp.parent
    if output_root:
        output_dir = output_root / sample_dir.name / testcase
    else:
        output_dir = sample_dir / "npu_validation" / testcase
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_kernel = input_cpp.read_text(encoding="utf-8")
    aicore_arch = aicore_arch or _infer_aicore_arch(raw_kernel, soc_version)
    kernel_out = output_dir / f"{testcase}_kernel.cpp"
    kernel_out.write_text(_replace_includes(raw_kernel), encoding="utf-8")

    rows, cols = _parse_shape(raw_kernel)
    logical_elem_count = rows * cols
    kernel_name = _parse_kernel_name(raw_kernel)
    raw_params = _parse_kernel_params(raw_kernel)
    mrgsort_block_len = _infer_mrgsort_block_len(raw_kernel) if "TMRGSORT" in raw_kernel else None

    pointer_param_names = [_extract_cpp_name(p) for p in raw_params if _is_gm_pointer_param(p)]
    inferred_void_ptr_types = {}
    for raw in raw_params:
        if not _is_gm_pointer_param(raw):
            continue
        name = _extract_cpp_name(raw)
        cpp_type = _extract_cpp_type(raw)
        if cpp_type == "void":
            inferred = _infer_void_gm_pointee_type(raw_kernel, name)
            if inferred:
                inferred_void_ptr_types[name] = inferred

    output_ptr = _detect_output_pointer_param(raw_kernel, pointer_param_names)
    if output_ptr is None and pointer_param_names:
        output_ptr = pointer_param_names[0] if len(pointer_param_names) == 1 else pointer_param_names[-1]

    params = []
    for raw in raw_params:
        name = _extract_cpp_name(raw)
        cpp_type = _extract_cpp_type(raw)
        if cpp_type == "void" and name in inferred_void_ptr_types:
            cpp_type = inferred_void_ptr_types[name]
        if _is_gm_pointer_param(raw):
            params.append(
                {
                    "kind": "ptr",
                    "raw": raw,
                    "name": name,
                    "cpp_type": cpp_type,
                    "host_type": _cpp_host_type(cpp_type),
                    "role": "output" if name == output_ptr else "input",
                }
            )
        else:
            params.append(
                {
                    "kind": "scalar",
                    "raw": raw,
                    "name": name,
                    "cpp_type": cpp_type,
                    "host_type": _cpp_host_type(cpp_type),
                }
            )

    input_ptrs = [p for p in params if p["kind"] == "ptr" and p["role"] == "input"]
    output_ptrs = [p for p in params if p["kind"] == "ptr" and p["role"] == "output"]

    ptr_elem_counts = {p["name"]: logical_elem_count for p in params if p["kind"] == "ptr"}
    inferred_counts = _infer_gm_pointer_elem_counts(raw_kernel, pointer_param_names)
    for name, cnt in inferred_counts.items():
        ptr_elem_counts[name] = max(ptr_elem_counts.get(name, logical_elem_count), cnt)

    templates_root = Path(__file__).resolve().parents[1] / "templates"
    template = (templates_root / "main_template.cpp").read_text(encoding="utf-8")
    case_name = f"case_{rows}x{cols}"

    launch_name = f"Launch{kernel_name[0].upper()}{kernel_name[1:]}"

    launch_decl_params = []
    launch_call_args = []
    for p in params:
        if p["kind"] == "ptr":
            launch_decl_params.append(f"{p['host_type']} *{p['name']}")
            launch_call_args.append(f"{p['name']}Device")
        else:
            launch_decl_params.append(f"{p['host_type']} {p['name']}")
            launch_call_args.append(p["name"])

    param_decls_lines = []
    if any(p["kind"] == "ptr" for p in params):
        for p in params:
            if p["kind"] != "ptr":
                continue
            elem_cnt = ptr_elem_counts.get(p["name"], logical_elem_count)
            param_decls_lines.append(f"    size_t elemCount_{p['name']} = {elem_cnt};")
            param_decls_lines.append(
                f"    size_t fileSize_{p['name']} = elemCount_{p['name']} * sizeof({p['host_type']});"
            )

    for p in params:
        if p["kind"] != "scalar":
            continue
        t = p["host_type"]
        if t == "bool":
            value = "true"
        elif re.match(r"^(u?int)(8|16|32|64)_t$", t) or t in {"int", "unsigned", "size_t"}:
            value = "1"
        elif t in {"float"}:
            value = "1.0f"
        elif t in {"double"}:
            value = "1.0"
        else:
            value = "0"
        param_decls_lines.append(f"    {t} {p['name']} = {value};")

    for p in params:
        if p["kind"] != "ptr":
            continue
        param_decls_lines.append(f"    {p['host_type']} *{p['name']}Host = nullptr;")
        param_decls_lines.append(f"    {p['host_type']} *{p['name']}Device = nullptr;")

    alloc_host = []
    alloc_device = []
    free_host = []
    free_device = []
    for p in params:
        if p["kind"] != "ptr":
            continue
        size_var = f"fileSize_{p['name']}"
        alloc_host.append(
            f"    ACL_CHECK(aclrtMallocHost((void **)(&{p['name']}Host), {size_var}));"
        )
        alloc_device.append(
            f"    ACL_CHECK(aclrtMalloc((void **)&{p['name']}Device, {size_var}, ACL_MEM_MALLOC_HUGE_FIRST));"
        )
        free_device.append(f"    aclrtFree({p['name']}Device);")
        free_host.append(f"    aclrtFreeHost({p['name']}Host);")

    read_inputs = []
    copy_inputs = []
    for p in input_ptrs:
        size_var = f"fileSize_{p['name']}"
        read_inputs.append(
            f"    ReadFile(\"./{p['name']}.bin\", {size_var}, {p['name']}Host, {size_var});"
        )
        copy_inputs.append(
            f"    ACL_CHECK(aclrtMemcpy({p['name']}Device, {size_var}, {p['name']}Host, {size_var}, ACL_MEMCPY_HOST_TO_DEVICE));"
        )

    output_copy_back = []
    output_write = []
    for p in output_ptrs:
        size_var = f"fileSize_{p['name']}"
        output_copy_back.append(
            f"    ACL_CHECK(aclrtMemcpy({p['name']}Host, {size_var}, {p['name']}Device, {size_var}, ACL_MEMCPY_DEVICE_TO_HOST));"
        )
        output_write.append(
            f"    WriteFile(\"./{p['name']}.bin\", {p['name']}Host, {size_var});"
        )

    param_decls = "\n".join(param_decls_lines)
    main_cpp = (
        template
        .replace("@TEST_SUITE@", testcase.upper())
        .replace("@CASE_NAME@", case_name)
        .replace(
            "@LAUNCH_DECL@",
            f"void {launch_name}({', '.join(launch_decl_params + ['void *stream'])});",
        )
        .replace("@PARAM_DECLS@", param_decls)
        .replace("@ALLOC_HOST@", "\n".join(alloc_host))
        .replace("@ALLOC_DEVICE@", "\n".join(alloc_device))
        .replace("@READ_INPUTS@", "\n".join(read_inputs))
        .replace("@COPY_TO_DEVICE@", "\n".join(copy_inputs))
        .replace(
            "@LAUNCH_CALL@",
            f"    {launch_name}({', '.join(launch_call_args + ['stream'])});",
        )
        .replace("@COPY_BACK@", "\n".join(output_copy_back))
        .replace("@WRITE_OUTPUT@", "\n".join(output_write))
        .replace("@FREE_DEVICE@", "\n".join(free_device))
        .replace("@FREE_HOST@", "\n".join(free_host))
    )
    (output_dir / "main.cpp").write_text(main_cpp, encoding="utf-8")

    golden_template = (templates_root / "golden_template.py").read_text(encoding="utf-8")
    input_generate = []
    input_names = []
    elem_count = logical_elem_count
    # Some kernels use an integer tensor as "indices". The safe in-range domain
    # depends on the op semantics:
    # - TSCATTER: indices are row indices in [0, rows)
    # - TGATHER/TGATHERB: indices are linear indices in [0, rows*cols)
    index_mod = None
    if "TSCATTER" in raw_kernel:
        index_mod = max(rows, 1)
    elif any(m in raw_kernel for m in ("TGATHER", "TGATHERB")):
        index_mod = max(elem_count, 1)
    mrgsort_packed = "TMRGSORT" in raw_kernel
    if mrgsort_packed and not mrgsort_block_len:
        mrgsort_block_len = 64
    for p in input_ptrs:
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        name = p["name"]
        size = ptr_elem_counts.get(name, elem_count)
        if mrgsort_packed and np_dtype in ("np.float32", "np.float16"):
            input_generate.append(f"    # TMRGSORT expects packed (value, index) structures (8 bytes each).")
            input_generate.append(f"    # Generate per-block sorted inputs to match pto-isa ST data layout.")
            if np_dtype == "np.float32":
                input_generate.append(f"    {name}__words_per_struct = 2  # float32(4B) + uint32(4B)")
                input_generate.append(f"    {name}__struct_dtype = np.dtype([('v', np.float32), ('i', np.uint32)])")
                input_generate.append(f"    {name}__value_dtype = np.float32")
            else:
                input_generate.append(f"    {name}__words_per_struct = 4  # float16(2B) + pad(2B) + uint32(4B)")
                input_generate.append(
                    f"    {name}__struct_dtype = np.dtype([('v', np.float16), ('pad', np.uint16), ('i', np.uint32)])"
                )
                input_generate.append(f"    {name}__value_dtype = np.float16")

            input_generate.append(f"    {name}__struct_count = {size} // {name}__words_per_struct")
            input_generate.append(f"    {name}__block_len = {mrgsort_block_len}")
            input_generate.append(f"    {name}__structs_per_block = {name}__block_len // {name}__words_per_struct")
            input_generate.append(
                f"    {name}__values = np.random.uniform(low=0, high=1, size=({name}__struct_count,)).astype({name}__value_dtype)"
            )
            input_generate.append(f"    {name}__idx = np.arange({name}__struct_count, dtype=np.uint32)")
            input_generate.append(f"    if {name}__structs_per_block > 0 and {name}__struct_count > 0:")
            input_generate.append(f"        pad = (-{name}__struct_count) % {name}__structs_per_block")
            input_generate.append(f"        if pad:")
            input_generate.append(
                f"            {name}__values = np.concatenate(({name}__values, np.zeros(pad, dtype={name}__values.dtype)))"
            )
            input_generate.append(
                f"            {name}__idx = np.concatenate(({name}__idx, np.zeros(pad, dtype={name}__idx.dtype)))"
            )
            input_generate.append(f"        v = {name}__values.reshape(-1, {name}__structs_per_block)")
            input_generate.append(f"        i = {name}__idx.reshape(-1, {name}__structs_per_block)")
            input_generate.append(f"        order = np.argsort(-v, kind='stable', axis=1)")
            input_generate.append(
                f"        {name}__values = np.take_along_axis(v, order, axis=1).reshape(-1)[:{name}__struct_count]"
            )
            input_generate.append(
                f"        {name}__idx = np.take_along_axis(i, order, axis=1).reshape(-1)[:{name}__struct_count]"
            )
            input_generate.append(f"    {name}__packed = np.empty(({name}__struct_count,), dtype={name}__struct_dtype)")
            input_generate.append(f"    {name}__packed['v'] = {name}__values")
            if np_dtype == "np.float16":
                input_generate.append(f"    {name}__packed['pad'] = np.uint16(0)")
            input_generate.append(f"    {name}__packed['i'] = {name}__idx")
            input_generate.append(f"    {name}__packed.tofile(\"{name}.bin\")")
            # Dummy ndarray for the golden function signature (not used today).
            input_generate.append(f"    {name} = np.zeros(({size},), dtype={np_dtype})")
        elif np_dtype.startswith("np.int") or np_dtype.startswith("np.uint"):
            if index_mod is not None:
                input_generate.append(
                    f"    {name} = (np.arange({size}, dtype=np.int64) % {index_mod}).astype({np_dtype})"
                )
            else:
                input_generate.append(f"    {name} = np.zeros(({size},), dtype={np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        else:
            input_generate.append(f"    {name} = np.random.random(size=({size},)).astype({np_dtype})")
            input_generate.append(f"    {name}.tofile(\"{name}.bin\")")
        input_names.append(name)

    golden_outputs = []
    output_writes = []
    for idx, p in enumerate(output_ptrs):
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        size = ptr_elem_counts.get(p["name"], elem_count)
        golden_outputs.append(f"    outputs.append(np.zeros({size}, dtype={np_dtype}))")
        output_writes.append(f"    outputs[{idx}].tofile(\"golden_{p['name']}.bin\")")

    golden_py = (
        golden_template
        .replace("@GOLDEN_ARGS@", ", ".join(input_names))
        .replace("@GOLDEN_CALL_ARGS@", ", ".join(input_names))
        .replace("@GOLDEN_RET@", "outputs")
        .replace("@INPUT_GENERATE@", "\n".join(input_generate))
        .replace("@GOLDEN_OUTPUTS@", "\n".join(golden_outputs))
        .replace("@OUTPUT_WRITES@", "\n".join(output_writes))
    )
    (output_dir / "golden.py").write_text(golden_py, encoding="utf-8")

    launch_fn_params = ", ".join(launch_decl_params + ["void *stream"])
    kernel_call_args = []
    for p in params:
        if p["kind"] == "ptr":
            kernel_call_args.append(f"({_strip_param_name(p['raw'], p['name'])}){p['name']}")
        else:
            kernel_call_args.append(p["name"])
    kernel_call_args = ", ".join(kernel_call_args)
    launch_cpp = (
        INCLUDE_REPLACEMENT
        + "\n"
        f"__global__ AICORE void {kernel_name}({', '.join(raw_params)});\n\n"
        f"void {launch_name}({launch_fn_params}) {{\n"
        f"    {kernel_name}<<<1, nullptr, stream>>>({kernel_call_args});\n"
        f"}}\n"
    )
    (output_dir / "launch.cpp").write_text(launch_cpp, encoding="utf-8")

    mem_base_define = "MEMORY_BASE"
    if "910b" in (soc_version or "").lower():
        mem_base_define = "REGISTER_BASE"

    cce_stack_size_opt = ""
    # `-mllvm -cce-aicore-stack-size=...` is rejected on some targets (e.g.
    # dav-l310 / dav-l311).
    if not aicore_arch.startswith(("dav-l310", "dav-l311")):
        cce_stack_size_opt = '    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"\n'

    cmake_content = f"""
cmake_minimum_required(VERSION 3.16)

# Prefer setting compilers before project() so CMake picks up bisheng correctly.
set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

project({testcase}_npu_validation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE npu)
endif()
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend910)
endif()

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please source the CANN set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
endif()

set(PTO_ISA_ROOT "" CACHE PATH "Path to pto-isa repo")
if(NOT PTO_ISA_ROOT)
    set(_PTO_ISA_CANDIDATES
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../pto-isa"
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../pto-isa"
        "${{CMAKE_CURRENT_LIST_DIR}}/../../../../../../pto-isa"
    )
    foreach(_cand IN LISTS _PTO_ISA_CANDIDATES)
        if(EXISTS "${{_cand}}/include" AND EXISTS "${{_cand}}/tests/common")
            set(PTO_ISA_ROOT "${{_cand}}" CACHE PATH "Path to pto-isa repo" FORCE)
            break()
        endif()
    endforeach()
endif()
if(NOT PTO_ISA_ROOT)
    message(FATAL_ERROR "Cannot find PTO_ISA_ROOT, please pass -DPTO_ISA_ROOT=/path/to/pto-isa.")
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
    -fPIC
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -fenable-matrix
    --cce-aicore-enable-tl
    -fPIC
    -Xhost-start -Xhost-end
{cce_stack_size_opt}\
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/tests/common
    ${{ASCEND_HOME_PATH}}/include
    ${{ASCEND_DRIVER_PATH}}/kernel/inc
)

	add_library({testcase}_kernel SHARED {testcase}_kernel.cpp launch.cpp)
	target_compile_options({testcase}_kernel PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} --cce-aicore-arch={aicore_arch} -D{mem_base_define} -std=c++17)
	target_include_directories({testcase}_kernel PRIVATE
	    ${{ASCEND_HOME_PATH}}/pkg_inc/
	    ${{ASCEND_HOME_PATH}}/pkg_inc/profiling/
	    ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
	)
target_link_options({testcase}_kernel PRIVATE --cce-fatobj-link)

add_executable({testcase} main.cpp)
target_compile_options({testcase} PRIVATE ${{CMAKE_CPP_COMPILE_OPTIONS}})
target_include_directories({testcase} PRIVATE
    ${{PTO_ISA_ROOT}}/include
    ${{PTO_ISA_ROOT}}/tests/common
)

target_link_directories({testcase} PUBLIC
    ${{ASCEND_HOME_PATH}}/lib64
    ${{ASCEND_HOME_PATH}}/aarch64-linux/simulator/${{SOC_VERSION}}/lib
    ${{ASCEND_HOME_PATH}}/simulator/${{SOC_VERSION}}/lib
    ${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib
)

target_link_libraries({testcase} PRIVATE
    {testcase}_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase
)
"""
    (output_dir / "CMakeLists.txt").write_text(cmake_content.strip() + "\n", encoding="utf-8")

    compare_template = (templates_root / "compare_template.py").read_text(encoding="utf-8")
    compare_lines = ["    ok = True"]
    for p in output_ptrs:
        np_dtype = _np_dtype_for_cpp(p["cpp_type"])
        name = p["name"]
        compare_lines.append(
            f"    ok = compare_bin(\"golden_{name}.bin\", \"{name}.bin\", {np_dtype}, 0.0) and ok"
        )
    compare_py = compare_template.replace("@COMPARES@", "\n".join(compare_lines))
    (output_dir / "compare.py").write_text(compare_py, encoding="utf-8")

    run_sh = (templates_root / "run_sh_template.sh").read_text(encoding="utf-8")
    run_sh = run_sh.replace("@EXECUTABLE@", testcase)
    run_sh = run_sh.replace("@RUN_MODE@", run_mode)
    run_sh = run_sh.replace("@SOC_VERSION@", soc_version)
    run_path = output_dir / "run.sh"
    run_path.write_text(run_sh, encoding="utf-8")
    run_path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description="Generate NPU validation testcase from PTOAS kernel.")
    parser.add_argument("--input", required=True, help="Input PTOAS .cpp file")
    parser.add_argument("--testcase", default=None, help="Testcase name (default: derived from input filename)")
    parser.add_argument("--output-root", default=None, help="Output testcases root directory")
    parser.add_argument("--run-mode", default="npu", choices=["sim", "npu"], help="Run mode for run.sh")
    parser.add_argument("--soc-version", default="Ascend910", help="SOC version for run.sh")
    parser.add_argument(
        "--aicore-arch",
        default=None,
        help="Override AICore arch passed to bisheng (e.g. dav-c220-vec|dav-c220-cube|dav-c310-vec|dav-c310-cube)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else None
    testcase = args.testcase or _derive_testcase_name(Path(args.input))
    generate_testcase(
        Path(args.input),
        output_root,
        testcase,
        args.run_mode,
        args.soc_version,
        aicore_arch=args.aicore_arch,
    )


if __name__ == "__main__":
    main()
