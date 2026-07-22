# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Wheel-installed `ptoas` console entry."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NoReturn


def _prepend_env_path(env: dict[str, str], name: str, value: Path) -> None:
    if not value.exists():
        return
    current = env.get(name, "")
    rendered = str(value)
    parts = [part for part in current.split(os.pathsep) if part]
    if rendered in parts:
        parts.remove(rendered)
    parts.insert(0, rendered)
    env[name] = os.pathsep.join(parts)


def _has_cli_option(argv: list[str], option: str) -> bool:
    option_with_value = f"{option}="
    for arg in argv:
        if arg == option or arg.startswith(option_with_value):
            return True
    return False


def _resolve_runtime_root(package_root: Path) -> Path:
    runtime_root = package_root / "_runtime"
    if runtime_root.exists():
        return runtime_root
    env_install_dir = os.environ.get("PTO_INSTALL_DIR")
    if env_install_dir:
        return Path(env_install_dir)
    return package_root.parent.parent / "install"


def main() -> NoReturn:
    package_root = Path(__file__).resolve().parent
    runtime_root = _resolve_runtime_root(package_root)
    binary = runtime_root / "bin" / "ptoas"
    if not binary.is_file():
        raise SystemExit(
            f"wheel runtime is missing the packaged ptoas binary: {binary}"
        )

    python_root = package_root.parent if runtime_root.name == "_runtime" else runtime_root
    tileops_dir = runtime_root / "share" / "ptoas" / "TileOps"
    env = os.environ.copy()
    env["PTOAS_HOME"] = str(runtime_root)
    env["PTOAS_BIN"] = str(binary)
    env["PTOAS_TILEOPS_DIR"] = str(tileops_dir)

    _prepend_env_path(env, "PATH", binary.parent)
    _prepend_env_path(env, "PYTHONPATH", python_root)
    _prepend_env_path(env, "LD_LIBRARY_PATH", runtime_root / "lib")
    _prepend_env_path(env, "DYLD_LIBRARY_PATH", runtime_root / "lib")

    argv = [str(binary)]
    user_args = sys.argv[1:]
    if not _has_cli_option(user_args, "--tilelang-path"):
        argv.extend(["--tilelang-path", str(tileops_dir)])
    if not _has_cli_option(user_args, "--tilelang-pkg-path"):
        argv.extend(["--tilelang-pkg-path", str(python_root)])
    argv.extend(user_args)
    os.execvpe(str(binary), argv, env)


if __name__ == "__main__":
    main()
