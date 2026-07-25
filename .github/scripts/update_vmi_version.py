#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import pathlib
import re
import sys


VMI_VERSION_RE = re.compile(
    r'(set\s*\(\s*PTOAS_VMI_VERSION\s+")([0-9]+\.[0-9]+\.[0-9]+)("\s*\))'
)
TAG_VERSION_RE = re.compile(r"^vmi-v([0-9]+)\.([0-9]+)\.([0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the VMI version in the top-level CMakeLists.txt."
    )
    parser.add_argument(
        "--cmake-file",
        default="CMakeLists.txt",
        help="Path to the top-level CMakeLists.txt file.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Released VMI version, for example vmi-v0.1.3.",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="Advance the resolved version by one patch step.",
    )
    return parser.parse_args()


def normalize_version(version: str) -> str:
    match = TAG_VERSION_RE.fullmatch(version.strip())
    if not match:
        raise ValueError(f"invalid VMI release tag '{version}'")
    return ".".join(match.groups())


def bump_version(version: str) -> str:
    major, minor, patch = (int(part) for part in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def update_version(cmake_file: pathlib.Path, version: str) -> bool:
    content = cmake_file.read_text(encoding="utf-8")
    updated, count = VMI_VERSION_RE.subn(
        lambda match: f"{match.group(1)}{version}{match.group(3)}",
        content,
        count=1,
    )
    if count != 1:
        raise ValueError(
            f'could not find \'set(PTOAS_VMI_VERSION "x.y.z")\' in {cmake_file}'
        )
    if updated == content:
        return False
    cmake_file.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    args = parse_args()
    version = normalize_version(args.version)
    if args.next:
        version = bump_version(version)
    update_version(pathlib.Path(args.cmake_file), version)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
