#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ptoas import _launcher


class WheelLauncherTests(unittest.TestCase):
    def _make_runtime_tree(self, temp_root: Path) -> Path:
        package_root = temp_root / "site-packages" / "ptoas"
        runtime_root = package_root / "_runtime"
        (runtime_root / "bin").mkdir(parents=True, exist_ok=True)
        (runtime_root / "lib").mkdir(parents=True, exist_ok=True)
        (runtime_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
        (temp_root / "site-packages" / "ptodsl").mkdir(parents=True, exist_ok=True)
        (temp_root / "site-packages" / "tilelang_dsl").mkdir(parents=True, exist_ok=True)
        (temp_root / "site-packages" / "mlir").mkdir(parents=True, exist_ok=True)
        (runtime_root / "bin" / "ptoas").write_text("", encoding="utf-8")
        return package_root

    def _make_editable_tree(self, temp_root: Path) -> tuple[Path, Path]:
        repo_root = temp_root / "repo"
        package_root = repo_root / "ptodsl" / "ptoas"
        install_root = repo_root / "install"
        (package_root).mkdir(parents=True, exist_ok=True)
        (install_root / "bin").mkdir(parents=True, exist_ok=True)
        (install_root / "lib").mkdir(parents=True, exist_ok=True)
        (install_root / "share" / "ptoas" / "TileOps").mkdir(parents=True, exist_ok=True)
        (install_root / "tilelang_dsl").mkdir(parents=True, exist_ok=True)
        (install_root / "mlir").mkdir(parents=True, exist_ok=True)
        (install_root / "bin" / "ptoas").write_text("", encoding="utf-8")
        return package_root, install_root

    def test_launcher_exports_runtime_contract_and_injects_default_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root = self._make_runtime_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.dict(_launcher.os.environ, {}, clear=True), mock.patch.object(
                _launcher, "__file__", str(fake_launcher)
            ), mock.patch.object(
                _launcher.sys, "argv", ["ptoas", "--version"]
            ), mock.patch.object(_launcher.os, "execvpe") as execvpe:
                _launcher.main()

            execvpe.assert_called_once()
            binary, argv, env = execvpe.call_args.args
            self.assertEqual(binary, str(package_root / "_runtime" / "bin" / "ptoas"))
            self.assertEqual(argv[:5], [
                str(package_root / "_runtime" / "bin" / "ptoas"),
                "--tilelang-path",
                str(package_root / "_runtime" / "share" / "ptoas" / "TileOps"),
                "--tilelang-pkg-path",
                str(package_root.parent),
            ])
            self.assertEqual(argv[-1], "--version")
            self.assertEqual(env["PTOAS_HOME"], str(package_root / "_runtime"))
            self.assertEqual(env["PTOAS_BIN"], str(package_root / "_runtime" / "bin" / "ptoas"))
            self.assertEqual(
                env["PTOAS_TILEOPS_DIR"],
                str(package_root / "_runtime" / "share" / "ptoas" / "TileOps"),
            )
            self.assertTrue(env["PATH"].split(os.pathsep)[0].endswith("ptoas/_runtime/bin"))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(package_root.parent))

    def test_resolve_runtime_root_defaults_to_repo_install_tree(self):
        package_root = Path("/tmp/repo/ptodsl/ptoas")
        with mock.patch.dict(_launcher.os.environ, {}, clear=True):
            runtime_root = _launcher._resolve_runtime_root(package_root)
        self.assertEqual(runtime_root, Path("/tmp/repo/install"))

    def test_launcher_falls_back_to_env_install_tree_for_editable_installs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root, install_root = self._make_editable_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.dict(
                _launcher.os.environ,
                {"PTO_INSTALL_DIR": str(install_root)},
                clear=True,
            ), mock.patch.object(
                _launcher.sys, "argv", ["ptoas", "--version"]
            ), mock.patch.object(_launcher.os, "execvpe") as execvpe:
                _launcher.main()

            execvpe.assert_called_once()
            binary, argv, env = execvpe.call_args.args
            self.assertEqual(binary, str(install_root / "bin" / "ptoas"))
            self.assertEqual(argv[:5], [
                str(install_root / "bin" / "ptoas"),
                "--tilelang-path",
                str(install_root / "share" / "ptoas" / "TileOps"),
                "--tilelang-pkg-path",
                str(install_root),
            ])
            self.assertEqual(argv[-1], "--version")
            self.assertEqual(env["PTOAS_HOME"], str(install_root))
            self.assertEqual(env["PTOAS_BIN"], str(install_root / "bin" / "ptoas"))
            self.assertEqual(
                env["PTOAS_TILEOPS_DIR"],
                str(install_root / "share" / "ptoas" / "TileOps"),
            )
            self.assertTrue(env["PATH"].split(os.pathsep)[0].endswith("install/bin"))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(install_root))

    def test_launcher_respects_explicit_tilelang_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            package_root = self._make_runtime_tree(temp_root)
            fake_launcher = package_root / "_launcher.py"
            fake_launcher.write_text("", encoding="utf-8")

            with mock.patch.object(_launcher, "__file__", str(fake_launcher)), mock.patch.object(
                _launcher.sys,
                "argv",
                [
                    "ptoas",
                    "--tilelang-path=/tmp/custom-tileops",
                    "--tilelang-pkg-path",
                    "/tmp/custom-python",
                    "--help",
                ],
            ), mock.patch.object(_launcher.os, "execvpe") as execvpe:
                _launcher.main()

            execvpe.assert_called_once()
            _, argv, _ = execvpe.call_args.args
            self.assertEqual(argv, [
                str(package_root / "_runtime" / "bin" / "ptoas"),
                "--tilelang-path=/tmp/custom-tileops",
                "--tilelang-pkg-path",
                "/tmp/custom-python",
                "--help",
            ])


if __name__ == "__main__":
    unittest.main()
