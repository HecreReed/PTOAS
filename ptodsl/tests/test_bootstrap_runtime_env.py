#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class BootstrapRuntimeEnvTests(unittest.TestCase):
    def test_bootstrap_keeps_installed_mlir_environment_unchanged(self):
        repo_root = Path(__file__).resolve().parents[2]
        bootstrap_path = repo_root / "ptodsl" / "ptodsl" / "_bootstrap.py"
        module_name = "_test_ptodsl_bootstrap_runtime_env"
        spec = importlib.util.spec_from_file_location(module_name, bootstrap_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        mlir_module = types.ModuleType("mlir")
        mlir_module.__path__ = []
        mlir_dialects_module = types.ModuleType("mlir.dialects")
        mlir_dialects_module.__path__ = []
        mlir_pto_module = types.ModuleType("mlir.dialects.pto")
        mlir_llvm_module = types.ModuleType("mlir.dialects.llvm")
        mlir_ir_module = types.ModuleType("mlir.ir")

        class FakeContext:
            pass

        class FakeLocation:
            pass

        def register_pto_dialect(ctx, load=False):
            ctx.pto_loaded = load

        def register_llvm_dialect(ctx, load=False):
            ctx.llvm_loaded = load

        mlir_ir_module.Context = FakeContext
        mlir_ir_module.Location = FakeLocation
        mlir_pto_module.register_dialect = register_pto_dialect
        mlir_llvm_module.register_dialect = register_llvm_dialect
        mlir_dialects_module.pto = mlir_pto_module
        mlir_dialects_module.llvm = mlir_llvm_module
        mlir_module.dialects = mlir_dialects_module
        mlir_module.ir = mlir_ir_module

        fake_sys_modules = {
            "mlir": mlir_module,
            "mlir.dialects": mlir_dialects_module,
            "mlir.dialects.pto": mlir_pto_module,
            "mlir.dialects.llvm": mlir_llvm_module,
            "mlir.ir": mlir_ir_module,
        }

        required_specs = {
            "mlir.ir": object(),
            "mlir.dialects.pto": object(),
        }
        original_find_spec = importlib.util.find_spec
        original_sys_path = list(sys.path)

        def fake_find_spec(name, package=None):
            if name in required_specs:
                return required_specs[name]
            return original_find_spec(name, package)

        with mock.patch.object(sys, "path", list(original_sys_path)), mock.patch.dict(
            sys.modules, fake_sys_modules, clear=False
        ), mock.patch("importlib.util.find_spec", side_effect=fake_find_spec):
            module = importlib.util.module_from_spec(spec)
            sys.modules.pop(module_name, None)
            try:
                spec.loader.exec_module(module)
                self.assertEqual(sys.path, original_sys_path)
                ctx = module.make_context()
                self.assertTrue(getattr(ctx, "pto_loaded", False))
                self.assertTrue(getattr(ctx, "llvm_loaded", False))
            finally:
                sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
