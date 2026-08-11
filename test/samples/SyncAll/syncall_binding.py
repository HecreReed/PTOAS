# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptoas.mlir.ir import Attribute, Context, InsertionPoint, IntegerType, Location, Module, UnitAttr
from ptoas.mlir.dialects import func, pto


def _mode(name):
    return Attribute.parse(f"#pto.sync_all_mode<{name}>")


def _core_type(name):
    return Attribute.parse(f"#pto.sync_core_type<{name}>")


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            # PTO-ISA uses element 0 as the shared counter but reserves one
            # exclusive 64-byte cache line: 16 x i32.

            fn_ty = func.FunctionType.get([ptr_i32, i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("syncall_binding_kernel", fn_ty)
                fn.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                gm_workspace_ptr, used_cores = entry.arguments
                pto.syncall(
                    _mode("soft"),
                    _core_type("aiv_only"),
                    gm_workspace=gm_workspace_ptr,
                    used_cores=used_cores,
                )
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
