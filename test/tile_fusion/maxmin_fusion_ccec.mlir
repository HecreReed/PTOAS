// RUN: ptoas %S/maxmin_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// CHECK: [op-fusion] found 1 group(s) in @maxmin_block
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tmax
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tmin
// CHECK: [op-fusion] instantiated template: op=tmax dtype=f32
// CHECK: [op-fusion] instantiated template: op=tmin dtype=f32
// CHECK: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// CHECK: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)
// CHECK: [op-fusion] low-level loop fusion changed 1 fused function(s)
