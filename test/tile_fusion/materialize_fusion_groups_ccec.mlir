// RUN: ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tadd
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tsub
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tmul
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tdiv
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tmax
// CHECK-DAG: [op-fusion] imported OP-Lib template: op=tmin
// CHECK: [op-fusion] instantiated template: op=tmul dtype=f32
// CHECK: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// CHECK: [op-fusion] instantiated template: op=tadd dtype=f32
// CHECK: [op-fusion] materialized group_id=1 into @__pto_fused_group_1_1
// CHECK: [op-fusion] instantiate+inline touched 2 fused function(s), inlined 6 call(s)
// CHECK: [op-fusion] low-level loop fusion changed 2 fused function(s)
