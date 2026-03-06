// RUN: ptoas %S/tadd_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s --check-prefix=TADD
// RUN: ptoas %S/tsub_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s --check-prefix=TSUB
// RUN: ptoas %S/tmul_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s --check-prefix=TMUL
// RUN: ptoas %S/tdiv_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s --check-prefix=TDIV
// RUN: ptoas %S/tmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s --check-prefix=TMAX
// RUN: ptoas %S/tmin_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib_ccec --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s --check-prefix=TMIN

// TADD: [op-fusion] found 1 group(s) in @tadd_block
// TADD-DAG: [op-fusion] imported OP-Lib template: op=tadd
// TADD: [op-fusion] instantiated template: op=tadd dtype=f32
// TADD: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// TADD: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)

// TSUB: [op-fusion] found 1 group(s) in @tsub_block
// TSUB-DAG: [op-fusion] imported OP-Lib template: op=tsub
// TSUB: [op-fusion] instantiated template: op=tsub dtype=f32
// TSUB: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// TSUB: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)

// TMUL: [op-fusion] found 1 group(s) in @tmul_block
// TMUL-DAG: [op-fusion] imported OP-Lib template: op=tmul
// TMUL: [op-fusion] instantiated template: op=tmul dtype=f32
// TMUL: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// TMUL: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)

// TDIV: [op-fusion] found 1 group(s) in @tdiv_block
// TDIV-DAG: [op-fusion] imported OP-Lib template: op=tdiv
// TDIV: [op-fusion] instantiated template: op=tdiv dtype=f32
// TDIV: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// TDIV: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)

// TMAX: [op-fusion] found 1 group(s) in @tmax_block
// TMAX-DAG: [op-fusion] imported OP-Lib template: op=tmax
// TMAX: [op-fusion] instantiated template: op=tmax dtype=f32
// TMAX: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// TMAX: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)

// TMIN: [op-fusion] found 1 group(s) in @tmin_block
// TMIN-DAG: [op-fusion] imported OP-Lib template: op=tmin
// TMIN: [op-fusion] instantiated template: op=tmin dtype=f32
// TMIN: [op-fusion] materialized group_id=0 into @__pto_fused_group_0_0
// TMIN: [op-fusion] instantiate+inline touched 1 fused function(s), inlined 2 call(s)
