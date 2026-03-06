module {
  func.func @__pto_ccec_tadd_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tadd",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    ccec.vbin kind = "add"
      ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
      outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
    return
  }

  func.func @__pto_ccec_tsub_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tsub",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    ccec.vbin kind = "sub"
      ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
      outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
    return
  }

  func.func @__pto_ccec_tmul_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tmul",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    ccec.vbin kind = "mul"
      ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
      outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
    return
  }

  func.func @__pto_ccec_tdiv_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tdiv",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    ccec.vbin kind = "div"
      ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
      outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
    return
  }

  func.func @__pto_ccec_tmax_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tmax",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    ccec.vbin kind = "max"
      ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
      outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
    return
  }

  func.func @__pto_ccec_tmin_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tmin",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    ccec.vbin kind = "min"
      ins(%src0, %src1 : memref<?x?xf32, #pto.address_space<vec>>, memref<?x?xf32, #pto.address_space<vec>>)
      outs(%dst : memref<?x?xf32, #pto.address_space<vec>>)
    return
  }
}
