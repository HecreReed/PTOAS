from mlir.ir import Context, Location, Module, InsertionPoint, Attribute, IntegerType
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build(M=32, N=32, K=32, TM=32, TN=32, TK=32):
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            # tensor view shape
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_a = pto.PartitionTensorViewType.get([TM, TK], f32, ctx)
            tile_view_b = pto.PartitionTensorViewType.get([TK, TN], f32, ctx)
            tile_view_c = pto.PartitionTensorViewType.get([TM, TN], f32, ctx)

            mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT, ctx)
            left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT, ctx)
            right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT, ctx)
            acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC, ctx)

            pd = pto.PadValueAttr.get(pto.PadValue.Zero, ctx)
            cfg_mat = pto.TileBufConfigAttr.get(pto.BLayoutAttr.get(
                pto.BLayout.ColMajor, ctx), pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx), 512, pd, ctx)
            cfg_right = pto.TileBufConfigAttr.get(pto.BLayoutAttr.get(
                pto.BLayout.RowMajor, ctx), pto.SLayoutAttr.get(pto.SLayout.ColMajor, ctx), 512, pd, ctx)
            cfg_acc = pto.TileBufConfigAttr.get(pto.BLayoutAttr.get(
                pto.BLayout.ColMajor, ctx), pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx), 1024, pd, ctx)

            # cbuf type and layout
            mat_tile_a = pto.TileBufType.get([TM, TK], f32, mat, [TM, TK], cfg_mat, ctx)
            mat_tile_b = pto.TileBufType.get([TK, TN], f32, mat, [TK, TN], cfg_mat, ctx)

            # l0 type and layout
            left_tile = pto.TileBufType.get([TM, TK], f32, left, [TM, TK], cfg_mat, ctx)
            right_tile = pto.TileBufType.get(
                [TK, TN], f32, right, [TK, TN], cfg_right, ctx)
            acc_tile = pto.TileBufType.get([TM, TN], f32, acc, [TM, TN], cfg_acc, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vector_cube_mixed_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                ctm = arith.ConstantOp(IndexType.get(ctx), TM).result
                ctn = arith.ConstantOp(IndexType.get(ctx), TN).result
                ctk = arith.ConstantOp(IndexType.get(ctx), TK).result
                cm = arith.ConstantOp(IndexType.get(ctx), M).result
                cn = arith.ConstantOp(IndexType.get(ctx), N).result
                ck = arith.ConstantOp(IndexType.get(ctx), K).result

                arg0, arg1, arg2 = entry.arguments
                cid = pto.GetBlockIdxOp().result
                sub_bid = pto.GetSubBlockIdxOp().result
                sub_bnum = pto.GetSubBlockNumOp().result
                cidmul = arith.MulIOp(cid, sub_bnum).result
                vid = arith.AddIOp(cidmul, sub_bid).result
                offset_0 = arith.MulIOp(arith.IndexCastOp(IndexType.get(), vid).result, arith.MulIOp(ctm, ctk).result).result
                offset_1 = arith.MulIOp(arith.IndexCastOp(IndexType.get(), vid).result, arith.MulIOp(ctk, ctn).result).result
                offset_2 = arith.MulIOp(arith.IndexCastOp(IndexType.get(), vid).result, arith.MulIOp(ctm, ctn).result).result

                tv0 = pto.MakeTensorViewOp(
                    tv2_f32, arg0, [ctm, ctk], [ck, c1]).result
                tv1 = pto.MakeTensorViewOp(
                    tv2_f32, arg1, [ctk, ctn], [cn, c1]).result
                tv2 = pto.MakeTensorViewOp(
                    tv2_f32, arg2, [ctm, ctn], [cn, c1]).result

                # Updated subview with constants instead of literals
                sv0 = pto.PartitionViewOp(tile_view_a, tv0, offsets=[
                                    c0, offset_0], sizes=[ctm, ctk]).result
                sv1 = pto.PartitionViewOp(tile_view_b, tv1, offsets=[
                                    c0, offset_1], sizes=[ctk, ctn]).result
                sv2 = pto.PartitionViewOp(tile_view_c, tv2, offsets=[
                                    c0, offset_2], sizes=[ctm, ctn]).result

                # allocate cbuf
                aMatTile = pto.AllocTileOp(mat_tile_a).result
                bMatTile = pto.AllocTileOp(mat_tile_b).result
                # allocate L0A/L0B/L0C
                aTile = pto.AllocTileOp(left_tile).result
                bTile = pto.AllocTileOp(right_tile).result
                cTile = pto.AllocTileOp(acc_tile).result

                # load from gm to cbuf
                pto.TLoadOp(None, sv0, aMatTile)
                pto.TLoadOp(None, sv1, bMatTile)

                # move to l0A/l0B from cbuf
                pto.TMovOp(None, aMatTile, aTile)
                pto.TMovOp(None, bMatTile, bTile)

                pto.TMatmulOp(None, aTile, bTile, cTile)

                # Write output tile back to GM.
                pto.TStoreOp(None, cTile, sv2)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
