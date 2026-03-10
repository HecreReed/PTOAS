set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PTO)

# [关键] 重命名为 PTOPassesIncGen (去掉 MLIR 前缀)
add_public_tablegen_target(PTOPassesIncGen)
