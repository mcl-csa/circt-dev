add_circt_dialect(HWArith hwarith)
add_circt_dialect_doc(HWArith hwarith)

set(LLVM_TARGET_DEFINITIONS HWArith.td)
mlir_tablegen(HWArithEnums.h.inc -gen-enum-decls)
mlir_tablegen(HWArithEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHWArithEnumsIncGen)
