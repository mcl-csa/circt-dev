add_circt_dialect(HIR hir)
add_circt_dialect_doc(HIR hir)
set(LLVM_TARGET_DEFINITIONS HIR.td)

mlir_tablegen(HIREnums.h.inc -gen-enum-decls)
mlir_tablegen(HIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHIREnumsIncGen)
add_dependencies(circt-headers MLIRHIREnumsIncGen)

mlir_tablegen(HIRAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hir)
mlir_tablegen(HIRAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hir)
add_public_tablegen_target(MLIRHIRAttrsIncGen)
add_dependencies(circt-headers MLIRHIRAttrsIncGen)


set(LLVM_TARGET_DEFINITIONS HIROpInterfaces.td)
mlir_tablegen(HIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HIROpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRHIRInterfacesIncGen)
