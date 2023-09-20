//===- C Interface for the HIR Dialect -------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/HIR.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HIR, hir, circt::hir::HIRDialect)
