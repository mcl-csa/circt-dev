//===- AffineToHIR.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AUTOAFFINETOHIR_H_
#define CIRCT_CONVERSION_AUTOAFFINETOHIR_H_

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>
#include <stack>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createAutoAffineToHIRPass();
} // namespace circt
#endif // CIRCT_CONVERSION_AFFINETOHIR_H_
