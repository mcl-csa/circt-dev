//===- InstrumentCosim.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_INSTRUMENTCOSIM_H_
#define CIRCT_CONVERSION_INSTRUMENTCOSIM_H_

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

namespace circt {
std::unique_ptr<mlir::Pass> createInstrumentCosimPass();
} // namespace circt

#endif // CIRCT_CONVERSION_INSTRUMENTCOSIM_H_
