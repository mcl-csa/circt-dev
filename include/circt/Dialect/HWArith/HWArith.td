//===- HWArith.td - HWArith dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the HWArith dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HWARITH_TD
#define HWARITH_TD

include "mlir/IR/OpBase.td"

def HWArithDialect : Dialect {
  let name = "hwarith";
  let cppNamespace = "::circt::hwarith";

  let summary = "Types and operations for the HWArith dialect";
  let description = [{
    This dialect defines the `HWArith` dialect, modeling bit-width aware
    arithmetic operations.
  }];
}

// Base class for the operation in this dialect.
class HWArithOp<string mnemonic, list<Trait> traits = []> :
    Op<HWArithDialect, mnemonic, traits>;

include "circt/Dialect/HWArith/HWArithTypes.td"
// It is important that HWArithTypes.td is included first
include "circt/Dialect/HWArith/HWArithOps.td"

#endif // HWARITH_TD
