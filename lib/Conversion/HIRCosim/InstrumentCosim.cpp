//===- InstrumentCosim.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/InstrumentCosim.h"
#include "../PassDetail.h"
#include "CPUModuleBuilder.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <string>

using namespace mlir;
using namespace circt;
using namespace hir;

namespace {

struct InstrumentCosim : public InstrumentCosimBase<InstrumentCosim> {
  void runOnOperation() override;
};
} // namespace
LogicalResult insertProbes(affine::AffineLoadOp op, int64_t *probeID) {
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  auto name = "probe_" + std::to_string((*probeID));

  builder.create<hir::ProbeOp>(builder.getUnknownLoc(), op.getResult(), name)
      ->setAttr("id", builder.getI64IntegerAttr(*probeID));
  (*probeID)++;

  for (auto idx : op.getIndices()) {
    auto name = "probe_" + std::to_string((*probeID));
    if (idx.getType().isa<IndexType>())
      idx = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(),
                                               builder.getI32Type(), idx);

    builder.create<hir::ProbeOp>(builder.getUnknownLoc(), idx, name)
        ->setAttr("id", builder.getI64IntegerAttr(*probeID));
    (*probeID)++;
  }
  return success();
}

void InstrumentCosim::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  if (!this->entry.hasValue())
    module->emitError(
        "Entry function not specified. Use 'entry' pass parameter to set "
        "the entry function.");
  int64_t probeID = 0;
  module.walk([&probeID](Operation *operation) {
    if (auto op = dyn_cast<affine::AffineLoadOp>(operation)) {
      if (failed(insertProbes(op, &probeID)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  auto verilogModule = dyn_cast<mlir::ModuleOp>(module->clone());
  auto cpuModule = CPUModuleBuilder(module);

  std::error_code er;
  std::string outputDir;

  if (this->dir.hasValue()) {
    outputDir = this->dir.getValue();
    llvm::sys::fs::create_directories(outputDir);
  } else
    outputDir = ".";

  auto cpuFile = llvm::raw_fd_ostream(outputDir + "/cpu-sim.mlir", er,
                                      llvm::sys::fs::CD_CreateAlways);
  auto jsonFile = llvm::raw_fd_ostream(outputDir + "/cosim.json", er,
                                       llvm::sys::fs::CD_CreateAlways);
  if (failed(cpuModule.walk()))
    return;

  cpuModule.print(cpuFile);
  cpuModule.printJSON(jsonFile);

  auto verilogFile = llvm::raw_fd_ostream(outputDir + "/verilog-sim.mlir", er,
                                          llvm::sys::fs::CD_CreateAlways);
  verilogModule.print(verilogFile);
}

std::unique_ptr<mlir::Pass> circt::createInstrumentCosimPass() {
  return std::make_unique<InstrumentCosim>();
}