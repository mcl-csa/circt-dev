//===- InstrumentCosim.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/GenCosimFiles.h"
#include "../PassDetail.h"
#include "CPUModuleBuilder.h"
#include "CosimInfo.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <string>
#include <system_error>

using namespace mlir;
using namespace circt;
using namespace hir;

namespace {

struct GenCosimFilesPass : public GenCosimFilesBase<GenCosimFilesPass> {
  void runOnOperation() override;
};
} // namespace

LogicalResult insertProbe(OpBuilder &builder, Value value, int64_t probeID) {
  auto name = "probe_" + std::to_string(probeID);

  builder.create<hir::ProbeOp>(builder.getUnknownLoc(), value, name)
      ->setAttr("id", builder.getI32IntegerAttr(probeID));

  return success();
}

void GenCosimFilesPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  if (!this->entry.hasValue())
    module->emitError(
        "Entry function not specified. Use 'entry' pass parameter to set "
        "the entry function.");
  int64_t probeID = 0;
  llvm::DenseSet<Value> probedValues;
  module.walk([&probedValues, &probeID](Operation *operation) {
    OpBuilder builder(operation);

    // All probes should have a valid signal. The value of store may not have
    // a valid signal associated so we don't probe it.
    if (auto op = dyn_cast<affine::AffineLoadOp>(operation)) {
      builder.setInsertionPointAfter(op);
      if (failed(insertProbe(builder, op.getResult(), probeID++)))
        return WalkResult::interrupt();
      probedValues.insert(op.getResult());
    } else if (auto op = dyn_cast<func::CallOp>(operation)) {
      builder.setInsertionPointAfter(op);
      for (auto result : op.getResults())
        if (failed(insertProbe(builder, result, probeID++)))
          return WalkResult::interrupt();
    } else if (auto op = dyn_cast<affine::AffineForOp>(operation)) {
      builder.setInsertionPointToStart(op.getBody(0));
      if (failed(insertProbe(builder, op.getInductionVar(), probeID++)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  CosimInfo info(module);
  std::string outputDir;
  std::error_code er;

  if (failed(info.walk()))
    return;
  if (this->dir.hasValue()) {
    outputDir = this->dir.getValue();
    std::error_code status = llvm::sys::fs::create_directories(outputDir);
  } else
    outputDir = ".";

  auto jsonFile = llvm::raw_fd_ostream(outputDir + "/cosim.json", er,
                                       llvm::sys::fs::CD_CreateAlways);
  info.print(jsonFile);

  auto verilogModule = dyn_cast<mlir::ModuleOp>(module->clone());
  auto cpuModule = CPUModuleBuilder(module);

  auto cpuFile = llvm::raw_fd_ostream(outputDir + "/cpu-sim.mlir", er,
                                      llvm::sys::fs::CD_CreateAlways);
  if (failed(cpuModule.walk()))
    return;

  cpuModule.print(cpuFile);

  auto verilogFile = llvm::raw_fd_ostream(outputDir + "/verilog-sim.mlir", er,
                                          llvm::sys::fs::CD_CreateAlways);
  verilogModule.print(verilogFile);
}

std::unique_ptr<mlir::Pass> circt::createGenCosimFilesPass() {
  return std::make_unique<GenCosimFilesPass>();
}
