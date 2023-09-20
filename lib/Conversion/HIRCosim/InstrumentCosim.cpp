//===- InstrumentCosim.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/InstrumentCosim.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
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

using namespace mlir;
using namespace circt;
using namespace hir;

namespace {

struct InstrumentCosim : public InstrumentCosimBase<InstrumentCosim> {
  void runOnOperation() override;
};

class CPUModuleBuilder {
public:
  CPUModuleBuilder(mlir::ModuleOp mod) : mod(mod) {
    cosimInfo["Description"] = "Cosimulation information.";
  }
  LogicalResult walk();
  void print(llvm::raw_ostream &os);
  void printJSON(llvm::raw_ostream &os);

private:
  LogicalResult visitOp(Operation *op);
  LogicalResult visitOp(hir::FuncExternOp op);
  LogicalResult visitOp(hir::ProbeOp op);
  LogicalResult visitOp(func::FuncOp op);

private:
  mlir::ModuleOp mod;
  llvm::json::Object cosimInfo;
};

// Helper functions.
static void
removeHIRAttrs(std::function<void(StringAttr attrName)> const &removeAttr,
               DictionaryAttr attrList) {
  for (auto attr : attrList) {
    if (llvm::isa_and_nonnull<hir::HIRDialect>(attr.getNameDialect())) {
      removeAttr(attr.getName());
    }
  }
}
} // namespace

LogicalResult CPUModuleBuilder::walk() {
  OpBuilder builder(this->mod);
  builder.setInsertionPointToStart(mod.getBody());
  auto funcTy = builder.getFunctionType(builder.getI32Type(),
                                        llvm::SmallVector<Type>({}));
  builder.create<func::FuncOp>(builder.getUnknownLoc(), "record", funcTy,
                               builder.getStringAttr("private"), ArrayAttr(),
                               ArrayAttr());

  this->mod.walk([this](Operation *operation) {
    if (isa<hir::ReturnOp>(operation)) {
      return WalkResult::advance();
    }
    if (auto op = dyn_cast<mlir::func::FuncOp>(operation)) {
      if (failed(visitOp(op))) {
        return WalkResult::interrupt();
      }
    }
    if (auto op = dyn_cast<hir::FuncExternOp>(operation)) {
      if (failed(visitOp(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto op = dyn_cast<hir::ProbeOp>(operation)) {
      if (failed(visitOp(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (failed(visitOp(operation))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(hir::FuncExternOp op) {
  op.erase();
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(func::FuncOp op) {
  if (op.getArgAttrs()) {
    // Remove hir arg attributes.
    for (size_t i = 0; i < op.getNumArguments(); i++)

      removeHIRAttrs(
          [&op, i](StringAttr attrName) { op.removeArgAttr(i, attrName); },
          op.getArgAttrDict(i));
  }

  // Remove hir result attributes.
  for (size_t i = 0; i < op->getNumResults(); i++)
    removeHIRAttrs(
        [&op, i](StringAttr attrName) { op.removeResultAttr(i, attrName); },
        op.getResultAttrDict(i));
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(hir::ProbeOp op) {
  OpBuilder builder(op);
  builder.create<mlir::func::CallOp>(
      op.getLoc(), TypeRange(),
      SymbolRefAttr::get(builder.getStringAttr("record")), op.getInput());
  op.erase();
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(Operation *operation) {
  if (llvm::isa_and_nonnull<hir::HIRDialect>(operation->getDialect())) {
    return operation->emitError("Found unsupported HIR operation.");
  }

  removeHIRAttrs(
      [operation](StringAttr attrName) { operation->removeAttr(attrName); },
      operation->getAttrDictionary());
  return success();
}

void CPUModuleBuilder::print(llvm::raw_ostream &os) { mod.print(os); }
void CPUModuleBuilder::printJSON(llvm::raw_ostream &os) {
  os << llvm::json::Value(std::move(cosimInfo));
}

void InstrumentCosim::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  if (!this->entry.hasValue())
    module->emitError(
        "Entry function not specified. Use 'entry' pass parameter to set "
        "the entry function.");
  auto verilogModule = dyn_cast<mlir::ModuleOp>(module->clone());
  auto cpuModule = CPUModuleBuilder(module);

  std::error_code er;
  std::string outputDir;

  if (this->dir.hasValue()) {
    outputDir = this->dir.getValue();
    llvm::sys::fs::create_directories(outputDir);
  } else
    outputDir = ".";

  auto cpuFile = llvm::raw_fd_ostream(outputDir + "/cpu-module.mlir", er,
                                      llvm::sys::fs::CD_CreateAlways);
  auto jsonFile = llvm::raw_fd_ostream(outputDir + "/cosim.json", er,
                                       llvm::sys::fs::CD_CreateAlways);
  if (failed(cpuModule.walk()))
    return;

  cpuModule.print(cpuFile);
  cpuModule.printJSON(jsonFile);

  auto verilogFile = llvm::raw_fd_ostream(outputDir + "/verilog-module.mlir",
                                          er, llvm::sys::fs::CD_CreateAlways);
  verilogModule.print(verilogFile);
}

std::unique_ptr<mlir::Pass> circt::createInstrumentCosimPass() {
  return std::make_unique<InstrumentCosim>();
}