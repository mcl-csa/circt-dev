//===- AutoAffineToHIR.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AutoAffineToHIRPass.h"
#include "../PassDetail.h"
#include "circt/Conversion/AffineToHIR.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/Analysis/OpFusionAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HIR/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <list>
#include <stack>
#include <string>

using namespace circt;
namespace {
struct AutoAffineToHIRPass : public AutoAffineToHIRBase<AutoAffineToHIRPass> {
  void runOnOperation() override;

private:
  llvm::DenseMap<StringRef, mlir::func::CallOp> mapLabelToCallOp;
};
} // namespace

void AutoAffineToHIRPass::runOnOperation() {
  SmallVector<Operation *> toErase;
  getOperation().walk<WalkOrder::PreOrder>(
      [this, &toErase](Operation *operation) {
        if (operation->getParentOp() == getOperation())
          toErase.push_back(operation);
        return WalkResult::advance();
      });

  OpBuilder builder(getOperation());
  builder.setInsertionPointToStart(
      &*getOperation().getBodyRegion().getBlocks().begin());
  // Copy the original module to an inner module.
  auto originalModule =
      dyn_cast<mlir::ModuleOp>(builder.clone(*getOperation()));

  // Remove all original operations.
  for (auto *operation : toErase)
    operation->erase();

  int lbl = 0;
  originalModule.walk([this, &builder, &lbl](Operation *operation) {
    if (auto op = dyn_cast<mlir::func::CallOp>(operation)) {
      if (!op->hasAttrOfType<StringAttr>("hir.label"))
        op->setAttr("hir.label", builder.getStringAttr(op.getCallee() + "_" +
                                                       std::to_string(lbl++)));
      mapLabelToCallOp[op->getAttrOfType<StringAttr>("hir.label").strref()] =
          op;
    }
  });

  builder.setInsertionPointAfter(originalModule);
  auto clonedModule = dyn_cast<mlir::ModuleOp>(builder.clone(*originalModule));
  AffineToHIRImpl affineToHIRPass(clonedModule, this->dbg);
  affineToHIRPass.runOnOperation();
  originalModule->setAttr("affine.seq", builder.getUnitAttr());
  clonedModule->setAttr("hir.par", builder.getUnitAttr());
  DenseMap<StringRef, AccessInfo> mapLabelToAccessInfo;
  clonedModule->walk([&mapLabelToAccessInfo](hir::FuncOp op) {
    OpFusionAnalysis opFusionAnalysis(op);
    opFusionAnalysis.getAnalysis(mapLabelToAccessInfo);
  });
  for (auto it : mapLabelToAccessInfo) {
    mapLabelToCallOp[it.getFirst()]->setAttr("access_info",
                                             it.getSecond().intoAttr(builder));
  }
}

//-----------------------------------------------------------------------------
std::unique_ptr<mlir::Pass> circt::createAutoAffineToHIRPass() {
  return std::make_unique<AutoAffineToHIRPass>();
}