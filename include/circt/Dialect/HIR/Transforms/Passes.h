//===- Passes.h - HIR pass entry points ------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>
namespace circt {
namespace hir {

std::unique_ptr<OperationPass<hw::HWModuleOp>> createFuseHWInstPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createOptBitWidthPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createOptTimePass();
std::unique_ptr<OperationPass<hir::FuncOp>> createSimplifyCtrlPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemrefLoweringPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createVerifySchedulePass();
std::unique_ptr<OperationPass<hir::FuncOp>> createLoopUnrollPass();
std::unique_ptr<OperationPass<hir::FuncOp>> createOpFusionPass();

void registerPassPipelines();
void initHIRTransformationPasses();
} // namespace hir
} // namespace circt
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES_H
