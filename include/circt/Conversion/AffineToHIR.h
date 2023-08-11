//===- AffineToHIR.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AFFINETOHIR_H_
#define CIRCT_CONVERSION_AFFINETOHIR_H_

#include "AffineToHIRUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/Transforms/HIRPassImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>
#include <stack>

class HIRScheduler;
namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createAffineToHIR();
} // namespace circt

class AffineToHIRImpl : HIRPassImplBase<mlir::ModuleOp> {
public:
  AffineToHIRImpl(mlir::ModuleOp op, bool dbg)
      : HIRPassImplBase(op), builder(op), dbg(dbg), instNum(0) {}
  void runOnOperation();

private:
  void pushInsertionBlk(mlir::Block &);
  void popInsertionBlk();
  mlir::SmallVector<mlir::Value> getFlattenedHIRIndices(mlir::OperandRange,
                                                        mlir::AffineMap,
                                                        circt::hir::MemrefType,
                                                        mlir::Value, int64_t);

private:
  mlir::LogicalResult visitOperation(mlir::Operation *);
  mlir::LogicalResult visitOp(mlir::func::FuncOp);
  mlir::LogicalResult visitOp(circt::hir::ProbeOp);
  mlir::LogicalResult visitOp(mlir::func::ReturnOp);
  mlir::LogicalResult visitOp(mlir::AffineForOp);
  mlir::LogicalResult visitOp(mlir::AffineLoadOp);
  mlir::LogicalResult visitOp(mlir::AffineStoreOp);
  mlir::LogicalResult visitOp(mlir::AffineYieldOp);
  mlir::LogicalResult visitOp(mlir::arith::ConstantOp);
  mlir::LogicalResult visitOp(mlir::memref::AllocaOp);
  mlir::LogicalResult visitOp(mlir::func::CallOp);
  mlir::LogicalResult visitArithOp(mlir::Operation *);

private:
  mlir::OpBuilder builder;
  HIRScheduler *scheduler;
  mlir::Optional<BlockArgManager> blkArgManager;
  ValueConverter valueConverter;
  std::stack<OpBuilder::InsertionGuard> insertionGuards;
  llvm::DenseMap<std::pair<Value, Region *>, Value> mapValueToRegionArg;
  bool dbg;
  DenseMap<StringRef, DenseSet<StringRef>> mapFuncNameToInstanceNames;
  int64_t instNum;
};
#endif // CIRCT_CONVERSION_AFFINETOHIR_H_
