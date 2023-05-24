#include "circt/Conversion/SchedulingAnalysis.h"
#include "PragmaHandler.h"
#include "circt/Conversion/SchedulingUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <memory>
using namespace mlir;

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------
/// Finds the containing user (such as a ForOp) that is immediately in the
/// lexical scope.
static Operation *getOpContainingUserInScope(Operation *user, Region *scope) {
  while (user->getParentRegion() != scope)
    user = user->getParentOp();
  return user;
}

//-----------------------------------------------------------------------------
// class SchedulingAnalysis methods.
//-----------------------------------------------------------------------------
HIRScheduler::HIRScheduler(mlir::func::FuncOp op, llvm::raw_ostream &logger)
    : Scheduler(logger), funcOp(op) {}

int64_t HIRScheduler::getPortNumForMemoryOp(Operation *operation) {
  // FIXME: Handle multiple read/write ports.
  if (auto loadOp = dyn_cast<AffineLoadOp>(operation)) {
    auto pragma = MemrefPragmaHandler(loadOp.getMemref());
    assert(pragma.getNumRdPorts() == 1);
    for (int i = 0; i < pragma.getNumPorts(); i++) {
      if (pragma.getPortKind(i) == MemrefPragmaHandler::PortKind::READ_ONLY ||
          pragma.getPortKind(i) == MemrefPragmaHandler::PortKind::READ_WRITE) {
        return i;
      }
    }
  }
  auto storeOp = dyn_cast<AffineStoreOp>(operation);
  auto pragma = MemrefPragmaHandler(storeOp.getMemref());
  assert(pragma.getNumWrPorts() == 1);
  for (int i = 0; i < pragma.getNumPorts(); i++) {
    if (pragma.getPortKind(i) == MemrefPragmaHandler::PortKind::WRITE_ONLY ||
        pragma.getPortKind(i) == MemrefPragmaHandler::PortKind::READ_WRITE) {
      return i;
    }
  }
  operation->emitError("Could not find port.");
  return 0;
}

LogicalResult HIRScheduler::insertMemoryDependencies() {
  SmallVector<MemOpInfo> memOperations;
  funcOp.walk([&memOperations](Operation *operation) {
    if (isa<AffineLoadOp, AffineStoreOp>(operation))
      memOperations.push_back(MemOpInfo(operation, memOperations.size()));
    return WalkResult::advance();
  });

  for (size_t i = 0; i < memOperations.size(); i++) {
    auto srcOpInfo = memOperations[i];
    for (size_t j = 0; j < memOperations.size(); j++) {
      auto destOpInfo = memOperations[j];
      if (srcOpInfo.getMemRef() != destOpInfo.getMemRef())
        continue;

      // FIXME: Currently we assume load-to-load dependence to avoid port
      // conflicts.
      // if (isa<AffineLoadOp>(srcOp) && isa<AffineLoadOp>(destOp))
      //  continue;

      logger << "\n=========================================\n";
      logger << "MemoryDependenceILP:";
      logger << "\n=========================================\n\n";
      MemoryDependenceILPHandler memoryDependenceILP(srcOpInfo, destOpInfo,
                                                     logger);

      // If the solution is not found then there is no dependence between the
      // operations.

      logger << "Source:";
      srcOpInfo.getOperation()->getLoc().print(logger);
      logger << "\nDest:";
      destOpInfo.getOperation()->getLoc().print(logger);
      logger << "\n\nILP:\n";
      logger << "----\n";
      memoryDependenceILP.solve();
      memoryDependenceILP.dump();

      if (!memoryDependenceILP.isSolved()) {
        logger << "\nNo dependence found.\n";
        logger << "\n=========================================\n\n";
        continue;
      }
      int64_t dist = (int64_t)memoryDependenceILP.Objective().Value();
      logger << "\nDependence found. dist= " << dist << "\n";
      logger << "\n=========================================\n\n";

      // FIXME: Currently we assume a simple dual port RAM.
      bool const portConflict =
          (isa<AffineLoadOp>(srcOpInfo.getOperation()) &&
           isa<AffineLoadOp>(destOpInfo.getOperation())) ||
          (isa<AffineStoreOp>(srcOpInfo.getOperation()) &&
           isa<AffineStoreOp>(destOpInfo.getOperation()));

      int64_t delay = isa<AffineLoadOp>(srcOpInfo.getOperation())
                          ? (portConflict ? 1 : 0)
                          : srcOpInfo.getDelay();
      assert(delay >= 0);
      DependenceConstraint dep("mem_dep", srcOpInfo.getOperation(),
                               destOpInfo.getOperation(), delay - dist);
      this->addDependenceConstraint(dep);
    }
  }
  return success();
}

LogicalResult HIRScheduler::insertSSADependencies() {
  auto walkResult = funcOp.walk([this](Operation *operation) {
    if (operation->getNumResults() == 0)
      return WalkResult::advance();

    if (isa<arith::ConstantOp, AffineForOp, memref::AllocaOp>(operation))
      return WalkResult::advance();

    int64_t delay;
    if (isa<arith::ArithmeticDialect>(operation->getDialect()))
      delay = ArithOpInfo(operation).getDelay();
    else if (isa<AffineLoadOp, AffineStoreOp>(operation))
      delay = MemOpInfo(operation, 0 /*dont care*/).getDelay();
    else {
      llvm_unreachable("Unknown operation.");
    }

    for (auto result : operation->getResults()) {
      for (auto *user : result.getUsers()) {
        auto *userAtThisScope =
            getOpContainingUserInScope(user, operation->getParentRegion());
        DependenceConstraint dep("ssa_dep", operation, userAtThisScope, delay);
        this->addDependenceConstraint(dep);
        this->addDelayRegisterCost(dep,
                                   result.getType().getIntOrFloatBitWidth());
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

LogicalResult HIRScheduler::init() {
  if (failed(insertMemoryDependencies()))
    return failure();
  if (failed(insertSSADependencies()))
    return failure();
  if (this->solve() != ResultStatus::OPTIMAL) {
    return this->dump(), funcOp->emitError("Could not find schedule.");
  }
  return success();
}