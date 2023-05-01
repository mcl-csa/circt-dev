#include "circt/Conversion/SchedulingAnalysis.h"
#include "circt/Conversion/SchedulingUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdlib>
#include <memory>
//-----------------------------------------------------------------------------
// class OpInfo methods.
//-----------------------------------------------------------------------------
using namespace mlir;
OpInfo::OpInfo(Operation *operation, int staticPos)
    : operation(operation), staticPos(staticPos) {
  auto *currentOperation = operation;
  while (auto affineForOp = currentOperation->getParentOfType<AffineForOp>()) {
    parentLoops.push_back(affineForOp);
    parentLoopIVs.push_back(affineForOp.getInductionVar());
    currentOperation = affineForOp;
  }
}

Operation *OpInfo::getOperation() { return operation; }
int OpInfo::getStaticPosition() { return staticPos; }
size_t OpInfo::getNumParentLoops() { return parentLoops.size(); }
AffineForOp OpInfo::getParentLoop(int i) { return parentLoops[i]; }
Value OpInfo::getParentLoopIV(int i) { return parentLoopIVs[i]; }

MemOpInfo::MemOpInfo(mlir::AffineLoadOp operation, int staticPos)
    : OpInfo(operation, staticPos) {}

MemOpInfo::MemOpInfo(mlir::AffineStoreOp operation, int staticPos)
    : OpInfo(operation, staticPos) {}

Value MemOpInfo::getMemRef() {
  if (auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation()))
    return storeOp.getMemRef();
  auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation());
  assert(loadOp);
  return loadOp.getMemRef();
}

int64_t MemOpInfo::getDelay() {
  return getOperation()->getAttrOfType<IntegerAttr>("hir.delay").getInt();
}

size_t MemOpInfo::getNumMemDims() {
  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation()))
    return loadOp.getAffineMap().getNumDims();

  auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation());
  assert(storeOp);
  return storeOp.getAffineMap().getNumDims();
}

int64_t MemOpInfo::getConstCoeff(int64_t dim) {
  SmallVector<int64_t> coeffs;
  AffineExpr expr;
  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation())) {
    expr = loadOp.getAffineMap().getResult(dim);
  } else {
    auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation());
    expr = storeOp.getAffineMap().getResult(dim);
  }

  // If the expression could not be flattened then treat the whole dim as zero
  // (i.e. all accesses alias on this dim).
  if (failed(getFlattenedAffineExpr(expr, getNumMemDims(), 0, &coeffs)))
    return 0;

  return coeffs.back();
}

int64_t MemOpInfo::getIdxCoeff(mlir::Value var, int64_t dim) {
  auto indices = getIndices();
  Optional<size_t> varLoc = llvm::None;
  AffineExpr expr;

  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation())) {
    expr = loadOp.getAffineMap().getResult(dim);
    for (size_t i = 0; i < indices.size(); i++) {
      if (indices[i] == var) {
        varLoc = i;
        break;
      }
    }
  } else if (auto storeOp = dyn_cast<AffineLoadOp>(this->getOperation())) {
    expr = storeOp.getAffineMap().getResult(dim);
    for (size_t i = 0; i < indices.size(); i++) {
      if (indices[i] == var) {
        varLoc = i;
        break;
      }
    }
  } else {
    assert(false && "Must be a affine load or store op.");
  }

  if (!varLoc)
    return 0;

  SmallVector<int64_t> coeffs;
  if (failed(getFlattenedAffineExpr(expr, getNumMemDims(), 0, &coeffs)))
    return 0;

  return coeffs[*varLoc];
}

SmallVector<int64_t, 4>
MemOpInfo::getIdxCoeffs(mlir::ArrayRef<mlir::Value> indices, int64_t dim) {

  SmallVector<int64_t, 4> coeffs;
  for (auto idx : indices)
    coeffs.push_back(this->getIdxCoeff(idx, dim));

  return coeffs;
}

llvm::SmallVector<mlir::Value> MemOpInfo::getIndices() {
  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation()))
    return loadOp.getIndices();

  auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation());
  assert(storeOp);
  return storeOp.getIndices();
}

//-----------------------------------------------------------------------------
// class SchedulingAnalysis methods.
//-----------------------------------------------------------------------------
SchedulingAnalysis::SchedulingAnalysis(mlir::func::FuncOp op)
    : funcOp(op), scheduler(std::make_unique<Scheduler>()) {}

int64_t SchedulingAnalysis::getTimeOffset(Operation *operation) {
  return scheduler->getTimeOffset(operation);
}

int64_t SchedulingAnalysis::getPortNumForMemoryOp(Operation *operation) {
  assert(isa<AffineLoadOp>(operation) || isa<AffineStoreOp>(operation));
  // FIXME
}

LogicalResult SchedulingAnalysis::insertDependencies() {
  SmallVector<MemOpInfo> memOperations;
  funcOp.walk([&memOperations](Operation *operation) {
    if (auto loadOp = dyn_cast<AffineLoadOp>(operation))
      memOperations.push_back(MemOpInfo(loadOp, memOperations.size()));
    else if (auto storeOp = dyn_cast<AffineStoreOp>(operation))
      memOperations.push_back(MemOpInfo(storeOp, memOperations.size()));
    return WalkResult::advance();
  });
  for (size_t i = 0; i < memOperations.size(); i++) {
    auto srcOp = memOperations[i];
    for (size_t j = 0; j < memOperations.size(); j++) {
      auto destOp = memOperations[j];
      if (srcOp.getMemRef() != destOp.getMemRef())
        continue;

      // FIXME: Currently we assume load-to-load dependence to avoid port
      // conflicts.
      // if (isa<AffineLoadOp>(srcOp) && isa<AffineLoadOp>(destOp))
      //  continue;

      MemoryDependenceILPHandler memoryDependenceILP(srcOp, destOp);
      if (memoryDependenceILP.Solve() !=
          MemoryDependenceILPHandler::ResultStatus::OPTIMAL)
        return srcOp.getOperation()
                   ->emitError("Could not calculate slack.")
                   .attachNote(destOp.getOperation()->getLoc())
               << "destination op here.";
      int64_t dist = (int64_t)memoryDependenceILP.Objective().Value();

      // FIXME: Currently we assume a simple dual port RAM.
      bool const portConflict =
          (isa<AffineLoadOp>(srcOp) && isa<AffineLoadOp>(destOp)) ||
          (isa<AffineStoreOp>(srcOp) && isa<AffineStoreOp>(destOp));

      int64_t delay = isa<AffineLoadOp>(srcOp.getOperation())
                          ? (portConflict ? 1 : 0)
                          : srcOp.getDelay();
      DependenceConstraint dep(srcOp.getOperation(), destOp.getOperation(),
                               dist + delay);
      this->scheduler->addDependenceConstraint(dep);
    }
  }
  return success();
}
