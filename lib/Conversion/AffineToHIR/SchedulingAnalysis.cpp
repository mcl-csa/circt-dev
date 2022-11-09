#include "SchedulingAnalysis.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
//-----------------------------------------------------------------------------
// class OpInfo methods.
//-----------------------------------------------------------------------------

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
ArrayRef<AffineForOp> OpInfo::getParentLoops() { return parentLoops; }
ArrayRef<Value> OpInfo::getParentLoopIVs() { return parentLoopIVs; }

//-----------------------------------------------------------------------------
// class SchedulingAnalysis methods.
//-----------------------------------------------------------------------------
SchedulingAnalysis::SchedulingAnalysis(Operation *operation,
                                       std::string &logFile)
    : funcOp(dyn_cast<mlir::func::FuncOp>(operation)), logFile(logFile) {
  std::error_code ec;
  llvm::raw_fd_ostream os(logFile, ec);
  os << "----------------------------------------------------------------------"
        "----------\n";
  os << "This log file contains the ILP formulation for affine-to-hir "
        "pass.\n";
  os << "----------------------------------------------------------------------"
        "----------\n\n\n";
  os.close();

  SmallVector<SSADependence> ssaDependences;
  llvm::DenseMap<Value, ArrayAttr> mapMemrefToPortsAttr;
  initOperationInfo();
  populateMemrefToPortsAttrMapping(funcOp, mapMemrefToPortsAttr);
  initSlackAndDelayForMemoryDependencies(mapMemrefToPortsAttr);
  if (failed(populateSSADependences(funcOp, ssaDependences)))
    return;
  auto scheduler = std::make_unique<SchedulingILPHandler>(
      SchedulingILPHandler(operations, mapMemoryDependenceToSlackAndDelay,
                           ssaDependences, mapMemrefToPortsAttr, logFile));
  this->schedule = scheduler->getSchedule();
  this->mapOperationToPortNumAndDelay = scheduler->getPortAssignments();
}

bool SchedulingAnalysis::hasSolution() { return this->schedule.hasValue(); }

int64_t SchedulingAnalysis::getTimeOffset(Operation *operation) {
  assert(schedule.hasValue());
  auto v = schedule->find(operation);
  assert(v != schedule->end());
  return v->second;
}

std::pair<int64_t, int64_t>
SchedulingAnalysis::getPortNumAndDelayForMemoryOp(Operation *operation) {
  assert(isa<AffineLoadOp>(operation) || isa<AffineStoreOp>(operation));
  auto portNumAndDelay = this->mapOperationToPortNumAndDelay.find(operation);
  return portNumAndDelay->getSecond();
}

void SchedulingAnalysis::initOperationInfo() {
  int staticPos = 0;
  funcOp.walk<mlir::WalkOrder::PreOrder>(
      [this, &staticPos](Operation *operation) {
        mapOperationToInfo[operation] = OpInfo(operation, staticPos++);
        operations.push_back(operation);
        return WalkResult::advance();
      });
}

void SchedulingAnalysis::initSlackAndDelayForMemoryDependencies(
    DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr) {
  SmallVector<Operation *> memOperations;
  funcOp.walk([&memOperations](Operation *operation) {
    if (isa<AffineLoadOp, AffineStoreOp>(operation))
      memOperations.push_back(operation);
    return WalkResult::advance();
  });
  for (auto *srcOp : memOperations) {
    for (auto *destOp : memOperations) {
      if (getMemrefFromAffineLoadOrStoreOp(srcOp) !=
          getMemrefFromAffineLoadOrStoreOp(destOp))
        continue;
      // FIXME: Currently we assume load-to-load dependence to avoid port
      // conflicts. if (isa<AffineLoadOp>(srcOp) && isa<AffineLoadOp>(destOp))
      //  continue;
      MemoryDependenceILPHandler memoryDependenceILP(
          mapOperationToInfo[srcOp], mapOperationToInfo[destOp], logFile);
      auto dist = memoryDependenceILP.calculateSlack();

      // FIXME: Currently we assume a simple dual port RAM.
      bool potentialPortConflict =
          (isa<AffineLoadOp>(srcOp) && isa<AffineLoadOp>(destOp)) ||
          (isa<AffineStoreOp>(srcOp) && isa<AffineStoreOp>(destOp));
      if (dist) {
        int minRequiredDelay;
        minRequiredDelay = getMemOpSafeDelay(srcOp, mapMemrefToPortsAttr);
        if (potentialPortConflict)
          minRequiredDelay = std::max(minRequiredDelay, 1);

        mapMemoryDependenceToSlackAndDelay[std::make_pair(srcOp, destOp)] =
            std::make_pair(dist.getValue(), minRequiredDelay);
      }
    }
  }
}

int64_t SchedulingAnalysis::getLoopII(AffineForOp op) {
  return ::getLoopII(op);
}
