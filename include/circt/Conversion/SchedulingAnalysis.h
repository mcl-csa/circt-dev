#ifndef HIR_SCHEDULING_ANALYSIS_H
#define HIR_SCHEDULING_ANALYSIS_H

#include "SchedulingUtils.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

/// This class calculates the schedule (time offset) of all operations such that
/// the provided target loop II do not cause a dependence or resource violation.
class HIRScheduler : Scheduler {
public:
  HIRScheduler(mlir::func::FuncOp funcOp, llvm::raw_ostream &logger);
  int64_t getPortNumForMemoryOp(mlir::Operation *);
  mlir::LogicalResult init();
  using Scheduler::getTimeOffset;

private:
  using Scheduler::logger;
  mlir::LogicalResult insertMemoryDependencies();
  mlir::LogicalResult insertSSADependencies();
  mlir::func::FuncOp funcOp;
};

#endif
