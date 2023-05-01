#ifndef HIR_SCHEDULING_ANALYSIS_H
#define HIR_SCHEDULING_ANALYSIS_H

#include "SchedulingUtils.h"
#include <memory>

/// This class calculates the schedule (time offset) of all operations such that
/// the provided target loop II do not cause a dependence or resource violation.
class SchedulingAnalysis {
public:
  SchedulingAnalysis(mlir::func::FuncOp funcOp);
  int64_t getTimeOffset(mlir::Operation *);
  int64_t getPortNumForMemoryOp(mlir::Operation *);

private:
  mlir::LogicalResult insertDependencies();

private:
  mlir::func::FuncOp funcOp;
  std::unique_ptr<Scheduler> scheduler;
};

#endif
