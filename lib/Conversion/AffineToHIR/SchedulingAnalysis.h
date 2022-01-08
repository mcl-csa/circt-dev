#ifndef HIR_SCHEDULING_ANALYSIS_H
#define HIR_SCHEDULING_ANALYSIS_H

#include "SchedulingUtils.h"


/// This class calculates the schedule (time offset) of all operations such that
/// the provided target loop II do not cause a dependence or resource violation.
///
/// The ILP variables of loop IV are associated with the actual IV
/// (mlir::Value) of the affine-for loops. ILP variables representing time
/// offset are associated with the corresponding operation in affine dialect
/// (since the corresponding time vars will only be available after lowering to
/// HIR).
class SchedulingILPHandler;
class SchedulingAnalysis {
public:
  SchedulingAnalysis(Operation *operation, std::string &logFile);
  bool hasSolution();
  int64_t getTimeOffset(Operation *);
  std::pair<int64_t, int64_t> getPortNumAndDelayForMemoryOp(Operation *);
  int64_t getLoopII(AffineForOp);

private:
  void initOperationInfo();
  void initSlackAndDelayForMemoryDependencies(
      DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr);

private:
  llvm::DenseMap<Operation *, OpInfo> mapOperationToInfo;
  llvm::DenseMap<std::pair<Operation *, Operation *>,
                 std::pair<int64_t, int64_t>>
      mapMemoryDependenceToSlackAndDelay;
  SmallVector<SSADependence>
      ssaDependences;
  mlir::func::FuncOp funcOp;
  SmallVector<Operation *> operations;
  llvm::Optional<llvm::DenseMap<Operation *, int64_t>> schedule;
  llvm::DenseMap<Operation *, std::pair<int64_t, int64_t>>
      mapOperationToPortNumAndDelay;
  std::string logFile;
};

#endif
