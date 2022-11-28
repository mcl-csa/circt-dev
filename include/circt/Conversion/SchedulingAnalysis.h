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
  SchedulingAnalysis(mlir::Operation *operation, const std::string &logFile);
  bool hasSolution();
  int64_t getTimeOffset(mlir::Operation *);
  std::pair<int64_t, int64_t> getPortNumAndDelayForMemoryOp(mlir::Operation *);
  int64_t getLoopII(mlir::AffineForOp);

private:
  void initOperationInfo();
  void initSlackAndDelayForMemoryDependencies(
      mlir::DenseMap<mlir::Value, mlir::ArrayAttr> &mapMemrefToPortsAttr);

private:
  llvm::DenseMap<mlir::Operation *, OpInfo> mapOperationToInfo;
  llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                 std::pair<int64_t, int64_t>>
      mapMemoryDependenceToSlackAndDelay;
  llvm::SmallVector<SSADependence> ssaDependences;
  mlir::func::FuncOp funcOp;
  llvm::SmallVector<mlir::Operation *> operations;
  llvm::Optional<llvm::DenseMap<mlir::Operation *, int64_t>> schedule;
  llvm::DenseMap<mlir::Operation *, std::pair<int64_t, int64_t>>
      mapOperationToPortNumAndDelay;
  std::string logFile;
};

#endif
