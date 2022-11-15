//=========- ScheduleAnalysis.cpp - Generate schedule info ----------------===//
//
// This file defines the TimingInfo analysis class. This class holds the
// scheduling info.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Location.h"
#include <set>
namespace circt {
namespace hir {

/// This class maintains the mapping between equivalent time variables.
/// It is used to find alternate timevars which are equivalent but require
/// smaller offset (and thus less shift registers).
class EquivalentTimeMap {
  /// If t2 = t1+10 then this maps t1 -> 10,t2) so that if we find t1+11
  /// somewhere, we can search for t1 and get t2+1 == t1+11.
  DenseMap<Value, SmallVector<std::pair<int64_t, Value>>>
      mapTimeVarToOffsetAndEquivalentTimeVar;

  llvm::DenseMap<ScheduledOp, unsigned int> &mapOpToLexicalOrder;
  unsigned int getLexicalOrder(ScheduledOp);
  unsigned int getLexicalOrder(Value);

public:
  EquivalentTimeMap(
      llvm::DenseMap<ScheduledOp, unsigned int> &mapOpToLexicalOrder)
      : mapOpToLexicalOrder(mapOpToLexicalOrder) {}
  void registerEquivalentTime(Value timeVar, Time time);
  /// Get an equivalent time with smaller offset.
  Time getEquivalentTimeWithSmallerOffset(ScheduledOp);
};

/// This class builds the scheduling info for each operation.
/// The schedule-info contains
///   - List of all root time-vars (which can not be expressed as a fixed offset
///   from another time-var).
///   - Mapping from a Value to a time instant. The value is valid at that time
///   instant.
class TimingInfo {
public:
  TimingInfo(FuncOp);
  bool isValidAtTime(Value v, hir::Time time);
  bool isAlwaysValid(Value);
  Time getTime(Value);
  /// Get a new timevar based time which is equivalent to original time but has
  /// smaller offset (and thus requires less shift registers to implement).
  Time getOptimizedTime(hir::ScheduledOp);

private:
  void registerValue(Value, Time);
  LogicalResult visitOp(ScheduledOp);
  LogicalResult visitCombOp(Operation *operation);
  LogicalResult visitOp(circt::hw::ConstantOp);
  LogicalResult visitOp(mlir::arith::ConstantOp);

private:
  llvm::DenseMap<Value, hir::Time> mapValueToTime;
  llvm::SmallDenseSet<Value> setOfConstants;
  EquivalentTimeMap equivalentTimeMap;
  llvm::DenseMap<ScheduledOp, unsigned int> mapOpToLexicalOrder;
  unsigned int currentLexicalPos;
};

} // namespace hir
} // namespace circt
