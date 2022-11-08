#ifndef HIR_SCHEDULING_UTILS_H
#define HIR_SCHEDULING_UTILS_H
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "glpk.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>

using namespace mlir;
struct RowVarInfo {
  std::string name;
  int boundKind;
  int lb;
  int ub;
};

struct ColumnVarInfo {
  std::string name;
  int boundKind;
  int lb;
  int ub;
  int objectiveCoeff;
};
struct SSADependence {
  SSADependence(Operation *destOp, Value value, int64_t delay)
      : destOp(destOp), value(value), delay(delay) {}

private:
  Operation *destOp;
  Value value;
  int64_t delay;

public:
  bool srcIsRegionArg() { return value.getDefiningOp() == NULL; }
  Operation *getSrcOp() {
    auto op = value.getDefiningOp();
    assert(op);
    return op;
  }
  Operation *getDestOp() { return destOp; }
  int64_t getMinimumDelay() { return delay; }
};

// Helper functions.
int getLoopII(AffineForOp affineForOp);

Value getMemrefFromAffineLoadOrStoreOp(Operation *operation);

int64_t getMemOpSafeDelay(Operation *operation,
                          DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr);

llvm::Optional<int64_t> getResultDelay(OpResult v);

void populateMemrefToPortsAttrMapping(
    mlir::func::FuncOp funcOp,
    llvm::DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr);

LogicalResult populateSSADependences(mlir::func::FuncOp funcOp,
                                     SmallVector<SSADependence> &SSADependence);

/// This class is a general base class for all ILP problems.
class ILPHandler {
public:
  ILPHandler(const char *ilpName, int optKind, std::string &logFile);
  void incrObjectiveCoeff(int columnNum, int valueToIncr);
  void addColumnVar(int boundKind, int lb, int ub, int objectiveCoeff = 0);
  void addRow(ArrayRef<int> rowCoeffs, int boundKind, int lb, int ub);
  llvm::Optional<int64_t> solve();
  void dumpInput();
  void dumpResult();
  int getNumCols();
  int64_t getColVarValue(int64_t col);

private:
  SmallVector<ColumnVarInfo> columnVars;
  SmallVector<RowVarInfo> rowVars;
  SmallVector<int, 4> ia;
  SmallVector<int, 4> ja;
  SmallVector<double, 4> ar;
  std::string ilpName;
  int optKind;
  glp_prob *mip;

public:
  std::string logFile;
};

/// OpInfo contains the list of parent loop initiation intervals and their
/// induction vars.
struct OpInfo {
  OpInfo(Operation *operation, int staticPos);
  OpInfo() { staticPos = -1; }
  Operation *getOperation();
  ArrayRef<AffineForOp> getParentLoops();
  ArrayRef<Value> getParentLoopIVs();
  int getStaticPosition();

private:
  SmallVector<AffineForOp> parentLoops;
  SmallVector<Value> parentLoopIVs;
  Operation *operation;
  int staticPos; // Represents the position of this op in the code. Used to
                 // figure out if one op occurs before another in the static
                 // order & to uniquely identify the static op.
};

/// This class solves the following minimization problem:
///   d = minimize (II_to * I_to - II_from*I_from)
/// - I_from is an array of induction vars of parent loops of the `from`
/// operation.
/// - II_from is an array of the corresponding loop initiation intervals.
class MemoryDependenceILPHandler : ILPHandler {
public:
  MemoryDependenceILPHandler(OpInfo fromInfo, OpInfo toInfo,
                             std::string &logFile);

  llvm::Optional<int64_t> calculateSlack();

private:
  void insertRowCoefficients(SmallVectorImpl<int> &rowCoeffVec,
                             ArrayRef<int64_t> coeffs, OperandRange memIndices,
                             ArrayRef<Value> loopIVs, bool isNegativeCoeff);
  void addILPColumns();
  void addHappensBeforeConstraintRow();
  void addMemoryConstraintILPRows();

private:
  OpInfo fromInfo;
  OpInfo toInfo;
};

/// This class calculates the final schedule given the slack between memory
/// ops
// and the minimum delays between def and use.
class SchedulingILPHandler : ILPHandler {
public:
  SchedulingILPHandler(const SmallVector<Operation *> operations,
                       const llvm::DenseMap<std::pair<Operation *, Operation *>,
                                            std::pair<int64_t, int64_t>>
                           &mapMemoryDependenceToSlackAndDelay,
                       const SmallVector<SSADependence> &SSADependence,
                       llvm::DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr,
                       std::string &logFile);
  void addILPColumns(llvm::raw_fd_ostream &);
  void addMemoryDependenceRows();
  void addSSADependenceRows();
  void addMaxTimeOffsetRows();
  llvm ::Optional<DenseMap<Operation *, int64_t>> getSchedule();
  llvm::DenseMap<Operation *, std::pair<int64_t, int64_t>> getPortAssignments();

private:
  const SmallVector<Operation *> operations;
  const llvm::DenseMap<std::pair<Operation *, Operation *>,
                       std::pair<int64_t, int64_t>>
      &mapMemoryDependenceToSlackAndDelay;
  const SmallVector<SSADependence> ssaDependences;
  llvm::DenseMap<Value, ArrayAttr> mapMemrefToPortsAttr;
  llvm::DenseMap<Operation *, size_t> mapOperationToCol;
  // Pre-specified value for time variables.
  // FIXME: Implement this.
  llvm::DenseMap<Operation *, Optional<int64_t>>
      mapOperationToPreSpecifiedSchedule;
};
#endif
