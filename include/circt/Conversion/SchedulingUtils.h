#ifndef HIR_SCHEDULING_UTILS_H
#define HIR_SCHEDULING_UTILS_H
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "glpk.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdint>
#include <string>

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
  SSADependence(mlir::Operation *destOp, mlir::Value value, int64_t delay)
      : destOp(destOp), value(value), delay(delay) {}

private:
  mlir::Operation *destOp;
  mlir::Value value;
  int64_t delay;

public:
  bool srcIsRegionArg() { return value.getDefiningOp() == NULL; }
  mlir::Operation *getSrcOp() {
    auto *op = value.getDefiningOp();
    assert(op);
    return op;
  }
  mlir::Operation *getDestOp() { return destOp; }
  int64_t getMinimumDelay() { return delay; }
  mlir::Value getSSAVar() { return value; }
};

// Helper functions.
int getLoopII(mlir::AffineForOp affineForOp);

mlir::Value getMemrefFromAffineLoadOrStoreOp(mlir::Operation *operation);

int64_t getMemOpSafeDelay(
    mlir::Operation *operation,
    mlir::DenseMap<mlir::Value, mlir::ArrayAttr> &mapMemrefToPortsAttr);

llvm::Optional<int64_t> getResultDelay(mlir::OpResult v);

void populateMemrefToPortsAttrMapping(
    mlir::func::FuncOp funcOp,
    llvm::DenseMap<mlir::Value, mlir::ArrayAttr> &mapMemrefToPortsAttr);

mlir::LogicalResult
populateSSADependences(mlir::func::FuncOp funcOp,
                       mlir::SmallVector<SSADependence> &ssaDependence);

/// This class is a general base class for all ILP problems.
class ILPHandler {
public:
  ILPHandler(const char *ilpName, int optKind, const std::string &logFile);
  void incrObjectiveCoeff(int columnNum, int valueToIncr);
  void addColumnVar(int boundKind, int lb, int ub, int objectiveCoeff = 0);
  void addRow(mlir::ArrayRef<int> rowCoeffs, int boundKind, int lb, int ub);
  llvm::Optional<int64_t> solve();
  void dumpInput();
  void dumpResult();
  int getNumCols();
  int64_t getColVarValue(int64_t col);

private:
  mlir::SmallVector<ColumnVarInfo> columnVars;
  mlir::SmallVector<RowVarInfo> rowVars;
  mlir::SmallVector<int, 4> ia;
  mlir::SmallVector<int, 4> ja;
  mlir::SmallVector<double, 4> ar;
  std::string ilpName;
  int optKind;
  glp_prob *mip;

public:
  std::string logFile;
};

/// OpInfo contains the list of parent loop initiation intervals and their
/// induction vars.
struct OpInfo {
  OpInfo(mlir::Operation *operation, int staticPos);
  OpInfo() { staticPos = -1; }
  mlir::Operation *getOperation();
  mlir::ArrayRef<mlir::AffineForOp> getParentLoops();
  mlir::ArrayRef<mlir::Value> getParentLoopIVs();
  int getStaticPosition();

private:
  mlir::SmallVector<mlir::AffineForOp> parentLoops;
  mlir::SmallVector<mlir::Value> parentLoopIVs;
  mlir::Operation *operation;
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
  int64_t insertRowCoefficients(mlir::SmallVectorImpl<int> &rowCoeffVec,
                                mlir::ArrayRef<int64_t> coeffs,
                                mlir::OperandRange memIndices,
                                mlir::ArrayRef<mlir::Value> loopIVs,
                                bool isNegativeCoeff);
  void addILPColumns();
  void addHappensBeforeConstraintRow();
  void addMemoryConstraintILPRows();

private:
  OpInfo fromInfo;
  OpInfo toInfo;
};

/// This struct holds information about which operations should be fused.
struct FusedOps {
  FusedOps(llvm::SmallVector<mlir::Operation *, 4> &operations,
           int64_t commonII, int64_t maxOpsPerCycle)
      : operations(operations), commonII(commonII),
        maxOpsPerCycle(maxOpsPerCycle) {}

  mlir::LogicalResult isSchedulable() {
    if (commonII * maxOpsPerCycle < (int64_t)operations.size())
      return mlir::failure();
    return mlir::success();
  }

  struct ILPVars {
    ILPVars(llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> cVars)
        : numOps(cVars.size()), numSlots(cVars[0].size()) {
      for (auto v : cVars)
        assert((int64_t)v.size() == numSlots);
    }

    int64_t getCVar(int64_t opNum, int64_t slot) { return cVars[opNum][slot]; }
    int64_t getRVar(int64_t opNum);
    int64_t getSlotVar(int64_t opNum);
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> cVars;
    llvm::SmallVector<int64_t, 4> rVars;
    llvm::SmallVector<int64_t, 4> slotVars;
    int64_t numOps;
    int64_t numSlots;
  };

  mlir::Operation *getOperation(int64_t opNum) { return operations[opNum]; }
  void setILPVars(ILPVars ilpVars) {
    assert(ilpVars.numOps = operations.size());
    assert(ilpVars.numSlots = commonII);
    this->ilpVars = ilpVars;
  }

private:
  llvm::SmallVector<mlir::Operation *, 4> operations;
  int64_t commonII;
  int64_t maxOpsPerCycle;
  llvm::Optional<ILPVars> ilpVars;
};

/// This class calculates the final schedule given the slack between memory
/// ops
// and the minimum delays between def and use.
class SchedulingILPHandler : ILPHandler {
public:
  SchedulingILPHandler(
      const mlir::SmallVector<mlir::Operation *> operations,
      const llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                           std::pair<int64_t, int64_t>>
          &mapMemoryDependenceToSlackAndDelay,
      const mlir::SmallVector<SSADependence> &ssaDependence,
      llvm::DenseMap<mlir::Value, mlir::ArrayAttr> &mapMemrefToPortsAttr,
      const std::string &logFile);
  void addILPColumns(llvm::raw_fd_ostream &);
  void addMemoryDependenceRows();
  void addSSADependenceRows();
  void addMaxTimeOffsetRows();
  void addFusedOpsConstraintRows(FusedOps group);
  llvm ::Optional<mlir::DenseMap<mlir::Operation *, int64_t>> getSchedule();
  llvm::DenseMap<mlir::Operation *, std::pair<int64_t, int64_t>>
  getPortAssignments();

private:
  const mlir::SmallVector<mlir::Operation *> operations;
  const llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                       std::pair<int64_t, int64_t>>
      &mapMemoryDependenceToSlackAndDelay;
  const mlir::SmallVector<SSADependence> ssaDependences;
  llvm::DenseMap<mlir::Value, mlir::ArrayAttr> mapMemrefToPortsAttr;
  llvm::DenseMap<mlir::Operation *, size_t> mapOperationToCol;
  // Pre-specified value for time variables.
  // FIXME: Implement this.
  llvm::DenseMap<mlir::Operation *, llvm::Optional<int64_t>>
      mapOperationToPreSpecifiedSchedule;
};
#endif
