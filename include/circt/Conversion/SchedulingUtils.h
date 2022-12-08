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
#include <variant>

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

/// This struct manages the mapping of ILP variables required for op fusion
/// constraints to the corresponding column numbers in the ILP constraint
/// matrix. The ILP constraints are:
/// time = n*II+r
/// r = 0*c0 + 1*c1 + 2*c0
/// c0+c1+c2 = 1
/// 0<= c0,c1,c2<=1
struct FusedOp {
  FusedOp(llvm::StringRef instanceName,
          llvm::SmallVector<mlir::Operation *> &operations, int64_t commonII,
          int64_t maxOpsPerCycle)
      : instanceName(instanceName), operations(operations), commonII(commonII),
        maxOpsPerCycle(maxOpsPerCycle) {
    assert(operations.size() > 1);
    for (auto *operation : operations)
      assert(operation->getName() == operations[0]->getName());
  }

  [[nodiscard]] bool isSchedulable() const {
    if (commonII * maxOpsPerCycle < (int64_t)operations.size())
      return false;
    return true;
  }

  [[nodiscard]] mlir::Operation *getOperation(int64_t opNum) const {
    return operations[opNum];
  }
  [[nodiscard]] int64_t getCommonII() const { return commonII; }
  [[nodiscard]] int64_t getNumOps() const { return operations.size(); }
  llvm::StringRef getInstanceName() const { return instanceName; }
  int64_t getMaxOpsPerCycle() { return maxOpsPerCycle; }

private:
  llvm::StringRef instanceName;
  llvm::SmallVector<mlir::Operation *> operations;
  int64_t commonII;
  int64_t maxOpsPerCycle;
};
struct FusedOpInfo {
  FusedOpInfo(llvm::SmallVector<llvm::SmallVector<int64_t>> cVars,
              llvm::SmallVector<int64_t> dVars,
              llvm::SmallVector<int64_t> rVars)
      : cVars(cVars), dVars(dVars), rVars(rVars), numOps(cVars.size()),
        numSlots(cVars[0].size()) {
    for (auto v : cVars)
      assert((int64_t)v.size() == numSlots);
    assert((int64_t)dVars.size() == numOps);
    assert((int64_t)rVars.size() == numOps);
  }

  int64_t getCVar(int64_t opNum, int64_t slot) {
    auto col = cVars[opNum][slot];
    assert(col > 0);
    return col;
  }
  int64_t getRVar(int64_t opNum);
  int64_t getSlotVar(int64_t opNum);
  llvm::SmallVector<llvm::SmallVector<int64_t>> cVars;
  llvm::SmallVector<int64_t> dVars;
  llvm::SmallVector<int64_t> rVars;
  int64_t numOps;
  int64_t numSlots;
};
/// This struct holds information about which operations should be fused.

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
      const llvm::SmallVector<FusedOp> &fusedOps, const std::string &logFile);
  void addILPColumns(llvm::raw_fd_ostream &);
  void addMemoryDependenceRows();
  void addSSADependenceRows();
  void addMaxTimeOffsetRows();
  void addFusedOpConstraintRows();
  llvm ::Optional<mlir::DenseMap<mlir::Operation *, int64_t>> getSchedule();
  llvm::DenseMap<mlir::Operation *, std::pair<int64_t, int64_t>>
  getPortAssignments();

private:
  FusedOpInfo addFusedOpCols(const FusedOp &fusedOp, int64_t *col);
  void dumpFusedOpCols(llvm::raw_fd_ostream &os, const FusedOp &fusedOp,
                       const FusedOpInfo &fusedOpInfo);

private:
  const mlir::SmallVector<mlir::Operation *> operations;
  const llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                       std::pair<int64_t, int64_t>>
      &mapMemoryDependenceToSlackAndDelay;
  const mlir::SmallVector<SSADependence> ssaDependences;
  llvm::DenseMap<mlir::Value, mlir::ArrayAttr> mapMemrefToPortsAttr;
  llvm::DenseMap<mlir::Operation *, size_t> mapOperationToCol;
  const llvm::SmallVector<FusedOp> fusedOps;
  llvm::SmallVector<FusedOpInfo> fusedOpInfos;
  int64_t maxTimeCol;
};
#endif
