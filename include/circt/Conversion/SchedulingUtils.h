#ifndef HIR_SCHEDULING_UTILS_H
#define HIR_SCHEDULING_UTILS_H
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "ortools/linear_solver/linear_solver.h"
#include <cstdint>
#include <stack>
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

mlir::Value getMemrefFromAffineLoadOrStoreOp(mlir::Operation *operation);

int64_t getMemOpSafeDelay(
    mlir::Operation *operation,
    mlir::DenseMap<mlir::Value, mlir::ArrayAttr> &mapMemrefToPortsAttr);

llvm::Optional<int64_t> getResultDelay(mlir::OpResult v);

void populateMemrefToPortsAttrMapping(
    mlir::func::FuncOp funcOp,
    llvm::DenseMap<mlir::Value, mlir::ArrayAttr> &mapMemrefToPortsAttr);

/// OpInfo contains the list of parent loop initiation intervals and their
/// induction vars.
struct OpInfo {
  OpInfo(mlir::Operation *operation, int staticPos);
  OpInfo() { staticPos = -1; }
  mlir::Operation *getOperation();
  int getStaticPosition();
  size_t getNumParentLoops();
  mlir::AffineForOp getParentLoop(int i);
  mlir::Value getParentLoopIV(int i);

private:
  mlir::SmallVector<mlir::AffineForOp> parentLoops;
  mlir::SmallVector<mlir::Value> parentLoopIVs;
  mlir::Operation *operation;
  int staticPos; // Represents the position of this op in the code. Used to
  // figure out if one op occurs before another in the static
  // order & to uniquely identify the static op.
};

struct MemOpInfo : public OpInfo {
  MemOpInfo(mlir::AffineLoadOp, int staticPos);
  MemOpInfo(mlir::AffineStoreOp, int staticPos);
  mlir::Value getMemRef();
  /// Get the number of dimensions of the MemRef.
  size_t getNumMemDims();
  llvm::SmallVector<mlir::Value> getIndices();
  int64_t getDelay();

  /// Get the coefficient for the 'var' in the flattened affine expression
  /// of the specific 'dim'.
  /// If the var is not in the expr it can not be flattened then it returns
  /// llvm::None.
  /// Ex: affine.load A[2i+j][i+3j]; getVarCoeff(i,0) == 2
  int64_t getIdxCoeff(mlir::Value idx, int64_t dim);
  int64_t getConstCoeff(int64_t dim);
  llvm::SmallVector<int64_t, 4>
  getIdxCoeffs(mlir::ArrayRef<mlir::Value> indices, int64_t dim);
};

struct ILPSolver : public operations_research::MPSolver {
  ILPSolver(const char *name);
  enum OptKind { MAX, MIN };

  std::pair<operations_research::MPVariable *,
            operations_research::MPVariable *>
  addBoundedILPVar(int64_t lb, int64_t ub, int64_t step, std::string &name);
  std::string dump();
  operations_research::MPVariable *addIntVar(int64_t lb, int64_t ub,
                                             std::string name = "");

private:
  int64_t varID;
};

/// This class solves the following minimization problem:
///   Objective = minimize (II_dest * I_dest - II_src*I_src).
/// * I_src is an array of induction vars of parent loops of the `src`
/// operation.
/// * II_src is an array of the corresponding loop initiation intervals.
class MemoryDependenceILPHandler : public ILPSolver {
public:
  MemoryDependenceILPHandler(MemOpInfo &src, MemOpInfo &dest);

private:
  operations_research::MPVariable *
  getOrAddBoundedILPSrcVar(int64_t lb, int64_t ub, int64_t step,
                           std::string &&name, mlir::Value var);
  operations_research::MPVariable *
  getOrAddBoundedILPDestVar(int64_t lb, int64_t ub, int64_t step,
                            std::string &&name, mlir::Value var);
  void addHappensBeforeConstraintRow();
  void addMemoryConstraints();
  void addObjective();
  std::stack<mlir::AffineForOp> getCommonParentLoops();

private:
  std::pair<int, llvm::SmallVector<int>>
  getIdxCoefficients(MemOpInfo memAccess, llvm::ArrayRef<mlir::Value> vars);

  MemOpInfo src;
  MemOpInfo dest;
  llvm::DenseMap<mlir::Value, std::tuple<int64_t, int64_t, int64_t,
                                         operations_research::MPVariable *>>
      mapValue2BoundedILPSrcVar;
  llvm::DenseMap<mlir::Value, std::tuple<int64_t, int64_t, int64_t,
                                         operations_research::MPVariable *>>
      mapValue2BoundedILPDestVar;
};

/// This struct manages the mapping of ILP variables required for op fusion
/// constraints to the corresponding column numbers in the ILP constraint
/// matrix. The ILP constraints are:
/// time = n*II+r
/// r = 0*c0 + 1*c1 + 2*c0
/// c0+c1+c2 = 1
/// 0<= c0,c1,c2<=1
// Memory or SSA dependence between two ops.
struct DependenceConstraint {
  DependenceConstraint(mlir::Operation *src, mlir::Operation *dest,
                       int64_t delay)
      : src(src), dest(dest), delay(delay) {}
  mlir::Operation *src;
  mlir::Operation *dest;
  int64_t delay;
};

/// This struct holds information about which operations should be fused.
struct ResourceConstraint {
  ResourceConstraint(llvm::SmallVector<mlir::Operation *, 4> &operations,
                     int64_t commonII, int64_t numResources)
      : operations(operations), commonII(commonII), numResources(numResources) {
    assert(operations.size() >= 2);
  }

  [[nodiscard]] bool isSchedulable() const {
    if (commonII * numResources < (int64_t)operations.size())
      return false;
    return true;
  }

  [[nodiscard]] mlir::ArrayRef<mlir::Operation *>
  getOperation(int64_t opNum) const {
    return operations;
  }
  [[nodiscard]] int64_t getCommonII() const { return commonII; }
  [[nodiscard]] int64_t getNumOps() const { return operations.size(); }
  [[nodiscard]] int64_t getNumResources() { return numResources; }

private:
  llvm::SmallVector<mlir::Operation *, 4> operations;
  int64_t commonII;
  int64_t numResources;
};

///  This class calculates the final schedule while minimizing the number of
///  delay registers required.
class Scheduler : ILPSolver {
public:
  Scheduler();
  void addDependenceConstraint(const DependenceConstraint dep);
  void addResourceConstraint(const ResourceConstraint resc);
  void addDelayRegisterCost(DependenceConstraint dep, size_t width);
  int64_t getTimeOffset(mlir::Operation *);
  int64_t getRequiredNumResources(ResourceConstraint);
  int64_t getResourceIdx(ResourceConstraint, mlir::Operation *);

private:
  llvm::Optional<operations_research::MPVariable *>
  getILPVar(mlir::Operation *op);
  operations_research::MPVariable *getOrAddTimeOffsetVar(mlir::Operation *op);

private:
  llvm::DenseMap<mlir::Operation *, operations_research::MPVariable *>
      mapOperationToVar;
  int64_t varNum;
};
#endif
