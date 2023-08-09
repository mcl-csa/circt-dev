#ifndef HIR_SCHEDULING_UTILS_H
#define HIR_SCHEDULING_UTILS_H
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "ortools/linear_solver/linear_solver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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

/// OpInfo contains the delay, list of parent loop initiation intervals and
/// their induction vars.
struct OpInfo {
  OpInfo(mlir::Operation *operation, int staticPos);
  OpInfo() { staticPos = -1; }
  mlir::Operation *getOperation();
  int getStaticPosition();
  size_t getNumParentLoops();
  mlir::AffineForOp getParentLoop(int i);
  mlir::Value getParentLoopIV(int i);
  virtual int64_t getDelay() = 0;
  virtual bool isConstant() = 0;
  virtual ~OpInfo(){};

private:
  mlir::SmallVector<mlir::AffineForOp> parentLoops;
  mlir::SmallVector<mlir::Value> parentLoopIVs;
  mlir::Operation *operation;
  int staticPos; // Represents the position of this op in the code. Used to
  // figure out if one op occurs before another in the static
  // order & to uniquely identify the static op.
};

struct MemOpInfo : OpInfo {
  MemOpInfo(mlir::Operation *, int staticPos);
  ~MemOpInfo() override {}
  mlir::Value getMemRef();
  /// Get the number of dimensions of the MemRef.
  size_t getNumMemDims();
  mlir::AffineMap getAffineMap();
  llvm::SmallVector<mlir::Value> getIndices();
  int64_t getDelay() override;
  bool isConstant() override;
  bool isLoad();
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

struct ArithOpInfo : OpInfo {
  ArithOpInfo(mlir::Operation *operation);
  ~ArithOpInfo() override {}
  int64_t getDelay() override;
  bool isConstant() override;

private:
  int64_t delay;
};

struct ILPSolver : public operations_research::MPSolver {
  ILPSolver(const char *name, llvm::raw_ostream &logger);

  std::pair<operations_research::MPVariable *,
            operations_research::MPVariable *>
  addBoundedILPVar(double lb, double ub, int64_t step, std::string &name);
  void dump();
  ResultStatus solve() {
    auto status = this->Solve();
    this->solved = (status == ResultStatus::OPTIMAL);
    return status;
  }
  bool isSolved() { return solved; };

  [[nodiscard("Unused remainder: Generating unnecessary constraints and "
              "vars.")]] operations_research::MPVariable *
  getOrAddRemainder(operations_research::MPVariable *var, int64_t divisor,
                    const std::string &name);

  [[nodiscard("Unused sum: Generating unnecessary constraints and "
              "vars.")]] operations_research::MPVariable *
  getOrAddSum(llvm::SmallVector<operations_research::MPVariable *, 4> &vars,
              std::string &name);

  /// Result is a boolean variable (lets say b). m should be a large number.
  /// b == 1 => lhs - rhs >= lb
  /// b == 0 => lhs - rhs >= lb - m

  operations_research::MPVariable *
  addConditionalGTE(operations_research::MPVariable *lhs,
                    operations_research::MPVariable *rhs, int64_t lb, double m,
                    const std::string &name);

private:
  std::map<llvm::SmallVector<operations_research::MPVariable *, 4>,
           operations_research::MPVariable *>
      mapVarsToSum;
  llvm::DenseMap<std::pair<operations_research::MPVariable *, int64_t>,
                 operations_research::MPVariable *>
      mapVarToRemainder;

protected:
  llvm::raw_ostream &logger;
  bool solved;
};

/// This class solves the following minimization problem:
///   Objective = minimize (II_dest * I_dest - II_src*I_src).
/// * I_src is an array of induction vars of parent loops of the `src`
/// operation.
/// * II_src is an array of the corresponding loop initiation intervals.
class MemoryDependenceILP : public ILPSolver {
public:
  MemoryDependenceILP(MemOpInfo &src, MemOpInfo &dest,
                      llvm ::DenseSet<size_t> ignoredDims,
                      llvm::raw_ostream &logger);

private:
  operations_research::MPVariable *getOrAddBoundedILPSrcVar(std::string &&name,
                                                            mlir::Value var);
  operations_research::MPVariable *getOrAddBoundedILPDestVar(std::string &&name,
                                                             mlir::Value var);
  void addHappensBeforeConstraintRow();
  void addMemoryConstraints();
  void addObjective();
  std::stack<mlir::AffineForOp> getCommonParentLoops();

private:
  using ILPSolver::logger;
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
  llvm::DenseSet<size_t> ignoredDims;
};

/// This struct manages the mapping of ILP variables required for op fusion
/// constraints to the corresponding column numbers in the ILP constraint
/// matrix. The ILP constraints are:
/// time = n*II+r
/// r = 0*c0 + 1*c1 + 2*c0
/// c0+c1+c2 = 1
/// 0<= c0,c1,c2<=1
// Memory or SSA dependence between two ops.
struct Dependence {
  Dependence(std::string &&name, mlir::Operation *src, mlir::Operation *dest,
             int64_t delay)
      : name(name), src(src), dest(dest), delay(delay) {}
  const std::string name;
  mlir::Operation *const src;
  mlir::Operation *const dest;
  const int64_t delay;
};

struct Resource {
  virtual size_t getNumResources() = 0;
  virtual ~Resource() {}
};

struct MemPortResource : public Resource {
  MemPortResource(mlir::Value mem, size_t numPorts)
      : mem(mem), numPorts(numPorts) {}
  virtual size_t getNumResources() override { return numPorts; }
  virtual ~MemPortResource() override {}

  mlir::Value mem;
  size_t numPorts;
};

/// This struct captures the information about two resource-conflicting
/// operations.
struct Conflict {
  Conflict(mlir::Operation *op1, mlir::Operation *op2, int64_t commonII,
           Resource *resource, int64_t depDelay)
      : op1(op1), op2(op2), commonII(commonII), resource(resource),
        depDelay(depDelay) {

    // If its the same operation then op2 occurs after op1 in the parallel
    // schedule as well. Thus dist =min(t_d - t_s) must be +ve.
    // Thus depDelay=1-dist, which signifies how much we need to push the dest
    // to ensure it occurs one cycle after source is  <= 0.
    // i.e. even if you pull the dest behind (in case of +ve dist) by reducing
    // the loop initiation interval(thats the only way since both the source
    // and dest op are the same), you still can schedule the dest op one cycle
    // after the source op.
    assert(op1 != op2);
  }

  mlir::Operation *const op1, *const op2;
  const int64_t commonII;
  Resource *const resource;
  /// The required delay if we assume that there is a true dependence from op1
  /// to op2.
  int64_t depDelay;
};

///  This class calculates the final schedule while minimizing the required
///  number of delay registers.
class Scheduler : public ILPSolver {
public:
  Scheduler(llvm::raw_ostream &logger);
  void addDependence(Dependence dep);
  void addConflict(Conflict conflict);
  void addDelayRegisterCost(Dependence dep, size_t width);
  int64_t getTimeOffset(mlir::Operation *);
  llvm::Optional<int64_t> getResourceAllocation(mlir::Operation *op,
                                                Resource *resource);

private:
  llvm::Optional<operations_research::MPVariable *>
  getILPVar(mlir::Operation *op);
  operations_research::MPVariable *getOrAddTimeOffset(mlir::Operation *op,
                                                      std ::string name);
  operations_research::MPVariable *getOrAddTotalTimeOffset(mlir::Operation *op,
                                                           std ::string name);
  operations_research::MPVariable *
  getOrAddResourceAllocation(mlir::Operation *op, Resource *resource,
                             const std ::string &name);

protected:
  using ILPSolver::logger;

private:
  size_t varNum;
  llvm::DenseMap<mlir::Operation *, operations_research::MPVariable *>
      mapOpToVar;

  // Currently only op as key should have been enough since we have only one
  // type of resource (memory ports) which is unique for each operation. But in
  // the future operations may be associated with multiple types of resources.
  llvm::DenseMap<std::pair<mlir::Operation *, Resource *>,
                 operations_research::MPVariable *>
      mapOpAndResourceToVar;
  operations_research::MPVariable *tmax;
};
#endif