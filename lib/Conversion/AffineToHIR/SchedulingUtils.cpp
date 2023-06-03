#include "circt/Conversion/SchedulingUtils.h"
#include "PragmaHandler.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ortools/linear_solver/linear_solver.h>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
using namespace mlir;
using namespace circt;
using namespace operations_research;
using std::to_string;

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------
void addCoeff(MPConstraint *constr, MPVariable *var, int64_t coeff) {
  auto prevCoeff = constr->GetCoefficient(var);
  constr->SetCoefficient(var, prevCoeff + coeff);
}
void addCoeff(MPObjective *constr, MPVariable *var, int64_t coeff) {
  auto prevCoeff = constr->GetCoefficient(var);
  constr->SetCoefficient(var, prevCoeff + coeff);
}

static Optional<std::tuple<int, int, int>> getConstantBounds(AffineForOp op) {
  int lb = op.getLowerBound().getMap().getSingleConstantResult();
  int ub = op.getUpperBound().getMap().getSingleConstantResult();
  int step = op.getStep();
  return std::make_tuple(lb, ub, step);
}
static Optional<std::tuple<int, int, int>> getConstantBounds(Value iv) {
  auto op = dyn_cast<mlir::AffineForOp>(iv.getParentRegion()->getParentOp());
  assert(op.getInductionVar() == iv);
  return getConstantBounds(op);
}

static SmallVector<Operation *, 4> getOpAndParents(Operation *operation) {
  SmallVector<Operation *, 4> opAndParents;
  while (!isa<mlir::func::FuncOp>(operation)) {
    opAndParents.push_back(operation);
    operation = operation->getParentOp();
  }
  return opAndParents;
}

//-----------------------------------------------------------------------------
// class OpInfo methods.
//-----------------------------------------------------------------------------
using namespace mlir;
OpInfo::OpInfo(Operation *operation, int staticPos)
    : operation(operation), staticPos(staticPos) {
  auto *currentOperation = operation;
  while (auto affineForOp = currentOperation->getParentOfType<AffineForOp>()) {
    parentLoops.push_back(affineForOp);
    parentLoopIVs.push_back(affineForOp.getInductionVar());
    currentOperation = affineForOp;
  }
}

Operation *OpInfo::getOperation() { return operation; }
int OpInfo::getStaticPosition() { return staticPos; }
size_t OpInfo::getNumParentLoops() { return parentLoops.size(); }
AffineForOp OpInfo::getParentLoop(int i) { return parentLoops[i]; }
Value OpInfo::getParentLoopIV(int i) { return parentLoopIVs[i]; }

//-----------------------------------------------------------------------------
// class MemOpInfo methods.
//-----------------------------------------------------------------------------
MemOpInfo::MemOpInfo(Operation *operation, int staticPos)
    : OpInfo(operation, staticPos) {
  assert(isa<AffineLoadOp>(operation) || isa<AffineStoreOp>(operation));
}

Value MemOpInfo::getMemRef() {
  if (auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation()))
    return storeOp.getMemRef();
  auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation());
  assert(loadOp);
  return loadOp.getMemRef();
}

int64_t MemOpInfo::getDelay() {
  auto pragmaHandler = MemrefPragmaHandler(this->getMemRef());
  if (isa<AffineLoadOp>(this->getOperation()))
    return pragmaHandler.getRdLatency();
  return pragmaHandler.getWrLatency();
}

bool MemOpInfo::isConstant() { return false; }

bool MemOpInfo::isLoad() { return isa<AffineLoadOp>(getOperation()); }

size_t MemOpInfo::getNumMemDims() {
  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation()))
    return loadOp.getAffineMap().getNumDims();

  auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation());
  assert(storeOp);
  return storeOp.getAffineMap().getNumDims();
}

int64_t MemOpInfo::getConstCoeff(int64_t dim) {
  SmallVector<int64_t> coeffs;
  AffineExpr expr;
  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation())) {
    expr = loadOp.getAffineMap().getResult(dim);
  } else {
    auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation());
    expr = storeOp.getAffineMap().getResult(dim);
  }

  // If the expression could not be flattened then treat the whole dim as zero
  // (i.e. all accesses alias on this dim).
  if (failed(getFlattenedAffineExpr(expr, getNumMemDims(), 0, &coeffs)))
    return 0;

  return coeffs.back();
}

int64_t MemOpInfo::getIdxCoeff(mlir::Value var, int64_t dim) {
  auto indices = getIndices();
  Optional<size_t> varLoc = llvm::None;
  AffineExpr expr;

  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation())) {
    expr = loadOp.getAffineMap().getResult(dim);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation())) {
    expr = storeOp.getAffineMap().getResult(dim);
  } else {
    llvm_unreachable("Must be a affine load or store op.");
  }

  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] == var) {
      varLoc = i;
      break;
    }
  }
  // var not in index is equivalent to saying coeff is zero.
  if (!varLoc)
    return 0;

  SmallVector<int64_t> coeffs;
  if (failed(getFlattenedAffineExpr(expr, getNumMemDims(), 0, &coeffs)))
    return 0;

  return coeffs[*varLoc];
}

llvm::SmallVector<mlir::Value> MemOpInfo::getIndices() {
  if (auto loadOp = dyn_cast<AffineLoadOp>(this->getOperation()))
    return loadOp.getIndices();

  auto storeOp = dyn_cast<AffineStoreOp>(this->getOperation());
  assert(storeOp);
  return storeOp.getIndices();
}
//-----------------------------------------------------------------------------
// ArithOpInfo.
//-----------------------------------------------------------------------------
ArithOpInfo::ArithOpInfo(mlir::Operation *operation) {
  assert(isa<arith::ArithmeticDialect>(operation->getDialect()));
  assert(!isa<arith::ConstantOp>(operation) && "arith.constant ");
  if (isa<arith::AddIOp>(operation)) {
    this->delay = 0;
  } else if (isa<arith::MulIOp>(operation)) {
    this->delay = 0;
  } else {
    operation->emitError();
    assert(false && "unsupported Arith operation");
  }
}

int64_t ArithOpInfo::getDelay() { return delay; }

bool ArithOpInfo::isConstant() {
  return isa<arith::ConstantOp>(getOperation());
}

//-----------------------------------------------------------------------------
// ILPSolver.
//-----------------------------------------------------------------------------

ILPSolver::ILPSolver(const char *name, llvm::raw_ostream &logger)
    : MPSolver(
          name,
          MPSolver::OptimizationProblemType::SCIP_MIXED_INTEGER_PROGRAMMING),
      logger(logger), solved(false) {}

void ILPSolver::dump() {

  if (this->Objective().minimization())
    logger << "Minimize: ";
  else if (this->Objective().maximization())
    logger << "Maximize: ";
  else
    llvm_unreachable("Objective not set.");
  for (auto *v : this->variables()) {
    int coeff = (long)this->Objective().GetCoefficient(v);
    if (coeff == 0)
      continue;
    if (coeff == 1)
      logger << " + " << v->name();
    else if (coeff == -1)
      logger << " - " << v->name();
    else if (coeff > 0)
      logger << " + " << coeff << v->name();
    else
      logger << " - " << -coeff << v->name();
  }

  logger << "\nBounds:\n";
  for (auto *v : this->variables()) {
    if (v->lb() > -infinity())
      logger << (long)v->lb() << "\t≤\t";
    logger << v->name();
    if (this->isSolved())
      logger << "(" << (long)v->solution_value() << ")";
    if (v->ub() < infinity())
      logger << "\t≤\t" << (long)v->ub();
    logger << "\n";
  }

  logger << "\nConstraints:\n";
  for (auto *constr : this->constraints()) {
    if (!constr->name().empty())
      logger << constr->name() << ": ";
    if (constr->lb() > -infinity())
      logger << (long)constr->lb() << "\t≤\t";
    bool constrAsAtleastOneVar = false;
    for (auto *v : this->variables()) {
      auto coeff = (long)constr->GetCoefficient(v);
      if (coeff == 0)
        continue;
      constrAsAtleastOneVar = true;
      if (coeff == 1)
        logger << " + " << v->name();
      else if (coeff == -1)
        logger << " - " << v->name();
      else if (coeff > 0)
        logger << " + " << coeff << v->name();
      else
        logger << " - " << -coeff << v->name();
    }
    if (!constrAsAtleastOneVar)
      logger << 0;
    if (constr->ub() < infinity())
      logger << "\t≤\t" << (long)constr->ub();
    logger << "\n";
  }
}

std::pair<MPVariable *, MPVariable *>
ILPSolver::addBoundedILPVar(double lb, double ub, int64_t step,
                            std::string &name) {
  assert(-infinity() < lb < infinity());
  assert(-infinity() < ub < infinity());
  assert(step != 0);
  auto *ilpVar = this->MakeIntVar(lb, ub, name);
  MPVariable *canonicalVar = ilpVar;
  if (step != 1) {
    canonicalVar = this->MakeIntVar(0, (ub - lb) / step, "cc_" + name);
    auto *constr = this->MakeRowConstraint(0, 0);
    addCoeff(constr, canonicalVar, step);
    addCoeff(constr, ilpVar, -1);
  }
  return std::make_pair(ilpVar, canonicalVar);
}

operations_research::MPVariable *
ILPSolver::getOrAddRemainder(operations_research::MPVariable *var,
                             int64_t divisor, const std::string &name) {
  auto key = std::make_pair(var, divisor);
  if (mapVarToRemainder[key] != NULL)
    return mapVarToRemainder[key];

  auto *constr = this->MakeRowConstraint(0, 0, "remainder");
  auto *quotient = this->MakeIntVar(0, infinity(), name + "_q");
  auto *remainder = this->MakeIntVar(0, divisor - 1, name);
  addCoeff(constr, quotient, divisor);
  addCoeff(constr, remainder, 1);
  mapVarToRemainder[key] = remainder;
  return remainder;
}

MPVariable *ILPSolver::getOrAddSum(
    llvm::SmallVector<operations_research::MPVariable *, 4> &vars,
    std::string &name) {
  if (mapVarsToSum.find(vars) != mapVarsToSum.end())

    return mapVarsToSum[vars];

  auto *constr = this->MakeRowConstraint(0, 0, "sum");
  double lb = 0;
  double ub = 0;
  for (auto *var : vars) {
    lb += var->lb();
    ub += var->ub();
    addCoeff(constr, var, 1);
  }
  auto *s = this->MakeIntVar(lb, ub, name);
  addCoeff(constr, s, -1);
  mapVarsToSum[vars] = s;
  return s;
}

MPVariable *ILPSolver::addConditionalGTE(operations_research::MPVariable *lhs,
                                         operations_research::MPVariable *rhs,
                                         int64_t lb, double m,
                                         const std::string &name) {

  auto *b = this->MakeBoolVar(name);

  // lhs - rhs >= lb - m + m*b
  auto *constr = this->MakeRowConstraint(lb - m, infinity(), "conditional-gte");
  addCoeff(constr, lhs, 1);
  addCoeff(constr, rhs, -1);
  addCoeff(constr, b, -m);
  return b;
}

//-----------------------------------------------------------------------------
// MemoryDependenceILPHandler.
//-----------------------------------------------------------------------------
MemoryDependenceILP::MemoryDependenceILP(MemOpInfo &src, MemOpInfo &dest,
                                         llvm::DenseSet<size_t> ignoredDims,
                                         llvm::raw_ostream &logger)
    : ILPSolver("MemoryDependenceILP", logger), src(src), dest(dest),
      ignoredDims(ignoredDims) {

  assert(this->src.getNumMemDims() == this->dest.getNumMemDims());
  // Add names of the ILP vars for the loop IVs.
  for (size_t i = 0; i < src.getNumParentLoops(); i++) {
    getOrAddBoundedILPSrcVar("s" + to_string(i),
                             src.getParentLoop(i).getInductionVar());
  }
  for (size_t i = 0; i < dest.getNumParentLoops(); i++) {
    getOrAddBoundedILPDestVar("d" + to_string(i),
                              dest.getParentLoop(i).getInductionVar());
  }
  addHappensBeforeConstraintRow();
  addMemoryConstraints();
  addObjective();
  this->MutableObjective()->SetMinimization();
}

operations_research::MPVariable *
MemoryDependenceILP::getOrAddBoundedILPSrcVar(std::string &&name,
                                              mlir::Value ssaVar) {

  assert(ssaVar);
  auto forOp = dyn_cast<AffineForOp>(ssaVar.getParentRegion()->getParentOp());
  assert(forOp);
  auto bounds = getConstantBounds(forOp);
  auto lb = std::get<0>(*bounds);
  auto ub = std::get<1>(*bounds);
  auto step = std::get<2>(*bounds);
  if (std::get<3>(mapValue2BoundedILPSrcVar[ssaVar]))
    return std::get<3>(mapValue2BoundedILPSrcVar[ssaVar]);

  assert(step != 0);
  // ILP expects inclusive ub. But ForOp ub is non-inclusive.
  auto vars = this->addBoundedILPVar(lb, ub - 1, step, name);
  auto *ilpVar = vars.first;
  mapValue2BoundedILPSrcVar[ssaVar] = std::make_tuple(lb, ub, step, ilpVar);
  return ilpVar;
}

operations_research::MPVariable *
MemoryDependenceILP::getOrAddBoundedILPDestVar(std::string &&name,
                                               mlir::Value ssaVar) {

  assert(ssaVar);
  auto forOp = dyn_cast<AffineForOp>(ssaVar.getParentRegion()->getParentOp());
  assert(forOp);
  auto bounds = getConstantBounds(forOp);
  auto lb = std::get<0>(*bounds);
  auto ub = std::get<1>(*bounds);
  auto step = std::get<2>(*bounds);
  if (std::get<3>(mapValue2BoundedILPDestVar[ssaVar]))
    return std::get<3>(mapValue2BoundedILPDestVar[ssaVar]);

  assert(step != 0);
  // ILP expects inclusive ub. But ForOp ub is non-inclusive.
  auto vars = this->addBoundedILPVar(lb, ub - 1, step, name);
  auto *ilpVar = vars.first;
  mapValue2BoundedILPDestVar[ssaVar] = std::make_tuple(lb, ub, step, ilpVar);
  return ilpVar;
}

std::stack<mlir::AffineForOp> MemoryDependenceILP::getCommonParentLoops() {
  std::stack<mlir::AffineForOp> commonLoops;
  for (size_t i = 0;
       i < std::min(src.getNumParentLoops(), dest.getNumParentLoops()); i++) {
    auto parent = src.getParentLoop(src.getNumParentLoops() - i - 1);
    if (parent != dest.getParentLoop(dest.getNumParentLoops() - i - 1))
      break;
    commonLoops.push(parent);
  }
  return commonLoops;
}

// Add constraint to ensure that the dynamic instance of the aliasing dest
// always occurs after src in the original sequential schedule.
void MemoryDependenceILP::addHappensBeforeConstraintRow() {
  if (dest.getOperation() == src.getOperation())
    assert(dest.getStaticPosition() == src.getStaticPosition());

  MPConstraint *constr;
  if (dest.getStaticPosition() > src.getStaticPosition()) {
    // If dest occurs after src in the original source code then the common
    // loops can all have same iv values.
    constr = this->MakeRowConstraint(0, infinity(), "happens-before");
  } else {
    // If dest occurs before src in the original source code then destination's
    // loop ivs must be lexicographically greater than the source.
    constr = this->MakeRowConstraint(1, infinity(), "happens-before");
  }

  int64_t coeff = 1;
  auto commonLoops = getCommonParentLoops();

  while (!commonLoops.empty()) {
    size_t loopNum = commonLoops.size() - 1;
    auto bounds = getConstantBounds(commonLoops.top());
    auto lb = std::get<0>(*bounds);
    auto ub = std::get<1>(*bounds);

    // FIXME: If we used the canonical ivs (step=1) of each iv which is
    // calculated in getOrAddBoundedILPSrcVar, then we can reduce the size of
    // coeff (canonical iv's bound is 0 to (ub-lb)/step). This helps if coeff
    // overflows due to very large bounds and step size.
    auto *destVar = this->getOrAddBoundedILPDestVar(
        "d" + to_string(loopNum), commonLoops.top().getInductionVar());
    addCoeff(constr, destVar, coeff);
    auto *srcVar = this->getOrAddBoundedILPSrcVar(
        "s" + to_string(loopNum), commonLoops.top().getInductionVar());
    addCoeff(constr, srcVar, -coeff);
    coeff *= (ub - lb);
    commonLoops.pop();
  }
}

// Add constraint for memory access address equality.
void MemoryDependenceILP::addMemoryConstraints() {
  auto srcIndices = src.getIndices();
  auto destIndices = dest.getIndices();
  for (size_t dim = 0; dim < src.getNumMemDims(); dim++) {
    if (ignoredDims.contains(dim))
      continue;
    auto constCoeffDifference =
        src.getConstCoeff(dim) - dest.getConstCoeff(dim);
    auto *constr =
        this->MakeRowConstraint(constCoeffDifference, constCoeffDifference,
                                "dim-equal:dim" + to_string(dim));

    for (size_t i = 0; i < destIndices.size(); i++) {
      auto idx = destIndices[i];
      auto coeff = dest.getIdxCoeff(idx, dim);
      auto *var = getOrAddBoundedILPDestVar("d" + to_string(i), idx);
      addCoeff(constr, var, coeff);
    }

    for (size_t i = 0; i < srcIndices.size(); i++) {
      auto idx = srcIndices[i];
      auto coeff = src.getIdxCoeff(idx, dim);
      auto bounds = getConstantBounds(idx);
      auto *var = getOrAddBoundedILPSrcVar("s" + to_string(i), idx);
      addCoeff(constr, var, -coeff);
    }
  }
}

void MemoryDependenceILP::addObjective() {
  for (size_t i = 0; i < src.getNumParentLoops(); i++) {
    auto parentLoop = src.getParentLoop(i);
    auto loopPragma = AffineForPragmaHandler(parentLoop);
    auto *iv =
        std::get<3>(mapValue2BoundedILPSrcVar[parentLoop.getInductionVar()]);
    assert(iv);
    addCoeff(this->MutableObjective(), iv, -loopPragma.getII());
  }
  for (size_t i = 0; i < dest.getNumParentLoops(); i++) {
    auto parentLoop = dest.getParentLoop(i);
    auto loopPragma = AffineForPragmaHandler(parentLoop);
    auto *iv =
        std::get<3>(mapValue2BoundedILPDestVar[parentLoop.getInductionVar()]);
    assert(iv);
    addCoeff(this->MutableObjective(), iv, loopPragma.getII());
  }
}

//-----------------------------------------------------------------------------
// SchedulingILPHandler.
//-----------------------------------------------------------------------------
Scheduler::Scheduler(llvm::raw_ostream &logger)
    : ILPSolver("SchedulingILP", logger), varNum(0) {

  this->MutableObjective()->SetMinimization();
}

Optional<operations_research::MPVariable *>
Scheduler::getILPVar(mlir::Operation *op) {

  if (this->mapOpToVar.find(op) == this->mapOpToVar.end())
    return llvm::None;
  return this->mapOpToVar[op];
}

operations_research::MPVariable *
Scheduler::getOrAddTimeOffset(mlir::Operation *op, std::string name) {
  auto var = getILPVar(op);
  if (var)
    return *var;
  this->mapOpToVar[op] = this->MakeIntVar(0, infinity(), name);
  return *getILPVar(op);
}

operations_research::MPVariable *
Scheduler::getOrAddTotalTimeOffset(mlir::Operation *operation,
                                   std ::string name) {

  SmallVector<MPVariable *, 4> timeOffsets;
  for (auto *op : getOpAndParents(operation)) {
    timeOffsets.push_back(
        this->getOrAddTimeOffset(op, "t" + to_string(this->varNum++)));
  }
  return this->getOrAddSum(timeOffsets, name);
}

operations_research::MPVariable *
Scheduler::getOrAddResourceAllocation(mlir::Operation *op, Resource *resource,
                                      const std ::string &name) {
  auto key = std::make_pair(op, resource);
  if (mapOpAndResourceToVar[key] != NULL)
    return mapOpAndResourceToVar[key];

  auto *p = this->MakeIntVar(0, resource->getNumResources() - 1, name);
  mapOpAndResourceToVar[key] = p;
  return p;
}

llvm::Optional<int64_t> Scheduler::getResourceAllocation(mlir::Operation *op,
                                                         Resource *resource) {

  auto key = std::make_pair(op, resource);
  if (mapOpAndResourceToVar[key] != NULL)
    return mapOpAndResourceToVar[key]->solution_value();
  return llvm::None;
}
void Scheduler::addDependence(Dependence dep) {
  // dest_time_offset - src_time_offset > delay.
  // delay can be -ve `dep` is a loop-carried dependence.
  auto *constr = this->MakeRowConstraint(dep.delay, infinity(), dep.name);

  // Add the time offsets of the dest op and all the enclosing regions to get
  // the total dest op time offset.
  for (auto *op : getOpAndParents(dep.dest))
    addCoeff(constr,
             this->getOrAddTimeOffset(op, "t" + to_string(this->varNum++)), 1);

  // Subtract the total src op time offset.
  for (auto *op : getOpAndParents(dep.src)) {
    addCoeff(constr,
             this->getOrAddTimeOffset(op, "t" + to_string(this->varNum++)), -1);
  }
}

void Scheduler::addConflict(Conflict conflict) {
  assert(conflict.op1 != conflict.op2);
  logger << "Potential port conflict between, \n";
  conflict.op1->print(logger);
  logger << "\nand, \n";
  conflict.op2->print(logger);
  logger << "\n";

  auto *ttOp1 =
      getOrAddTotalTimeOffset(conflict.op1, "tt" + to_string(this->varNum++));

  auto *ttOp2 =
      getOrAddTotalTimeOffset(conflict.op2, "tt" + to_string(this->varNum++));

  auto *r1 = this->getOrAddRemainder(ttOp1, conflict.commonII,
                                     "r" + to_string(this->varNum - 1));
  auto *p1 = this->getOrAddResourceAllocation(
      conflict.op1, conflict.resource, "p" + to_string(this->varNum - 1));

  auto *r2 = this->getOrAddRemainder(ttOp2, conflict.commonII,
                                     "r" + to_string(this->varNum));
  auto *p2 = this->getOrAddResourceAllocation(conflict.op2, conflict.resource,
                                              "p" + to_string(this->varNum));

  double m = 10000;
  auto *b1 = this->addConditionalGTE(ttOp2, ttOp1, conflict.depDelay, m,
                                     "b1_" + to_string(this->varNum));
  auto *b2 =
      this->addConditionalGTE(p1, p2, 1, m, "b2_" + to_string(this->varNum));
  auto *b3 =
      this->addConditionalGTE(p2, p1, 1, m, "b3_" + to_string(this->varNum));
  auto *b4 =
      this->addConditionalGTE(r1, r2, 1, m, "b4_" + to_string(this->varNum));
  auto *b5 =
      this->addConditionalGTE(r2, r1, 1, m, "b5_" + to_string(this->varNum));

  auto *constr = this->MakeRowConstraint(1, 1, "bank-conflict");
  addCoeff(constr, b1, 1);
  addCoeff(constr, b2, 1);
  addCoeff(constr, b3, 1);
  addCoeff(constr, b4, 1);
  addCoeff(constr, b5, 1);
}

void Scheduler::addDelayRegisterCost(Dependence dep, size_t width) {
  auto srcVar = this->getILPVar(dep.src);
  auto destVar = this->getILPVar(dep.dest);
  assert(srcVar);
  assert(destVar);
  addCoeff(this->MutableObjective(), *destVar, 1);
  auto srcCoeff = this->MutableObjective()->GetCoefficient(*srcVar);
  addCoeff(this->MutableObjective(), *srcVar, srcCoeff - 1);
}

int64_t Scheduler::getTimeOffset(mlir::Operation *op) {
  auto var = this->getILPVar(op);
  if (!var) {
    op->emitError(
        "Could not find time offset for this op. Setting it as zero.");
    return 0;
  }
  return (*var)->solution_value();
}
