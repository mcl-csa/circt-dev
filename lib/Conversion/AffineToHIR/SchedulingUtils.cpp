#include "circt/Conversion/SchedulingUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FileSystem.h"
#include <cstdint>
#include <ortools/linear_solver/linear_solver.h>
#include <string>
#include <tuple>
using namespace mlir;
using namespace circt;
using namespace operations_research;
using std::to_string;
// Local functions.
llvm::Optional<int>
getMemrefPortFromAffineLoadOrStoreOpAttr(mlir::Operation *operation) {
  assert(isa<mlir::AffineLoadOp>(operation) ||
         isa<mlir::AffineStoreOp>(operation));
  auto portNumAttr = operation->getAttrOfType<IntegerAttr>("hir.memref_port");
  if (!portNumAttr)
    return llvm::None;
  return portNumAttr.getInt();
}

static AffineMap getAffineMapForMemoryAccess(Operation *operation) {

  if (auto affineLoadOp = dyn_cast<AffineLoadOp>(operation)) {
    return affineLoadOp.getAffineMap();
  }
  auto affineStoreOp = dyn_cast<AffineStoreOp>(operation);

  assert(affineStoreOp);
  return affineStoreOp.getAffineMap();
}

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------

static int getLoopII(AffineForOp affineForOp) {
  assert(affineForOp->hasAttrOfType<mlir::IntegerAttr>("II"));
  return affineForOp->getAttrOfType<mlir::IntegerAttr>("II").getInt();
}

Value getMemrefFromAffineLoadOrStoreOp(Operation *operation) {
  assert(isa<AffineLoadOp>(operation) || isa<AffineStoreOp>(operation));
  if (auto storeOp = dyn_cast<AffineStoreOp>(operation))
    return storeOp.getMemRef();
  auto loadOp = dyn_cast<AffineLoadOp>(operation);
  return loadOp.getMemRef();
}

int64_t getMemOpSafeDelay(Operation *operation,
                          DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr) {
  if (auto affineStoreOp = dyn_cast<mlir::AffineStoreOp>(operation)) {
    auto ports = mapMemrefToPortsAttr[affineStoreOp.getMemRef()];
    for (auto port : ports) {
      if (helper::isMemrefWrPort(port)) {
        return helper::getMemrefPortWrLatency(port).getValue();
      }
    }
    operation->emitError("Could not find memref wr port");
    llvm_unreachable("Expected a write port.");
  }
  assert(isa<mlir::AffineLoadOp>(operation));
  // If the source op is a load then a store op can be scheduled in the same
  // cycle.
  return 0;
}

llvm::Optional<int64_t> getResultDelay(OpResult v) {
  auto *operation = v.getOwner();
  auto resultDelays = operation->getAttrOfType<ArrayAttr>("result_delays");
  if (!resultDelays) {
    return llvm::None;
  }
  if (resultDelays.size() != operation->getNumResults()) {
    return llvm::None;
  }
  auto delayAttr = resultDelays[v.getResultNumber()];
  if (!delayAttr.isa<IntegerAttr>()) {
    return llvm::None;
  }
  return delayAttr.dyn_cast<IntegerAttr>().getInt();
}

void populateMemrefToPortsAttrMapping(
    mlir::func::FuncOp funcOp,
    llvm::DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr) {
  funcOp.walk([&mapMemrefToPortsAttr](Operation *operation) {
    if (auto funcOp = dyn_cast<mlir::func::FuncOp>(operation)) {
      auto argAttrs = funcOp.getAllArgAttrs();
      auto args = funcOp.getBody().front().getArguments();
      assert(args.size() == argAttrs.size());
      if (argAttrs) {
        for (size_t i = 0; i < args.size(); i++) {
          if (args[i].getType().isa<mlir::MemRefType>()) {
            mapMemrefToPortsAttr[args[i]] =
                helper::extractMemrefPortsFromDict(
                    argAttrs[i].dyn_cast<DictionaryAttr>())
                    .getValue();
          }
        }
      }
    } else if (auto allocaOp = dyn_cast<mlir::memref::AllocaOp>(operation)) {
      auto ports =
          helper::extractMemrefPortsFromDict(allocaOp->getAttrDictionary())
              .getValue();
      mapMemrefToPortsAttr[allocaOp.getResult()] = ports;
    }
  });
}

// We need this because if the use is inside a nested region-op then the ssa var
// will be captured by this op (Ex: user inside forOp) using iter_args in hir
// dialect. Thus, we consider the enclosing parentOp (Ex: forOp) as the
// dependent op instead of the original user.
static Operation *getTopLevelDependentOp(Operation *commonParentOp,
                                         Operation *user) {
  Operation *actualDependentOp = user;
  while (actualDependentOp->getParentOp() != commonParentOp) {
    actualDependentOp = actualDependentOp->getParentOp();
  }
  return actualDependentOp;
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

//-----------------------------------------------------------------------------
// ILPSolver.
//-----------------------------------------------------------------------------

ILPSolver::ILPSolver(const char *name)
    : MPSolver(
          name,
          MPSolver::OptimizationProblemType::SCIP_MIXED_INTEGER_PROGRAMMING) {}

std::string ILPSolver::dump() {
  this->Name();
  std::string debugStr;
  assert(!this->Objective().terms().empty());
  assert(this->variables().size() > 0);

  debugStr = "\nObjective:\n\t\t";
  for (auto *v : this->variables()) {
    auto coeff = this->Objective().GetCoefficient(v);
    if (coeff != 0)
      debugStr +=
          (coeff > 0 ? " +" : " -") + to_string(coeff) + v->name() + "\n";
  }

  debugStr += "\nBounds:\n\t\t";
  for (auto *v : this->variables()) {
    debugStr += to_string(v->lb()) + "\t\t≤\t\t" + v->name() + "\t\t≤\t\t" +
                to_string(v->ub()) + "\n";
  }

  debugStr += "\nConstraints:\n\t\t";
  for (auto *constr : this->constraints()) {
    for (auto *v : this->variables()) {
      auto coeff = constr->GetCoefficient(v);
      if (coeff != 0)
        debugStr +=
            (coeff > 0 ? " +" : " -") + to_string(coeff) + v->name() + "\n";
    }
  }

  return debugStr;
}

std::pair<MPVariable *, MPVariable *>
ILPSolver::addBoundedILPVar(int64_t lb, int64_t ub, int64_t step,
                            std::string &name) {
  assert(step != 0);
  auto ilpVar = this->MakeIntVar(lb, ub, name);
  MPVariable *canonicalVar = ilpVar;
  if (step != 1) {
    canonicalVar = this->MakeIntVar(0, (ub - lb) / step, "cc_" + name);
    auto constr = this->MakeRowConstraint(0, 0);
    constr->SetCoefficient(canonicalVar, step);
    constr->SetCoefficient(ilpVar, -1);
  }
  return std::make_pair(ilpVar, canonicalVar);
}

operations_research::MPVariable *ILPSolver::addIntVar(int64_t lb, int64_t ub,
                                                      std::string name) {
  if (name == "")
    name = "c" + std::to_string(varID++);
  return this->MakeIntVar(lb, ub, name);
}

//-----------------------------------------------------------------------------
// MemoryDependenceILPHandler.
//-----------------------------------------------------------------------------
MemoryDependenceILPHandler::MemoryDependenceILPHandler(MemOpInfo &src,
                                                       MemOpInfo &dest)
    : ILPSolver("MemoryDependenceILP"), src(src), dest(dest) {

  assert(this->src.getNumMemDims() == this->dest.getNumMemDims());
  addHappensBeforeConstraintRow();
  addMemoryConstraints();
}

operations_research::MPVariable *
MemoryDependenceILPHandler::getOrAddBoundedILPSrcVar(int64_t lb, int64_t ub,
                                                     int64_t step,
                                                     std::string &&name,
                                                     mlir::Value ssaVar) {

  assert(ssaVar);
  if (std::get<3>(mapValue2BoundedILPSrcVar[ssaVar]))
    return std::get<3>(mapValue2BoundedILPSrcVar[ssaVar]);

  assert(step != 0);
  auto vars = this->addBoundedILPVar(lb, ub, step, name);
  auto *ilpVar = vars.first;
  mapValue2BoundedILPSrcVar[ssaVar] = std::make_tuple(lb, ub, step, ilpVar);
  return ilpVar;
}

operations_research::MPVariable *
MemoryDependenceILPHandler::getOrAddBoundedILPDestVar(int64_t lb, int64_t ub,
                                                      int64_t step,
                                                      std::string &&name,
                                                      mlir::Value ssaVar) {

  assert(ssaVar);
  if (std::get<3>(mapValue2BoundedILPDestVar[ssaVar]))
    return std::get<3>(mapValue2BoundedILPDestVar[ssaVar]);

  assert(step != 0);
  auto vars = this->addBoundedILPVar(lb, ub, step, name);
  auto ilpVar = vars.first;
  mapValue2BoundedILPDestVar[ssaVar] = std::make_tuple(lb, ub, step, ilpVar);
  return ilpVar;
}

std::stack<mlir::AffineForOp>
MemoryDependenceILPHandler::getCommonParentLoops() {
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
void MemoryDependenceILPHandler::addHappensBeforeConstraintRow() {
  if (dest.getOperation() == src.getOperation())
    assert(dest.getStaticPosition() == src.getStaticPosition());

  MPConstraint *constr;
  if (dest.getStaticPosition() > src.getStaticPosition()) {
    // If dest occurs after src in the original source code then the common
    // loops can all have same iv values.
    constr = this->MakeRowConstraint(0, infinity(), "hb");
  } else {
    // If dest occurs before src in the original source code then destination's
    // loop ivs must be lexicographically greater than the source.
    constr = this->MakeRowConstraint(1, infinity(), "hb");
  }

  int64_t coeff = 1;
  auto commonLoops = getCommonParentLoops();
  for (size_t i = 1; i < commonLoops.size(); i++) {
    auto bounds = getConstantBounds(commonLoops.top());
    auto lb = std::get<0>(*bounds);
    auto ub = std::get<1>(*bounds);
    auto step = std::get<2>(*bounds);

    // FIXME: If we used the canonical ivs (step=1) of each iv which is
    // calculated in getOrAddBoundedILPSrcVar, then we can reduce the size of
    // coeff (canonical iv's bound is 0 to (ub-lb)/step). This helps if coeff
    // overflows due to very large bounds and step size.
    auto destVar = this->getOrAddBoundedILPDestVar(
        lb, ub, step, "d" + to_string(i), commonLoops.top().getInductionVar());
    constr->SetCoefficient(destVar, coeff);
    auto srcVar = this->getOrAddBoundedILPSrcVar(
        lb, ub, step, "s" + to_string(i), commonLoops.top().getInductionVar());
    constr->SetCoefficient(srcVar, -coeff);
    coeff *= (ub - lb);
    commonLoops.pop();
  }
}

// Add constraint for memory access address equality.
void MemoryDependenceILPHandler::addMemoryConstraints() {
  auto srcIndices = src.getIndices();
  auto destIndices = dest.getIndices();
  for (size_t dim = 0; dim < src.getNumMemDims(); dim++) {
    auto constCoeffDifference =
        src.getConstCoeff(dim) - dest.getConstCoeff(dim);
    auto *constr = this->MakeRowConstraint(
        constCoeffDifference, constCoeffDifference, "dim" + to_string(dim));

    for (size_t i = 0; i < destIndices.size(); i++) {
      auto idx = destIndices[i];
      auto coeff = dest.getIdxCoeff(idx, dim);
      auto bounds = getConstantBounds(idx);
      auto *var = getOrAddBoundedILPDestVar(
          std::get<0>(*bounds), std::get<1>(*bounds), std::get<2>(*bounds),
          "d" + to_string(i), idx);
      constr->SetCoefficient(var, coeff);
    }

    for (size_t i = 0; i < srcIndices.size(); i++) {
      auto idx = srcIndices[i];
      auto coeff = src.getIdxCoeff(idx, dim);
      auto bounds = getConstantBounds(idx);
      auto *var = getOrAddBoundedILPSrcVar(
          std::get<0>(*bounds), std::get<1>(*bounds), std::get<2>(*bounds),
          "s" + to_string(i), idx);
      constr->SetCoefficient(var, -coeff);
    }
  }
}

void MemoryDependenceILPHandler::addObjective() {
  int64_t coeff = 1;
  auto commonLoops = getCommonParentLoops();
  for (size_t i = 0; i < commonLoops.size(); i++) {
    auto ii = getLoopII(commonLoops.top());
    coeff *= ii;
    commonLoops.pop();

    auto *destVar = std::get<3>(
        mapValue2BoundedILPDestVar[commonLoops.top().getInductionVar()]);
    assert(destVar);
    this->MutableObjective()->SetCoefficient(destVar, coeff);

    auto *srcVar = std::get<3>(
        mapValue2BoundedILPSrcVar[commonLoops.top().getInductionVar()]);
    assert(srcVar);
    this->MutableObjective()->SetCoefficient(srcVar, -coeff);
    this->MutableObjective()->SetMinimization();
  }
}

//-----------------------------------------------------------------------------
// SchedulingILPHandler.
//-----------------------------------------------------------------------------
Scheduler::Scheduler() : ILPSolver("SchedulingILP") {}

SmallVector<Operation *, 4> getOpAndParents(Operation *operation) {
  SmallVector<Operation *, 4> opAndParents;
  while (!isa<mlir::func::FuncOp>(operation)) {
    opAndParents.push_back(operation);
    operation = operation->getParentOp();
  }
  return opAndParents;
}

Optional<operations_research::MPVariable *>
Scheduler::getILPVar(mlir::Operation *op) {

  if (this->mapOperationToVar.find(op) == this->mapOperationToVar.end())
    return llvm::None;
  return this->mapOperationToVar[op];
}

operations_research::MPVariable *
Scheduler::getOrAddTimeOffsetVar(mlir::Operation *op) {
  auto var = getILPVar(op);
  if (var)
    return *var;
  return this->addIntVar(0, infinity());
}

void Scheduler::addDependenceConstraint(const DependenceConstraint dep) {
  auto *constr = this->MakeRowConstraint();

  // Add the time offsets of the dest op and all the enclosing regions to get
  // the total dest op time offset.
  for (auto *op : getOpAndParents(dep.dest))
    constr->SetCoefficient(this->getOrAddTimeOffsetVar(op), 1);

  // Subtract the total src op time offset.
  for (auto *op : getOpAndParents(dep.src))
    constr->SetCoefficient(this->getOrAddTimeOffsetVar(op), -1);

  // dest_time_offset - src_time_offset > delay.
  // delay can be -ve `dep` is a loop-carried dependence.
  constr->SetBounds(dep.delay, infinity());
}

void Scheduler::addResourceConstraint(const ResourceConstraint resc) {}

void Scheduler::addDelayRegisterCost(DependenceConstraint dep, size_t width) {
  auto srcVar = this->getILPVar(dep.src);
  auto destVar = this->getILPVar(dep.dest);
  assert(srcVar);
  assert(destVar);
  this->MutableObjective()->SetCoefficient(*destVar, 1);
  this->MutableObjective()->SetCoefficient(*srcVar, -1);
}

int64_t Scheduler::getTimeOffset(mlir::Operation *op) {
  auto var = this->getILPVar(op);
  assert(var);
  return (*var)->solution_value();
}

int64_t Scheduler::getRequiredNumResources(ResourceConstraint) {}

int64_t Scheduler::getResourceIdx(ResourceConstraint, mlir::Operation *) {}