#include "circt/Conversion/SchedulingUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include <cstdint>
#include <tuple>
using namespace mlir;
using namespace circt;
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

static OperandRange getMemIndices(Operation *operation) {
  if (auto affineLoadOp = dyn_cast<AffineLoadOp>(operation)) {
    return affineLoadOp.getIndices();
  }
  if (auto affineStoreOp = dyn_cast<AffineStoreOp>(operation)) {
    return affineStoreOp.getIndices();
  }
  llvm_unreachable("Operation should be AffineLoadOp or AffineStoreOp.");
}

static llvm::Optional<int> findLocInArray(Value iv, OperandRange indices) {
  for (size_t i = 0; i < indices.size(); i++) {
    if (iv == indices[i])
      return i;
  }
  return llvm::None;
}

//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------

int getLoopII(AffineForOp affineForOp) {
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

LogicalResult
populateSSADependences(mlir::func::FuncOp funcOp,
                       SmallVector<SSADependence> &ssaDependences) {
  funcOp.walk([&ssaDependences](Operation *operation) {
    if (isa<arith::ConstantOp, mlir::memref::AllocaOp>(operation))
      return WalkResult::advance();
    for (OpResult const result : operation->getResults()) {
      assert(!result.getType().isa<MemRefType>());
      auto delay = getResultDelay(result);
      if (!delay.hasValue()) {
        operation->emitError("Could not calculate result delay.");
      }
      for (auto *user : result.getUsers()) {
        // FIXME: getTopLevelDependentOp should not be required.
        auto *dependentOp =
            getTopLevelDependentOp(operation->getParentOp(), user);
        ssaDependences.push_back(
            SSADependence(dependentOp, result, delay.getValue()));
      }
    }
    for (auto &region : operation->getRegions()) {
      for (Value const regionArg : region.getArguments()) {
        if (regionArg.getType().isa<mlir::MemRefType>())
          continue;
        for (auto *user : regionArg.getUsers()) {
          auto *dependentOp = getTopLevelDependentOp(operation, user);
          ssaDependences.push_back(SSADependence(dependentOp, regionArg, 0));
        }
      }
    }
    return WalkResult::advance();
  });
  return success();
}

//-----------------------------------------------------------------------------
// ILPHandler methods.
//-----------------------------------------------------------------------------
ILPHandler::ILPHandler(const char *ilpName, int optKind,
                       const std::string &logFile)
    : ilpName(ilpName), optKind(optKind), mip(NULL), logFile(logFile) {
  ar.push_back(-1);
  ia.push_back(-1);
  ja.push_back(-1);
}

void ILPHandler::addRow(ArrayRef<int> rowCoeffs, int boundKind, int lb,
                        int ub) {
  assert(boundKind == GLP_DB || lb == ub);
  assert(rowCoeffs.size() == columnVars.size() &&
         "Wrong number of row coefficients.");
  for (size_t j = 0; j < rowCoeffs.size(); j++) {
    auto coeff = rowCoeffs[j];
    if (coeff != 0) {
      ia.push_back(rowVars.size() + 1);
      ja.push_back(j + 1);
      ar.push_back((double)coeff);
    }
  }

  std::string const name = "r" + std::to_string(rowVars.size() + 1);
  rowVars.push_back({.name = name, .boundKind = boundKind, .lb = lb, .ub = ub});
}
void ILPHandler::incrObjectiveCoeff(int columnNum, int valueToIncr) {
  columnVars[columnNum].objectiveCoeff += valueToIncr;
}

void ILPHandler::addColumnVar(int boundKind, int lb, int ub, int objCoeff) {
  assert(rowVars.size() == 0 &&
         "All columns must be added before adding rows.");

  std::string const name = "c" + std::to_string(columnVars.size() + 1);
  columnVars.push_back({.name = name,
                        .boundKind = boundKind,
                        .lb = lb,
                        .ub = ub,
                        .objectiveCoeff = objCoeff});
}

llvm::Optional<int64_t> ILPHandler::solve() {
  mip = glp_create_prob();
  glp_set_prob_name(mip, ilpName.c_str());
  glp_set_obj_dir(mip, optKind);

  // Set the rows.
  if (!rowVars.empty()) {
    glp_add_rows(mip, rowVars.size());
    for (size_t i = 0; i < rowVars.size(); i++) {
      glp_set_row_name(mip, i + 1, rowVars[i].name.c_str());
      glp_set_row_bnds(mip, i + 1, rowVars[i].boundKind, rowVars[i].lb,
                       rowVars[i].ub);
    }
  }

  // Set the columns.
  if (!columnVars.empty()) {
    glp_add_cols(mip, columnVars.size());
    for (size_t j = 0; j < columnVars.size(); j++) {
      glp_set_col_name(mip, j + 1, columnVars[j].name.c_str());
      glp_set_col_bnds(mip, j + 1, columnVars[j].boundKind, columnVars[j].lb,
                       columnVars[j].ub);
      glp_set_obj_coef(mip, j + 1, columnVars[j].objectiveCoeff);
      glp_set_col_kind(mip, j + 1, GLP_IV);
    }
  }

  glp_load_matrix(mip, ia.size() - 1, ia.data(), ja.data(), ar.data());

  // Solve the MIP.

  dumpInput();
  fflush(stdout);
  glp_term_out(GLP_OFF);
  glp_iocp parm;
  glp_init_iocp(&parm);
  parm.presolve = GLP_ON;
  glp_simplex(mip, NULL);
  if (glp_get_status(mip) != GLP_OPT)
    return llvm::None;
  glp_intopt(mip, &parm);
  if (glp_mip_status(mip) == GLP_NOFEAS)
    return llvm::None;
  assert(glp_mip_status(mip) == GLP_OPT);
  double d = glp_mip_obj_val(mip);
  dumpResult();
  return (int64_t)d;
}

void ILPHandler::dumpInput() {
  assert(mip);
  if (logFile == "/dev/null")
    return;
  glp_write_lp(mip, NULL, logFile.c_str());
}
void ILPHandler::dumpResult() {
  assert(mip);
  if (logFile == "/dev/null")
    return;
  glp_print_mip(mip, logFile.c_str());
}

int ILPHandler::getNumCols() { return (int)columnVars.size(); }

int64_t ILPHandler::getColVarValue(int64_t col) {
  assert(col > 0);
  return (int64_t)glp_mip_col_val(mip, col);
}

//-----------------------------------------------------------------------------
// MemoryDependenceILPHandler methods.
//-----------------------------------------------------------------------------
MemoryDependenceILPHandler::MemoryDependenceILPHandler(OpInfo fromInfo,
                                                       OpInfo toInfo,
                                                       std::string &logFile)
    : ILPHandler("MemoryDependenceILP", GLP_MIN, logFile), fromInfo(fromInfo),
      toInfo(toInfo) {}

llvm::Optional<int64_t> MemoryDependenceILPHandler::calculateSlack() {
  addILPColumns();
  addHappensBeforeConstraintRow();
  addMemoryConstraintILPRows();
  std::error_code ec;
  llvm::raw_fd_ostream os(logFile.c_str(), ec,
                          llvm::sys::fs::OpenFlags::OF_Append);
  os.changeColor(llvm::raw_ostream::Colors::GREEN);
  os << "\n\n-------------Calculating Min iter time "
        "dist.---------------------\n\n";
  os.changeColor(llvm::raw_ostream::Colors::BLUE);
  os << "\nSource:\n";
  os.resetColor();
  fromInfo.getOperation()->getLoc().print(os);
  os << "\n\t";
  fromInfo.getOperation()->print(os);
  os.changeColor(llvm::raw_ostream::Colors::BLUE);
  os << "\n\nDestination:\n";
  os.resetColor();
  toInfo.getOperation()->getLoc().print(os);
  os << "\n\t";
  toInfo.getOperation()->print(os);
  os.flush();
  auto d = solve();
  if (!d.hasValue()) {
    os.changeColor(llvm::raw_ostream::Colors::RED);
    os << "\n---------------------ERROR: No solution.---------------------\n\n";
    os.resetColor();
  } else {
    os << "\n-----------------------ILP Solved----------------------------\n\n";
  }
  os.close();
  return d;
}
std::tuple<int, int, int> getConstantLoopBounds(Value iv) {
  auto op = dyn_cast<mlir::AffineForOp>(iv.getParentRegion()->getParentOp());
  assert(op.getInductionVar() == iv);
  auto lb = op.getLowerBound().getMap().getSingleConstantResult();
  auto ub = op.getUpperBound().getMap().getSingleConstantResult();
  auto step = op.getStep();
  return std::make_tuple(lb, ub, step);
}

int64_t MemoryDependenceILPHandler::insertRowCoefficients(
    SmallVectorImpl<int> &rowCoeffVec, ArrayRef<int64_t> coeffs,
    OperandRange memIndices, ArrayRef<Value> loopIVs, bool isNegativeCoeff) {
  assert(memIndices.size() ==
         coeffs.size() -
             1); // coeffs contains the constant as well as the last element.
  int64_t accumulatedLowerBounds = 0;
  for (Value const iv : loopIVs) {
    auto bounds = getConstantLoopBounds(iv);
    int coeff = 0;
    auto loc = findLocInArray(iv, memIndices);
    if (loc.hasValue()) {
      // Multiply the loop step since the ilp variable corresponding to iv
      // represents the loop iteration number, i.e. it can be thought of a
      // canonical iv which lb=0 and step=1.
      // The actual iv = step*canonical_iv + lb.
      coeff = std::get<2>(bounds) * coeffs[loc.getValue()];
      accumulatedLowerBounds += std::get<0>(bounds);
    }
    rowCoeffVec.push_back(isNegativeCoeff ? -coeff : coeff);
  }
  return accumulatedLowerBounds;
}

void MemoryDependenceILPHandler::addILPColumns() {

  // Define the ILP column equations corresponding to each parent loop IV of
  // the `to` operation.
  for (size_t i = 0; i < toInfo.getParentLoops().size(); i++) {
    auto affineForOp = toInfo.getParentLoops()[i];
    addColumnVar(GLP_DB, affineForOp.getConstantLowerBound(),
                 affineForOp.getConstantUpperBound() - 1,
                 getLoopII(affineForOp));
  }

  // Define the ILP column equations corresponding to each parent loop IV of
  // the `from` operation.
  for (size_t j = 0; j < fromInfo.getParentLoops().size(); j++) {
    auto affineForOp = fromInfo.getParentLoops()[j];
    addColumnVar(GLP_DB, affineForOp.getConstantLowerBound(),
                 affineForOp.getConstantUpperBound() - 1,
                 -getLoopII(affineForOp));
  }
}

void MemoryDependenceILPHandler::addMemoryConstraintILPRows() {
  auto *from = fromInfo.getOperation();
  auto *to = toInfo.getOperation();
  auto fromMemAccessMap = getAffineMapForMemoryAccess(from);
  auto toMemAccessMap = getAffineMapForMemoryAccess(to);

  auto fromAffineExprs = fromMemAccessMap.getResults();
  auto toAffineExprs = toMemAccessMap.getResults();
  size_t const numRows = fromAffineExprs.size();

  assert(toAffineExprs.size() == numRows);

  // FIXME: Currently we handle port conflicts
  // by always assuming data dependence, i.e. treat the memory like a single
  // register. This means we put sequential ordering on this operations, though,
  // the actual requirement is just that they should not happen in same cycle.
  auto fromPortNum =
      getMemrefPortFromAffineLoadOrStoreOpAttr(fromInfo.getOperation());
  auto toPortNum =
      getMemrefPortFromAffineLoadOrStoreOpAttr(toInfo.getOperation());
  if (!fromPortNum.has_value() || !toPortNum.has_value() ||
      *fromPortNum == *toPortNum)
    return;

  // Add constraint for address equivalence.
  for (size_t row = 0; row < numRows; row++) {
    SmallVector<int, 4> rowCoeffVec;
    int64_t constCoeff = 0;
    SmallVector<int64_t> toCoeffs;
    assert(getFlattenedAffineExpr(toMemAccessMap.getResult(row),
                                  toMemAccessMap.getNumDims(), 0, &toCoeffs)
               .succeeded());
    constCoeff -=
        insertRowCoefficients(rowCoeffVec, toCoeffs, getMemIndices(to),
                              toInfo.getParentLoopIVs(), false);

    SmallVector<int64_t> fromCoeffs;
    assert(getFlattenedAffineExpr(fromMemAccessMap.getResult(row),
                                  fromMemAccessMap.getNumDims(), 0, &fromCoeffs)
               .succeeded());
    constCoeff +=
        insertRowCoefficients(rowCoeffVec, fromCoeffs, getMemIndices(from),
                              fromInfo.getParentLoopIVs(), true);
    constCoeff +=
        fromCoeffs[fromCoeffs.size() - 1] - toCoeffs[toCoeffs.size() - 1];
    addRow(rowCoeffVec, GLP_FX, constCoeff, constCoeff);
  }
}

// This function creates the ILP constraint that ensures destionation op happens
// after the source op.
// If the staticPos of the dest > src, then the common loop ivs of dest must be
// >= the src else it must be >.
void MemoryDependenceILPHandler::addHappensBeforeConstraintRow() {
  if (toInfo.getOperation() == fromInfo.getOperation())
    assert(toInfo.getStaticPosition() == fromInfo.getStaticPosition());

  int const staticDist =
      toInfo.getStaticPosition() - fromInfo.getStaticPosition();
  SmallVector<int> commonIVCoeffs;
  auto destIVs = toInfo.getParentLoopIVs();
  auto srcIVs = fromInfo.getParentLoopIVs();

  // Find the locations of the innermost common loop in the src and dest loop
  // nests.
  int destPos = destIVs.size() - 1;
  int srcPos = srcIVs.size() - 1;
  while (destPos >= 0 && srcPos >= 0) {
    if (destIVs[destPos] != srcIVs[srcPos])
      break;
    srcPos--;
    destPos--;
  }
  srcPos++;
  destPos++;

  int64_t coeff = 1;
  for (size_t i = srcPos; i < srcIVs.size(); i++) {
    AffineForOp forOp = fromInfo.getParentLoops()[i];
    assert(coeff < 1000'000); // Some bound to ensure no overflow occurs.

    commonIVCoeffs.push_back(coeff);
    coeff *= (forOp.getUpperBound()
                  .getMap()
                  .getResult(0)
                  .dyn_cast<AffineConstantExpr>()
                  .getValue() +
              1);
  }

  SmallVector<int> rowCoeffs;
  rowCoeffs.append(destIVs.size() + srcIVs.size(), 0);
  for (size_t i = 0; i < commonIVCoeffs.size(); i++) {
    rowCoeffs[destPos + i] = commonIVCoeffs[i];
    rowCoeffs[destIVs.size() + srcPos + i] = -commonIVCoeffs[i];
  }
  // FIXME: if staticDist == 0 then the delay should be equal to store op delay
  // not just one.
  auto lb = staticDist > 0 ? 0 : 1;

  addRow(rowCoeffs, GLP_LO, lb, lb);
}

//-----------------------------------------------------------------------------
// SchedulingILPHandler methods.
//-----------------------------------------------------------------------------

SchedulingILPHandler::SchedulingILPHandler(
    const SmallVector<Operation *> operations,
    const llvm::DenseMap<std::pair<Operation *, Operation *>,
                         std::pair<int64_t, int64_t>>
        &mapMemoryDependenceToSlackAndDelay,
    const SmallVector<SSADependence> &ssaDependences,
    llvm::DenseMap<Value, ArrayAttr> &mapMemrefToPortsAttr,
    const llvm::SmallVector<SchedulingConstraint> &schedulingConstraints,
    const std::string &logFile)
    : ILPHandler("SchedulingILP", GLP_MIN, logFile), operations(operations),
      mapMemoryDependenceToSlackAndDelay(mapMemoryDependenceToSlackAndDelay),
      ssaDependences(ssaDependences),
      mapMemrefToPortsAttr(mapMemrefToPortsAttr),
      schedulingConstraints(schedulingConstraints) {}

void SchedulingILPHandler::addILPColumns(llvm::raw_fd_ostream &os) {
  size_t col = 1;
  for (auto *operation : operations) {
    mapOperationToCol[operation] = col;
    addColumnVar(GLP_LO, 0, 0);
    // Print the mapping for debugging.
    os << "-----------------------------------------------------------------"
          "\nOperation mapped to c"
       << mapOperationToCol[operation] << " : ";
    operation->getLoc().print(os);
    os << "\n";
    // operation->print(os);
    // os << "\n";
    col += 1;
  }

  // This column corresponds to the max of all time-vars i.e. the total latency.\
  // It contributes to the objective (to minimize the overall latency).
  maxTimeCol = col;
  addColumnVar(GLP_LO, 0, 0, /*objective coeff*/ 1);
}
void setRowCoeffsOfOpAndItsParents(
    SmallVector<int> &rowCoeffs,
    DenseMap<Operation *, size_t> &mapOperationToCol, Operation *operation,
    int coeffValue) {
  while (!isa<mlir::func::FuncOp>(operation)) {
    int const loc = mapOperationToCol[operation] - 1;
    rowCoeffs[loc] += coeffValue;
    operation = operation->getParentOp();
  }
}
void SchedulingILPHandler::addMemoryDependenceRows() {
  for (auto depSlackAndDelay : mapMemoryDependenceToSlackAndDelay) {
    auto *srcOp = depSlackAndDelay.getFirst().first;
    auto *destOp = depSlackAndDelay.getFirst().second;
    auto slack = depSlackAndDelay.getSecond().first;
    auto delay = depSlackAndDelay.getSecond().second;
    SmallVector<int> rowCoeffs(this->getNumCols(), 0);
    setRowCoeffsOfOpAndItsParents(rowCoeffs, mapOperationToCol, srcOp, 1);
    setRowCoeffsOfOpAndItsParents(rowCoeffs, mapOperationToCol, destOp, -1);
    addRow(rowCoeffs, GLP_UP, slack - delay, slack - delay);
  }
}

void SchedulingILPHandler::addSSADependenceRows() {
  for (auto dep : ssaDependences) {
    SmallVector<int> rowCoeffs(getNumCols(), 0);
    Operation *destOp = dep.getDestOp();
    if (!dep.srcIsRegionArg()) {
      Operation *srcOp = dep.getSrcOp();
      assert(srcOp->getParentRegion() == destOp->getParentRegion());
      int const srcOpCol = mapOperationToCol[srcOp] - 1;
      rowCoeffs[srcOpCol] -= 1;
      incrObjectiveCoeff(srcOpCol, -1);
    }

    int64_t const delay = dep.getMinimumDelay();
    int const destOpCol = mapOperationToCol[destOp] - 1;
    rowCoeffs[destOpCol] = 1;
    // FIXME: Instead of '1', we should use width of the ssa var for the cost of
    // the delayOp.
    // FIXME: This cost is not accurate cost. For same value, delays will be
    // shared.
    // Add cost of a delayOp into the objective.
    incrObjectiveCoeff(destOpCol, 1);

    addRow(rowCoeffs, GLP_LO, delay, delay);
  }
}

/// Ensure that the maxTime variable is the upper bound of all variables.
void SchedulingILPHandler::addMaxTimeOffsetRows() {
  for (auto it : mapOperationToCol) {
    SmallVector<int> rowCoeffs(getNumCols(), 0);
    rowCoeffs[maxTimeCol - 1] = 1;
    rowCoeffs[it.getSecond()] -= 1;
    addRow(rowCoeffs, GLP_LO, 0, 0);
  }
}

llvm ::Optional<DenseMap<Operation *, int64_t>>
SchedulingILPHandler::getSchedule() {
  std::error_code ec;
  llvm::raw_fd_ostream os(logFile.c_str(), ec,
                          llvm::sys::fs::OpenFlags::OF_Append);
  addILPColumns(os);
  addMemoryDependenceRows();
  addSSADependenceRows();
  addMaxTimeOffsetRows();
  os.changeColor(llvm::raw_ostream::Colors::GREEN);
  os << "\n\n------------------------Top level "
        "ILP----------------------------\n";
  os.resetColor();
  os.flush();
  if (!solve().hasValue()) {
    os.changeColor(llvm::raw_ostream::Colors::RED);
    os << "\n---------------------ERROR: No solution.---------------------\n\n";
    os.resetColor();
    return llvm::None;
  }
  os << "\n-----------------------ILP Solved----------------------------\n\n";
  os.close();
  DenseMap<Operation *, int64_t> sched;
  for (auto *operation : operations) {
    sched[operation] = getColVarValue(mapOperationToCol[operation]);
  }
  return sched;
}

llvm::DenseMap<Operation *, std::pair<int64_t, int64_t>>
SchedulingILPHandler::getPortAssignments() {
  llvm::DenseMap<Operation *, std::pair<int64_t, int64_t>> portAssignments;
  for (auto *operation : this->operations) {
    Value mem;
    if (auto affineLoadOp = dyn_cast<AffineLoadOp>(operation)) {
      mem = affineLoadOp.getMemRef();
    } else if (auto affineStoreOp = dyn_cast<AffineStoreOp>(operation)) {
      mem = affineStoreOp.getMemRef();
    } else {
      continue;
    }
    ArrayAttr const ports = mapMemrefToPortsAttr[mem];
    if (!ports) {
      emitError(mem.getLoc()) << "Could not find the ports for this memref.";
      assert(ports);
    }
    if (auto pNum = operation->getAttrOfType<IntegerAttr>("hir.memref_port")) {
      int latency = 0;
      if (isa<AffineLoadOp>(operation)) {
        latency =
            helper::getMemrefPortRdLatency(ports[pNum.getInt()]).hasValue();
      } else {
        latency =
            helper::getMemrefPortWrLatency(ports[pNum.getInt()]).getValue();
      }
      portAssignments[operation] = std::make_pair(pNum.getInt(), latency);
      continue;
    }

    for (size_t i = 0; i < ports.size(); i++) {
      auto port = ports[i];
      auto portDict = port.dyn_cast<DictionaryAttr>();
      if (isa<AffineLoadOp>(operation) && helper::isMemrefRdPort(portDict)) {
        auto latency = helper::getMemrefPortRdLatency(portDict);
        assert(latency.hasValue());
        portAssignments[operation] = std::make_pair(i, latency.getValue());
      } else if (isa<AffineStoreOp>(operation) &&
                 helper::isMemrefWrPort(portDict)) {
        auto latency = helper::getMemrefPortWrLatency(portDict);
        assert(latency.hasValue());
        portAssignments[operation] = std::make_pair(i, latency.getValue());
      }
    }
  }
  return portAssignments;
}
