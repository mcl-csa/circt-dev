#include "circt/Conversion/SchedulingAnalysis.h"
#include "PragmaHandler.h"
#include "circt/Conversion/SchedulingUtils.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <memory>
using namespace mlir;
using RAMKind = MemrefPragmaHandler::RAMKind;
//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------
static llvm::DenseSet<size_t> getAddrDims(Value memref) {
  MemrefPragmaHandler pragmaHandler(memref);
  llvm::DenseSet<size_t> out;
  for (size_t i = 0; i < pragmaHandler.getNumDims(); i++) {
    if (pragmaHandler.getDimKind(i) == MemrefPragmaHandler::DimKind::ADDR)
      out.insert(i);
  }

  return out;
}
/// Finds the containing user (such as a ForOp) that is immediately in the
/// lexical scope.
static Operation *getOpContainingUserInScope(Operation *user, Region *scope) {
  while (user->getParentRegion() != scope)
    user = user->getParentOp();
  return user;
}

static Optional<size_t> getArgDelay(Operation *operation, size_t i) {
  if (auto op = dyn_cast<func::CallOp>(operation)) {
    return FuncExternPragmaHandler(op).getArgDelay(i);
  }
  return 0;
}

static Optional<size_t> getResultDelay(Operation *operation, size_t i) {
  if (isa<arith::ArithmeticDialect>(operation->getDialect()))
    return ArithOpInfo(operation).getDelay();
  if (isa<AffineLoadOp, AffineStoreOp>(operation))
    return MemOpInfo(operation, 0 /*dont care*/).getDelay();
  if (auto op = dyn_cast<func::CallOp>(operation)) {
    return FuncExternPragmaHandler(op).getResultDelay(i);
  }
  return llvm::None;
}

static LogicalResult getCommonII(Operation *op, Optional<int64_t> &commonII) {
  auto *parent = op->getParentOp();
  while (!isa<func::FuncOp>(parent)) {
    auto iiAttr = parent->getAttrOfType<IntegerAttr>("II");
    if (!iiAttr)
      return parent->emitError("Could not find II IntegerAttr.");

    if (!commonII)
      commonII = iiAttr.getInt();
    else
      commonII = std::gcd(*commonII, iiAttr.getInt());
    parent = parent->getParentOp();
  }
  return success();
}

static LogicalResult getCommonII(Operation *op1, Operation *op2,
                                 Optional<int64_t> commonII) {
  if (failed(getCommonII(op1, commonII)))
    return failure();

  if (failed(getCommonII(op2, commonII)))
    return failure();
  return success();
}

//-----------------------------------------------------------------------------
// class SchedulingAnalysis methods.
//-----------------------------------------------------------------------------
HIRScheduler::HIRScheduler(mlir::func::FuncOp op, llvm::raw_ostream &logger)
    : Scheduler(logger), funcOp(op) {}

int64_t HIRScheduler::getPortNumForMemoryOp(Operation *operation) {
  // FIXME: Handle multiple read/write ports.
  if (auto loadOp = dyn_cast<AffineLoadOp>(operation)) {
    auto pragma = MemrefPragmaHandler(loadOp.getMemref());
    assert(pragma.getNumRdPorts() > 0);
    auto *resource = mapMemref2RdPortResource[loadOp.getMemref()];
    // If resource was not allocated earlier then there was no potential port
    // conflict.
    if (!resource)
      return pragma.getRdPortID(0);
    return pragma.getRdPortID(
        this->getResourceAllocation(operation, resource).value_or(0));
  }

  auto storeOp = dyn_cast<AffineStoreOp>(operation);
  auto pragma = MemrefPragmaHandler(storeOp.getMemref());
  assert(pragma.getNumWrPorts() > 0);
  auto *resource = mapMemref2WrPortResource[storeOp.getMemref()];
  if (!resource)
    return pragma.getWrPortID(0);

  return pragma.getWrPortID(
      this->getResourceAllocation(operation, resource).value_or(0));
}

LogicalResult HIRScheduler::insertMemoryDependence(MemOpInfo src,
                                                   MemOpInfo dest) {
  if (src.getMemRef() != dest.getMemRef())
    return success();

  if (src.isLoad() && dest.isLoad())
    return success();

  logger << "\n=========================================\n";
  logger << "MemoryDependenceILP:";
  logger << "\n=========================================\n\n";
  MemoryDependenceILP memoryDependenceILP(src, dest, {}, logger);

  // If the solution is not found then there is no dependence between the
  // operations.

  logger << "Source:";
  src.getOperation()->getLoc().print(logger);
  logger << "\nDest:";
  dest.getOperation()->getLoc().print(logger);
  logger << "\n\nILP:\n";
  logger << "----\n";
  memoryDependenceILP.solve();
  memoryDependenceILP.dump();

  if (!memoryDependenceILP.isSolved()) {
    logger << "\nNo dependence found.\n";
    logger << "\n=========================================\n\n";
    return success();
  }
  int64_t dist = (int64_t)memoryDependenceILP.Objective().Value();
  logger << "\nDependence found. dist= " << dist << "\n";
  logger << "\n=========================================\n\n";

  int64_t delay = src.isLoad() ? 0 : src.getDelay();

  assert(delay >= 0);

  Dependence dep("mem_dep", src.getOperation(), dest.getOperation(),
                 delay - dist);
  this->addDependence(dep);
  return success();
}

static MemPortResource *getOrAddPortResource(
    Value mem, llvm::DenseMap<mlir::Value, MemPortResource *> &map,
    llvm::SmallVector<MemPortResource> &storage, int64_t numPorts) {
  auto it = map.find(mem);
  if (it == map.end()) {
    storage.push_back(MemPortResource(mem, numPorts));
    map[mem] = &storage.back();
    return &storage.back();
  }
  return it->getSecond();
}

MemPortResource *HIRScheduler::getOrAddRdPortResource(Value mem) {
  assert(mem.getType().isa<MemRefType>());
  auto pragma = MemrefPragmaHandler(mem);
  if (pragma.getRAMKind() == RAMKind::TMP)
    getOrAddPortResource(mem, mapMemref2WrPortResource, portResources,
                         pragma.getNumWrPorts());
  return getOrAddPortResource(mem, mapMemref2RdPortResource, portResources,
                              pragma.getNumRdPorts());
}

MemPortResource *HIRScheduler::getOrAddWrPortResource(Value mem) {
  auto pragma = MemrefPragmaHandler(mem);
  if (pragma.getRAMKind() == RAMKind::TMP)
    getOrAddPortResource(mem, mapMemref2RdPortResource, portResources,
                         pragma.getNumRdPorts());
  return getOrAddPortResource(mem, mapMemref2WrPortResource, portResources,
                              pragma.getNumWrPorts());
}

/// Inserts constraints to ensure that there are no conflicts between memory
/// ports.
/// FIXME: If load-load or load-store operations have same address during
/// conflicting cycles then there is no conflict. We do not capture this yet.
/// Atleast cover the case when the adress is constant (useful for registers).
/// Alternatively, mem2reg pass can take care of the register usecase.
LogicalResult HIRScheduler::insertPortConflict(MemOpInfo src, MemOpInfo dest) {
  if (src.getMemRef() != dest.getMemRef())
    return success();
  auto pragma = MemrefPragmaHandler(src.getMemRef());
  if (src.isLoad() && !dest.isLoad() && pragma.getRAMKind() == RAMKind::SMP)
    return success();

  logger << "\n=========================================\n";
  logger << "BankDependenceILP:";
  logger << "\n=========================================\n\n";

  auto addrDims = getAddrDims(src.getMemRef());
  MemoryDependenceILP bankDependenceILP(src, dest, addrDims, logger);

  // If the solution is not found then there is no bank conflict between the
  // operations.

  logger << "Source:";
  src.getOperation()->getLoc().print(logger);
  logger << "\nDest:";
  dest.getOperation()->getLoc().print(logger);
  logger << "\n\nILP:\n";
  logger << "----\n";
  bankDependenceILP.solve();
  bankDependenceILP.dump();

  if (!bankDependenceILP.isSolved()) {
    logger << "\nNo dependence found.\n";
    logger << "\n=========================================\n\n";
    return success();
  }

  int64_t dist = (int64_t)bankDependenceILP.Objective().Value();
  logger << "\nDependence found. dist= " << dist << "\n";
  logger << "\n=========================================\n\n";
  auto minRequiredDelay = 1 - dist;

  // If the source and dest are the same, scheduler can't do anything to remove
  // a port conflict. The port conflict of an operation with itself is only
  // possible if
  //  - There are outer unroll loops.
  //  - And, the initiation interval of said loops is such that multiple
  //  read/write operations are scheduled in same cycle.
  //
  // In cast of a port conflict the scheduler can't do anything since,
  // - Both source and dest op has same time offset and hence same remainder (so
  //   can't do modulo scheduling).
  // - Both source and dest have the same port (since source==dest), so can't
  //   use different ports to avoid the port conflict.
  //
  // If the dist <=0 then there may be a port conflict.
  // In such case we error out.
  // FIXME: We could solve another ILP to check if the self conflict actually
  // exist. In some cases the dist may be less than zero but no two operations
  // are scheduled in same cycle.
  if (src.getOperation() == dest.getOperation()) {
    if (dist <= 0)
      return src.getOperation()->emitError(
          "Could not schedule because of the operation has a port conflict "
          "with itself. Use a different initiation interval.\n");
    return success();
  }

  Optional<int64_t> commonII;
  if (failed(getCommonII(src.getOperation(), dest.getOperation(), commonII)))
    return failure();

  Resource *resource;
  if (src.isLoad()) {
    resource = getOrAddRdPortResource(src.getMemRef());
  } else {
    resource = getOrAddWrPortResource(src.getMemRef());
  }

  // If commonII is not present (both operations are at function level), then
  // assume a large enough commonII.
  Conflict conflict(src.getOperation(), dest.getOperation(),
                    commonII.value_or(100), resource, minRequiredDelay);

  this->addConflict(conflict);
  return success();
}

LogicalResult HIRScheduler::insertMemAccessConstraints() {
  SmallVector<MemOpInfo> memOperations;
  funcOp.walk([&memOperations](Operation *operation) {
    if (isa<AffineLoadOp, AffineStoreOp>(operation))
      memOperations.push_back(MemOpInfo(operation, memOperations.size()));
    return WalkResult::advance();
  });

  for (size_t i = 0; i < memOperations.size(); i++) {
    auto srcOpInfo = memOperations[i];
    for (size_t j = 0; j < memOperations.size(); j++) {
      auto destOpInfo = memOperations[j];
      if (failed(this->insertMemoryDependence(srcOpInfo, destOpInfo)))
        return failure();
      if (failed(this->insertPortConflict(srcOpInfo, destOpInfo)))
        return failure();
    }
  }
  return success();
}

static Optional<Dependence> getDependence(Operation *def, OpOperand &use,
                                          size_t resultDelay) {
  Optional<size_t> argDelay = 0;

  if (def) {
    argDelay = getArgDelay(use.getOwner(), use.getOperandNumber());
    if (!argDelay) {
      def->emitError("Could not find delay for arg ")
          << use.getOperandNumber() << ".";
      return llvm::None;
    }
  } else
    // if `def` is not specified then the operand must be a region arg.
    assert(!use.get().getDefiningOp());

  auto *userOpAtThisScope =
      getOpContainingUserInScope(use.getOwner(), use.get().getParentRegion());

  Dependence dep("ssa_dep", def, userOpAtThisScope, resultDelay - *argDelay);
  return dep;
}

LogicalResult HIRScheduler::insertSSADependencies() {
  auto walkResult = funcOp.walk([this](Operation *operation) {
    if (operation->getNumResults() == 0 && operation->getNumRegions() == 0)
      return WalkResult::advance();

    if (isa<arith::ConstantOp, memref::AllocaOp>(operation))
      return WalkResult::advance();

    for (size_t i = 0; i < operation->getNumResults(); i++) {
      auto resultDelay = getResultDelay(operation, i);
      if (!resultDelay) {
        operation->emitError("Could not find delay for result ") << i << ".";
        return WalkResult::interrupt();
      }

      auto result = operation->getResult(i);
      for (auto &use : result.getUses()) {
        auto dep = getDependence(operation, use, *resultDelay);
        if (!dep)
          return WalkResult::interrupt();
        this->addDependence(*dep);
        this->addDelayRegisterCost(*dep,
                                   result.getType().getIntOrFloatBitWidth());
      }
    }
    auto regions = operation->getRegions();
    if (regions.empty())
      return WalkResult::advance();

    // FIXME: Currently support only one-region operations.
    assert(regions.size() == 1);
    for (size_t i = 0; i < regions[0].getNumArguments(); i++) {
      auto regionArg = regions[0].getArgument(i);
      for (auto &use : regionArg.getUses()) {
        auto dep = getDependence(NULL, use, 0);
        if (!dep)
          return WalkResult::interrupt();
        // For index types (loop indices in affine dialect) the bitwidth is
        // assumed to be 32;
        size_t bitwidth = 32;
        if (regionArg.getType().isIntOrFloat())
          bitwidth = regionArg.getType().getIntOrFloatBitWidth();

        this->addDelayRegisterCost(*dep, bitwidth);
      }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}

LogicalResult HIRScheduler::init() {
  if (failed(insertMemAccessConstraints()))
    return failure();
  if (failed(insertSSADependencies()))
    return failure();
  logger << "\n=========================================\n";
  logger << "Scheduling ILP:";
  logger << "\n=========================================\n\n";
  if (this->solve() != ResultStatus::OPTIMAL) {
    this->dump();
    return funcOp->emitError("Could not find schedule.");
  }
  this->dump();
  return success();
}