//=========- LoopUnrollPass.cpp - Unroll loops---===//
//
// This file implements the HIR loop unrolling.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include <string>

using namespace circt;
namespace {

class LoopUnrollPass : public hir::LoopUnrollBase<LoopUnrollPass> {
public:
  void runOnOperation() override;
};
} // end anonymous namespace

LogicalResult unrollLoopFull(hir::ForOp forOp) {
  Block &loopBodyBlock = forOp.getLoopBody().front();
  // auto builder = OpBuilder::atBlockTerminator(&loopBodyBlock);
  auto builder = OpBuilder(forOp);
  builder.setInsertionPointAfter(forOp);

  if (failed(helper::isConstantIntValue(forOp.lb())))
    return forOp.emitError("Expected lower bound to be constant.");
  if (failed(helper::isConstantIntValue(forOp.ub())))
    return forOp.emitError("Expected upper bound to be constant.");
  if (failed(helper::isConstantIntValue(forOp.step())))
    return forOp.emitError("Expected step to be constant.");

  int64_t const lb = helper::getConstantIntValue(forOp.lb()).getValue();
  int64_t const ub = helper::getConstantIntValue(forOp.ub()).getValue();
  int64_t const step = helper::getConstantIntValue(forOp.step()).getValue();

  auto *context = builder.getContext();
  assert(forOp.offset() == 0);

  Value mappedIterTimeVar = forOp.tstart();
  SmallVector<Value> mappedIterArgs;
  for (auto iterArg : forOp.iter_args())
    mappedIterArgs.push_back(iterArg);

  // insert the unrolled body.
  for (int i = lb; i < ub; i += step) {
    auto loopIVOp = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), IndexType::get(context),
        builder.getIndexAttr(i));
    helper::setNames(loopIVOp, forOp.getInductionVarName());

    // Populate the operandMap.
    BlockAndValueMapping operandMap;
    for (size_t i = 0; i < forOp.iter_args().size(); i++) {
      Value const regionIterArg = loopBodyBlock.getArgument(i);
      operandMap.map(regionIterArg, mappedIterArgs[i]);
    }
    operandMap.map(forOp.getIterTimeVar(), mappedIterTimeVar);
    operandMap.map(forOp.getInductionVar(), loopIVOp.getResult());

    // Copy the loop body.
    for (auto &operation : loopBodyBlock) {
      if (auto nextIterOp = dyn_cast<hir::NextIterOp>(operation)) {
        assert(nextIterOp.offset() == 0);
        mappedIterArgs.clear();
        for (auto iterArg : nextIterOp.iter_args())
          mappedIterArgs.push_back(operandMap.lookup(iterArg));
        mappedIterTimeVar = operandMap.lookup(nextIterOp.tstart());
      } else if (auto probeOp = dyn_cast<hir::ProbeOp>(operation)) {
        auto unrolledName = builder.getStringAttr(probeOp.verilog_name() + "_" +
                                                  forOp.getInductionVarName() +
                                                  "_" + std::to_string(i));
        builder.create<hir::ProbeOp>(
            probeOp.getLoc(), operandMap.lookup(probeOp.input()), unrolledName);
      } else {
        auto *newOperation = builder.clone(operation, operandMap);
        // If its a CallOp then change the instance name. Otherwise it will get
        // fused during the op fusion pass. Ops across iterations of an unrolled
        // loop are not fused together.
        if (auto callOp = dyn_cast<hir::CallOp>(operation)) {
          auto instanceName =
              callOp.instance_name().str() + "_" + std::to_string(i);
          newOperation->setAttr("instance_name",
                                builder.getStringAttr(instanceName));
        }
      }
    }
  }

  assert(forOp.iter_args().size() == forOp.iterResults().size());
  // replace the ForOp results.
  forOp.t_end().replaceAllUsesWith(mappedIterTimeVar);
  for (size_t i = 0; i < forOp.iterResults().size(); i++)
    forOp.iterResults()[i].replaceAllUsesWith(mappedIterArgs[i]);

  forOp.erase();
  return success();
}

void LoopUnrollPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult const result = funcOp.walk([](Operation *operation) -> WalkResult {
    if (auto forOp = dyn_cast<hir::ForOp>(operation)) {
      if (forOp->getAttr("unroll") ||
          forOp.getInductionVar().getType().isa<mlir::IndexType>())
        if (failed(unrollLoopFull(forOp)))
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}
} // namespace hir
} // namespace circt
