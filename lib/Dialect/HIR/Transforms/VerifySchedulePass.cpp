//=========- VerifySchedulePass.cpp - Verify schedule of HIR dialect-------===//
//
// This file implements the HIR schedule verifier.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/Analysis/TimingInfo.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include <functional>
#include <list>
#include <stack>

using namespace circt;
using namespace hir;
using namespace llvm;
class VerifySchedulePass : public hir::VerifyScheduleBase<VerifySchedulePass> {
public:
  void runOnOperation() override;

private:
  LogicalResult verifyCombOp(Operation *);
  LogicalResult verifyOp(ScheduledOp);
  LogicalResult verifyOperation(Operation *);

private:
  TimingInfo *timingInfo;
};

void VerifySchedulePass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  auto schedInfo = TimingInfo(funcOp);
  this->timingInfo = &schedInfo;
  funcOp.walk([this](Operation *operation) {
    if (failed(verifyOperation(operation)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
}

LogicalResult VerifySchedulePass::verifyOperation(Operation *operation) {
  if (isa<comb::CombDialect>(operation->getDialect()))
    return verifyCombOp(operation);
  if (auto op = dyn_cast<hir::ScheduledOp>(operation))
    return verifyOp(op);
  return success();
}

LogicalResult VerifySchedulePass::verifyCombOp(Operation *operation) {
  Value nonConstantOperand;
  for (auto operand : operation->getOperands()) {
    if (timingInfo->isAlwaysValid(operand))
      continue;
    if (!nonConstantOperand) {
      nonConstantOperand = operand;
      continue;
    }
    auto time = timingInfo->getTime(nonConstantOperand);
    if (!timingInfo->isValidAtTime(operand, time)) {
      operation->emitError("Error in scheduling of operand.")
              .attachNote(operand.getLoc())
          << "Operand defined here.";
    }
  }
  return success();
}

LogicalResult VerifySchedulePass::verifyOp(ScheduledOp op) {
  for (auto operand : op->getOperands()) {
    if (helper::isBuiltinSizedType(operand.getType())) {
      if (!timingInfo->isValidAtTime(operand, op.getStartTime()))
        return op.emitError("Error in scheduling of operand.")
                   .attachNote(operand.getLoc())
               << "Operand defined here. ";
    }
  }
  return success();
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createVerifySchedulePass() {
  return std::make_unique<VerifySchedulePass>();
}
} // namespace hir
} // namespace circt
