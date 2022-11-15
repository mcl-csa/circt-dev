#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/Analysis/TimingInfo.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;

class OptBitWidthPass : public OptBitWidthBase<OptBitWidthPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hir::ForOp);
  LogicalResult visitOp(hir::DelayOp);
};

unsigned int getMinBitWidth(Value v, const DenseSet<Operation *> ignoredUsers) {
  assert(v.getType().isa<IntegerType>());
  int64_t originalBitwidth = v.getType().getIntOrFloatBitWidth();
  int64_t bitwidth = 1;
  for (Operation *user : v.getUsers()) {
    if (ignoredUsers.contains(user))
      continue;
    if (auto extractOp = dyn_cast<comb::ExtractOp>(user)) {
      if (extractOp.getLowBit() == 0) {
        bitwidth = std::max(
            bitwidth,
            (int64_t)extractOp.getResult().getType().getIntOrFloatBitWidth());
        continue;
      }
    }
    return originalBitwidth;
  }
  return bitwidth;
}

LogicalResult OptBitWidthPass::visitOp(DelayOp op) {
  if (!op.getResult().getType().isa<IntegerType>())
    return success();
  auto result = op.getResult();

  // Separate the use in NextIterOp into a new DelayOp so that this delayOp can
  // be optimized.
  for (auto &use : result.getUses()) {
    if (isa<hir::NextIterOp>(use.getOwner())) {
      OpBuilder builder(op);
      Value duplicatedResult = builder.clone(*op)->getResult(0);
      use.set(duplicatedResult);
    }
  }
  if (result.getUses().empty()) {
    return success();
  }

  // If all uses of the result are comb::ExtractOp then move the ExtractOp
  // before delayOp.
  auto bitwidth = getMinBitWidth(op.getResult(), DenseSet<Operation *>());
  if (bitwidth < op.getResult().getType().getIntOrFloatBitWidth()) {
    OpBuilder builder(op);
    Value newValue = builder.create<circt::comb::ExtractOp>(
        builder.getUnknownLoc(), builder.getIntegerType(bitwidth), op.input(),
        builder.getI32IntegerAttr(0));
    op.inputMutable().assign(newValue);
    op.getResult().setType(newValue.getType());
  }
  return success();
}

/// Reduce the bitwidth of inductionVar and iterArgs if all their uses require
/// fewer bits.
LogicalResult OptBitWidthPass::visitOp(hir::ForOp op) {
  OpBuilder builder(op);

  // Reduce bitwidth of iterArgs.
  auto iterArgs = op.getIterArgs();
  for (size_t i = 0; i < iterArgs.size(); i++) {
    auto arg = iterArgs[i];
    if (!arg.getType().isa<IntegerType>())
      continue;

    // Ignore the DelayOp users which just feedback the same value i.e. if we
    // have a cycle iterArg -> delay -> NextIterOp (back to iterArg) then
    // ignore such use.
    llvm::DenseSet<Operation *> ignoredUsers;
    for (auto *user : arg.getUsers()) {
      bool isIgnoredUser = false;
      if (isa<DelayOp>(user)) {
        auto delayedArgUses = user->getResult(0).getUses();
        if (delayedArgUses.empty())
          isIgnoredUser = true;
        for (auto &use : delayedArgUses) {
          if (isa<NextIterOp>(use.getOwner()) && use.getOperandNumber() == i) {
            isIgnoredUser = true;
          } else {
            isIgnoredUser = false;
            break;
          }
        }
      }
      if (isIgnoredUser)
        ignoredUsers.insert(user);
    }

    auto bitwidth = getMinBitWidth(arg, ignoredUsers);
    if (bitwidth < arg.getType().getIntOrFloatBitWidth()) {
      Value newValue = builder.create<circt::comb::ExtractOp>(
          builder.getUnknownLoc(), builder.getIntegerType(bitwidth),
          op.getIterArgOperand(i), builder.getI32IntegerAttr(0));
      op.setIterArgOperand(i, newValue);
      for (auto *user : ignoredUsers) {
        auto delayOp = dyn_cast<DelayOp>(user);
        delayOp.getResult().setType(builder.getIntegerType(bitwidth));
      }
    }
  }

  // If its an unroll loop then return.
  if (op.getInductionVar().getType().isa<IndexType>())
    return success();

  // Otherwise reduce bitwidth of Induction Var.
  auto lb = helper::getConstantIntValue(op.lb());
  auto ub = helper::getConstantIntValue(op.ub());
  auto step = helper::getConstantIntValue(op.step());

  auto bitwidth = getMinBitWidth(op.getInductionVar(), DenseSet<Operation *>());
  bitwidth = std::max(bitwidth, helper::clog2(*ub + 1));
  bitwidth = std::max(bitwidth, helper::clog2(*step + 1));

  if (lb.hasValue() && ub.hasValue() && step.hasValue() &&
      (bitwidth < op.getInductionVar().getType().getIntOrFloatBitWidth())) {
    builder.setInsertionPoint(op);
    auto ty = builder.getIntegerType(bitwidth);

    op.setInductionVar(ty);
    op.lbMutable().assign(builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(ty, *lb)));
    op.ubMutable().assign(builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(ty, *ub)));
    op.stepMutable().assign(builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), IntegerAttr::get(ty, *step)));
  }

  return success();
}

void OptBitWidthPass::runOnOperation() {
  auto funcOp = getOperation();
  // We need post-order walk to ensure that iter-args of inner loops are
  // optimized before visiting outer loops because outer loop induction-var is
  // often captured in the inner loop.
  funcOp.walk<mlir::WalkOrder::PostOrder>([this](Operation *operation) {
    if (auto op = dyn_cast<ForOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    if (auto op = dyn_cast<DelayOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createOptBitWidthPass() {
  return std::make_unique<OptBitWidthPass>();
}
} // namespace hir
} // namespace circt
