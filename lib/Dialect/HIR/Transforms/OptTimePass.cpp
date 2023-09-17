#include "PassDetails.h"
#include "circt/Dialect/HIR/Analysis/TimingInfo.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;

class OptTimePass : public OptTimeBase<OptTimePass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hir::ScheduledOp op);

private:
  std::optional<TimingInfo *> timingInfo;
};

LogicalResult OptTimePass::visitOp(hir::ScheduledOp op) {
  Time optTime = (*timingInfo)->getOptimizedTime(op);
  int64_t offset = optTime.getOffset();
  auto parentOp = dyn_cast<RegionOp>(op.getOperation()->getParentOp());
  auto ii = parentOp.getRegionII();
  if (offset < 16 || ii.value_or(-1) <= optTime.getOffset()) {
    op.setStartTime(optTime);
    return success();
  }

  OpBuilder builder(op);
  builder.setInsertionPoint(op);

  auto lb = builder.create<circt::hw::ConstantOp>(builder.getUnknownLoc(),
                                                  builder.getI64IntegerAttr(0));
  auto ub = builder.create<circt::hw::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64IntegerAttr(offset));
  auto step = builder.create<circt::hw::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64IntegerAttr(1));

  auto forOp = builder.create<hir::ForOp>(
      op.getLoc(), lb, ub, step, SmallVector<Value>({}), optTime.getTimeVar(),
      builder.getI64IntegerAttr(0),
      [](OpBuilder &builder, Value iv, ArrayRef<Value> iterArgs,
         Value tLoopBody) {
        auto nextIterOp = builder.create<hir::NextIterOp>(
            builder.getUnknownLoc(), Value(), SmallVector<Value>({}), tLoopBody,
            builder.getI64IntegerAttr(1));
        return nextIterOp;
      });
  op.setStartTime(Time(forOp.getResult(0), 0));
  return success();
}

void OptTimePass::runOnOperation() {
  auto funcOp = getOperation();
  auto tinfo = TimingInfo(funcOp);
  timingInfo = &tinfo;
  funcOp.walk([this](Operation *operation) {
    if (auto scheduledOp = dyn_cast<ScheduledOp>(operation)) {
      if (failed(visitOp(scheduledOp)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createOptTimePass() {
  return std::make_unique<OptTimePass>();
}
} // namespace hir
} // namespace circt
