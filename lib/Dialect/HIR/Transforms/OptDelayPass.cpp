#include "PassDetails.h"
#include "circt/Dialect/HIR/Analysis/TimingInfo.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"

using namespace circt;
using namespace hir;

class OptDelayPass : public OptDelayBase<OptDelayPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hir::DelayOp op);

private:
};

/// Replace the DelayOp with a set of registers.
LogicalResult OptDelayPass::visitOp(hir::DelayOp op) {
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  auto parentOp = dyn_cast<hir::RegionOp>(op->getParentOp());
  auto ii = parentOp.getRegionII();

  // Nothing to do if we don't know the initiation interval.
  if (!ii)
    return success();
  auto numReg = std::ceil(op.getDelay() / ii.value());
  // FIXME: Do this opt for the general case.
  if (numReg > 1)
    return success();
  auto hirTy =
      hir::MemrefType::get(builder.getContext(), {1}, op.getResult().getType(),
                           {hir::DimKind::BANK});
  auto mem =
      builder
          .create<hir::AllocaOp>(
              op->getLoc(), hirTy,
              MemKindEnumAttr::get(builder.getContext(), MemKindEnum::reg),
              helper::getPortAttrForReg(builder))
          .getResult();
  auto zeroIdx = helper::emitConstantOp(builder, 0).getResult();
  builder.create<hir::StoreOp>(
      op->getLoc(), op.getInput(), mem, zeroIdx, builder.getI64IntegerAttr(1),
      builder.getI64IntegerAttr(1), op.getTstart(), op.getOffsetAttr());
  auto loadOp = builder.create<hir::LoadOp>(
      op->getLoc(), op.getResult().getType(), mem, zeroIdx,
      builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(0),
      op.getTstart(),
      builder.getI64IntegerAttr(op.getOffset() + op.getDelay()));
  op->replaceAllUsesWith(loadOp);
  return success();
}

void OptDelayPass::runOnOperation() {
  getOperation().walk([this](Operation *operation) {
    if (auto delayOp = dyn_cast<hir::DelayOp>(operation)) {
      if (failed(visitOp(delayOp)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createOptDelayPass() {
  return std::make_unique<OptDelayPass>();
}
} // namespace hir
} // namespace circt
