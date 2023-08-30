//=========- OpFusionPass.cpp - Fuse ops---===//
//
// This file implements the HIR op fusion.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <numeric>
#include <string>
using namespace circt;
namespace {
struct AccessInfo {
  Value timeVar;
  int64_t startTime;
  int64_t endTime;
  int64_t minII;
  DictionaryAttr intoAttr(Builder &builder) {
    auto startTimeAttr = builder.getNamedAttr(
        "startTimeOffset", builder.getI64IntegerAttr(startTime));
    auto endTimeAttr = builder.getNamedAttr("endTimeOffset",
                                            builder.getI64IntegerAttr(endTime));
    auto minIIAttr =
        builder.getNamedAttr("minII", builder.getI64IntegerAttr(minII));

    return builder.getDictionaryAttr({startTimeAttr, endTimeAttr, minIIAttr});
  }
};

class OpFusionPass : public hir::OpFusionBase<OpFusionPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hir::CallOp);

private:
  llvm::DenseMap<hir::CallOp, AccessInfo> mapCallToTimeRange;
};
} // end anonymous namespace

Optional<AccessInfo> addAccessInfoAttr(hir::CallOp op) {
  if (op->getParentOfType<hir::WhileOp>())
    return llvm::None;
  if (!op->hasAttrOfType<IntegerAttr>("II"))
    return llvm::None;

  auto *operation = op.getOperation();
  AccessInfo info;
  info.minII = 0;
  info.timeVar = op.tstart();
  if (info.timeVar !=
      dyn_cast<hir::RegionOp>(operation->getParentOp()).getRegionTimeVars()[0])
    return llvm::None;
  info.startTime = op.offset();
  info.endTime = op.offset() + op->getAttrOfType<IntegerAttr>("II").getInt();
  while (auto parentForOp = operation->getParentOfType<hir::ForOp>()) {
    if (!parentForOp.getInitiationInterval().has_value())
      return llvm::None;

    if (!parentForOp.getTripCount().has_value())
      return llvm::None;

    info.minII =
        std::gcd(info.minII, parentForOp.getInitiationInterval().getValue());
    info.timeVar = parentForOp.tstart();
    info.startTime += parentForOp.offset();
    info.endTime += parentForOp.getInitiationInterval().getValue() *
                    parentForOp.getTripCount().getValue();
    operation = parentForOp;
  }
  Builder builder(op);
  op->setAttr("access_info", info.intoAttr(builder));
  return info;
}

LogicalResult OpFusionPass::visitOp(hir::CallOp op) { return success(); }

void OpFusionPass::runOnOperation() {
  hir::FuncOp funcOp = getOperation();
  WalkResult const result =
      funcOp.walk([this](Operation *operation) -> WalkResult {
        if (auto callOp = dyn_cast<hir::CallOp>(operation)) {
          if (failed(visitOp(callOp)))
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
std::unique_ptr<OperationPass<hir::FuncOp>> createOpFusionPass() {
  return std::make_unique<OpFusionPass>();
}
} // namespace hir
} // namespace circt
