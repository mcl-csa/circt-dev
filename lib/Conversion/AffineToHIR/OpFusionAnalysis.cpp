#include "circt/Conversion/OpFusionAnalysis.h"
using namespace mlir;
using namespace circt;
Optional<AccessInfo> getAccessInfoAttr(hir::CallOp op) {
  if (op->getParentOfType<hir::WhileOp>())
    return llvm::None;
  if (!op->hasAttrOfType<IntegerAttr>("hir.II"))
    return llvm::None;

  auto *operation = op.getOperation();
  AccessInfo info;
  info.minII = 0;
  info.timeVar = op.tstart();
  if (info.timeVar !=
      dyn_cast<hir::RegionOp>(operation->getParentOp()).getRegionTimeVars()[0])
    return llvm::None;
  info.startTime = op.offset();
  info.endTime =
      op.offset() + op->getAttrOfType<IntegerAttr>("hir.II").getInt();
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
  return info;
}

LogicalResult OpFusionAnalysis::visitOp(hir::CallOp op) {
  if (!op->hasAttrOfType<StringAttr>("hir.label"))
    return op->emitWarning("Could not find hir.label.");
  OpBuilder builder(op);
  auto info = getAccessInfoAttr(op);
  if (!info)
    return success();
  op->setAttr("access_info", info->intoAttr(builder));
  (*mapLabelToAccessInfo)[op->getAttrOfType<StringAttr>("hir.label").strref()] =
      *info;
  return success();
}

OpFusionAnalysis::OpFusionAnalysis(hir::FuncOp op) : HIRPassImplBase(op) {}

void OpFusionAnalysis::getAnalysis(
    llvm::DenseMap<StringRef, AccessInfo> &mapLabelToAccessInfo) {
  hir::FuncOp funcOp = getOperation();
  this->mapLabelToAccessInfo = &mapLabelToAccessInfo;
  funcOp.walk([this](Operation *operation) -> WalkResult {
    if (auto callOp = dyn_cast<hir::CallOp>(operation)) {
      if (failed(visitOp(callOp)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}
