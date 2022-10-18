#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>
using namespace circt;
using namespace hir;

Value getConstantX(OpBuilder &builder, int width) {
  return builder.create<sv::ConstantXOp>(builder.getUnknownLoc(),
                                         builder.getIntegerType(width));
}

class FusionAnalysis {
public:
  FusionAnalysis(hw::HWModuleOp);
  llvm::DenseMap<int, llvm::SmallVector<hw::InstanceOp, 2>> &getFusionGroups();
  int64_t getSelectArgNum(int group);

private:
  LogicalResult visitOp(hw::InstanceOp);

private:
  llvm::DenseMap<int, llvm::SmallVector<hw::InstanceOp, 2>> fusionGroups;
  llvm::DenseMap<int, int> fusionGroupSelectArgNum;
};

class FuseHWInstPass : public hir::FuseHWInstBase<FuseHWInstPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hw::InstanceOp);
};

// FusionAnalysis class methods.
FusionAnalysis::FusionAnalysis(hw::HWModuleOp op) {
  op.walk([this](Operation *operation) {
    if (auto instanceOp = dyn_cast<hw::InstanceOp>(operation)) {
      if (failed(visitOp(instanceOp)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

llvm::DenseMap<int, SmallVector<hw::InstanceOp, 2>> &
FusionAnalysis::getFusionGroups() {
  return this->fusionGroups;
}

int64_t FusionAnalysis::getSelectArgNum(int group) {
  assert(fusionGroupSelectArgNum.find(group) != fusionGroupSelectArgNum.end());
  return fusionGroupSelectArgNum[group];
}

LogicalResult FusionAnalysis::visitOp(hw::InstanceOp op) {
  auto hirAttrs = op->getAttrOfType<DictionaryAttr>("hir_attrs");
  auto fuseAttr =
      hirAttrs.getNamed("fuse")->getValue().dyn_cast<DictionaryAttr>();
  auto group =
      fuseAttr.getNamed("group")->getValue().dyn_cast<IntegerAttr>().getInt();
  this->fusionGroups[group].push_back(op);
  this->fusionGroupSelectArgNum[group] =
      fuseAttr.getNamed("select")->getValue().dyn_cast<IntegerAttr>().getInt();
  return success();
}

// FuseHWInstPass class methods.
void FuseHWInstPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();
  auto fusionAnalysis = FusionAnalysis(moduleOp);
  auto fusionGroups = fusionAnalysis.getFusionGroups();
  for (auto group : fusionGroups) {
    hw::InstanceOp firstInstanceOp = group.getSecond()[0];
    OpBuilder builder(firstInstanceOp);
    auto selectArgNum = fusionAnalysis.getSelectArgNum(group.getFirst());
    auto t = getConstantX(builder, 1);
    SmallVector<Value> inputs;
    for (auto inp : firstInstanceOp.getInputs()) {
      auto width = inp.getType().getIntOrFloatBitWidth();
      inputs.push_back(getConstantX(builder, width));
    }
    for (auto instanceOp : group.getSecond()) {
      for (size_t i = 0; i < instanceOp.getInputs().size(); i++) {
        inputs[i] = builder.create<comb::MuxOp>(
            builder.getUnknownLoc(), instanceOp.getInputs()[selectArgNum],
            instanceOp.getInputs()[i], inputs[i]);
      }
    }
    // builder.create<hw::InstanceOp>(
    //     builder.getUnknownLoc(), firstInstanceOp.getModuleName(),
    //     firstInstanceOp.getInstanceName(), inputs,
    //     firstInstanceOp.getParameters(), StringAttr());
  }
}

LogicalResult FuseHWInstPass::visitOp(hw::InstanceOp) { return success(); }

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hw::HWModuleOp>> createFuseHWInstPass() {
  return std::make_unique<FuseHWInstPass>();
}
} // namespace hir
} // namespace circt
