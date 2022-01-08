#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace hir;
namespace {

/// Lowers InstanceOp and CallInstanceOps.
class InstanceLoweringPass
    : public hir::InstanceLoweringBase<InstanceLoweringPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hir::InstanceOp);
};
} // end anonymous namespace

LogicalResult InstanceLoweringPass::visitOp(InstanceOp op) {
  OpBuilder builder(op);
  return success();
}

void InstanceLoweringPass::runOnOperation() {

  getOperation().walk([](Operation *operation) {
    if (auto op = dyn_cast<InstanceOp>(operation)) {
    }
    WalkResult::advance();
  });
}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createInstanceLoweringPass() {
  return std::make_unique<InstanceLoweringPass>();
}
} // namespace hir
} // namespace circt
