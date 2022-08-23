#include "PassDetails.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <stack>
#include <utility>

using namespace circt;
using namespace hir;

// Helper functions.
static void populateTypeConversion(TypeConverter &t) {
  t.addConversion([](hir::FuncType type, SmallVectorImpl<Type> &types) {
    types.push_back(IntegerType::get(type.getContext(), 32));
    return success();
  });
}
namespace {

/// Lowers InstanceOp and CallInstanceOps.
class InstanceLoweringPass
    : public hir::InstanceLoweringBase<InstanceLoweringPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitOp(hir::FuncOp);
};

} // end anonymous namespace

// Convert FuncType parameters to a set of input and output buses recursively so
// that FuncType is not nested anymore.
FuncType flattenFuncType(FuncType ty) {

  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  std::stack<std::pair<FuncType, unsigned int>> stack;
  stack.push(std::make_pair(ty, 0));
  while (true) {
    FuncType funcTy = stack.top().first;
    unsigned int idx = stack.top().second;
    if (idx >= funcTy.getNumInputs()) {
      stack.pop();
      continue;
    }
    Type argType = funcTy.getInputType(idx);
    idx += 1;
    if (auto argFuncTy = argType.dyn_cast<hir::FuncType>()) {
      stack.push(std::make_pair(argFuncTy, 0));
    }
  }
  return FuncType::get(ty.getContext(), inputTypes, inputAttrs,
                       ty.getResultTypes(), ty.getResultAttrs());
}

LogicalResult InstanceLoweringPass::visitOp(FuncOp op) { return success(); }

void InstanceLoweringPass::runOnOperation() {}

namespace circt {
namespace hir {
std::unique_ptr<OperationPass<hir::FuncOp>> createInstanceLoweringPass() {
  return std::make_unique<InstanceLoweringPass>();
}
} // namespace hir
} // namespace circt
