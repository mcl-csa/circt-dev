#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/Transforms/HIRPassImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <numeric>

struct AccessInfo {
  mlir::Value timeVar;
  int64_t startTime;
  int64_t endTime;
  int64_t minII;
  mlir::DictionaryAttr intoAttr(mlir::Builder &builder) {
    auto startTimeAttr = builder.getNamedAttr(
        "startTimeOffset", builder.getI64IntegerAttr(startTime));
    auto endTimeAttr = builder.getNamedAttr("endTimeOffset",
                                            builder.getI64IntegerAttr(endTime));
    auto minIIAttr =
        builder.getNamedAttr("minII", builder.getI64IntegerAttr(minII));

    return builder.getDictionaryAttr({startTimeAttr, endTimeAttr, minIIAttr});
  }
};

class OpFusionAnalysis : public HIRPassImplBase<circt::hir::FuncOp> {
public:
  OpFusionAnalysis(circt::hir::FuncOp op);
  void getAnalysis(llvm::DenseMap<mlir::StringRef, AccessInfo> &);

private:
  mlir::LogicalResult visitOp(circt::hir::CallOp);
  llvm::DenseMap<mlir::StringRef, AccessInfo> *mapLabelToAccessInfo;
};