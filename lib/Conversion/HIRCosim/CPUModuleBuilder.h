#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace circt;
using namespace hir;
class CPUModuleBuilder {
public:
  CPUModuleBuilder(mlir::ModuleOp mod) : mod(mod) {}
  LogicalResult walk();
  void print(llvm::raw_ostream &os);

private:
  LogicalResult visitOp(Operation *op);
  LogicalResult visitOp(hir::FuncExternOp op);
  LogicalResult visitOp(hir::ProbeOp op);
  LogicalResult visitOp(func::FuncOp op);

private:
  mlir::ModuleOp mod;
};