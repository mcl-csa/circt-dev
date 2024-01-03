#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"
#include <stack>
using namespace mlir;
using namespace circt;
using namespace hir;
class CPUModuleBuilder {
public:
  CPUModuleBuilder(mlir::ModuleOp mod) : mod(mod) {}
  LogicalResult walk();
  void print(llvm::raw_ostream &os);
  void printJSON(llvm::raw_ostream &os);

private:
  LogicalResult visitOp(Operation *op);
  LogicalResult visitOp(hir::FuncExternOp op);
  LogicalResult visitOp(hir::ProbeOp op);
  LogicalResult visitOp(func::FuncOp op);

private:
  mlir::ModuleOp mod;
  llvm::json::Object cosimInfo;
  llvm::json::Array probeStack;
};