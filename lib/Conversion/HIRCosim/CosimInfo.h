#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"
#include <mlir/IR/BuiltinOps.h>

class CosimInfo {
public:
  CosimInfo(mlir::ModuleOp);
  mlir::LogicalResult walk();
  void print(llvm::raw_ostream &os);

private:
  llvm::json::Array cosimInfo;
  llvm::json::Array probes;
  mlir::ModuleOp mod;
  mlir::LogicalResult visitOp(mlir::func::FuncOp);
  mlir::LogicalResult visitOp(circt::hir::ProbeOp);
};
