#include "CPUModuleBuilder.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;
using namespace circt;
using namespace hir;
// Helper functions.
static void
removeHIRAttrs(std::function<void(StringAttr attrName)> const &removeAttr,
               DictionaryAttr attrList) {
  for (auto attr : attrList) {
    if (llvm::isa_and_nonnull<hir::HIRDialect>(attr.getNameDialect())) {
      removeAttr(attr.getName());
    }
  }
}

LogicalResult CPUModuleBuilder::walk() {
  OpBuilder builder(this->mod);
  builder.setInsertionPointToStart(mod.getBody());
  auto funcTy =
      builder.getFunctionType({builder.getI32Type(), builder.getI32Type()}, {});
  auto op = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), "record", funcTy,
      builder.getStringAttr("private"), ArrayAttr(), ArrayAttr());

  op->setAttr("llvm.emit_c_interface", builder.getUnitAttr());

  // Postorder walk is required because visit to probeOps fill the probeStack
  // and then visit to the function op serializes the probeStack to json.
  this->mod.walk<WalkOrder::PostOrder>([this](Operation *operation) {
    if (isa<hir::ReturnOp>(operation)) {
      return WalkResult::advance();
    }
    if (auto op = dyn_cast<mlir::func::FuncOp>(operation)) {
      if (failed(visitOp(op))) {
        return WalkResult::interrupt();
      }
    }
    if (auto op = dyn_cast<hir::FuncExternOp>(operation)) {
      if (failed(visitOp(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (auto op = dyn_cast<hir::ProbeOp>(operation)) {
      if (failed(visitOp(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    if (failed(visitOp(operation))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(hir::FuncExternOp op) {
  op.erase();
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(func::FuncOp op) {
  if (!op->hasAttrOfType<UnitAttr>("hwAccel"))
    return success();

  op->setAttr("llvm.emit_c_interface", UnitAttr::get(op->getContext()));
  if (op.getArgAttrs()) {
    for (size_t i = 0; i < op.getNumArguments(); i++)
      removeHIRAttrs(
          [&op, i](StringAttr attrName) { op.removeArgAttr(i, attrName); },
          op.getArgAttrDict(i));
  }

  // Remove hir result attributes.
  for (size_t i = 0; i < op->getNumResults(); i++)
    removeHIRAttrs(
        [&op, i](StringAttr attrName) { op.removeResultAttr(i, attrName); },
        op.getResultAttrDict(i));
  return success();
}

func::CallOp emitRecordCall(OpBuilder &builder, Location loc, Value input,
                            IntegerAttr id) {
  auto idVar = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), id);
  return builder.create<func::CallOp>(
      loc, TypeRange(), SymbolRefAttr::get(builder.getStringAttr("record")),
      SmallVector<Value>({input, idVar}));
}

LogicalResult CPUModuleBuilder::visitOp(hir::ProbeOp op) {
  OpBuilder builder(op);
  emitRecordCall(builder, op.getLoc(), op.getInput(),
                 op->getAttrOfType<IntegerAttr>("id"));
  op.erase();
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(Operation *operation) {
  if (llvm::isa_and_nonnull<hir::HIRDialect>(operation->getDialect())) {
    return operation->emitError("Found unsupported HIR operation.");
  }

  removeHIRAttrs(
      [operation](StringAttr attrName) { operation->removeAttr(attrName); },
      operation->getAttrDictionary());
  return success();
}

void CPUModuleBuilder::print(llvm::raw_ostream &os) { mod.print(os); }