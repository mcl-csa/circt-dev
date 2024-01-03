#include "CPUModuleBuilder.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/JSON.h"
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
  auto funcTy = builder.getFunctionType(builder.getI32Type(),
                                        llvm::SmallVector<Type>({}));
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

llvm::json::Array getMemPortInfo(std::string &memName, Attribute argAttr) {
  auto portsAttr =
      argAttr.dyn_cast<DictionaryAttr>().getAs<ArrayAttr>("hir.memref.ports");
  llvm::json::Array portInfoArray;
  int portNum = 0;
  for (auto port : portsAttr) {
    auto portInfo = llvm::json::Object();
    portInfo["name"] = memName + "_p" + std::to_string(portNum++);
    if (helper::isMemrefRdPort(port))
      portInfo["rd_latency"] = helper::getMemrefPortRdLatency(port).value();
    if (helper::isMemrefWrPort(port))
      portInfo["wr_latency"] = helper::getMemrefPortWrLatency(port).value();
    portInfoArray.push_back(std::move(portInfo));
  }
  return portInfoArray;
}

LogicalResult writeInfoToJson(func::FuncOp op, llvm::json::Object &cosimInfo,
                              llvm::json::Array &probeStack) {
  auto funcInfo = llvm::json::Object();
  funcInfo["type"] = "function";
  auto types = op.getArgumentTypes();
  auto argAttrs = op.getArgAttrs().value();
  auto names = op->getAttrOfType<ArrayAttr>("argNames");
  if (!names) {
    return op.emitError("Could not find argNames.");
  }
  assert(names.size() == op.getNumArguments());

  llvm::json::Array argInfo;
  for (size_t i = 0; i < op.getNumArguments(); i++) {
    llvm::json::Object info;
    auto name = names[i].dyn_cast<StringAttr>().str();
    info["name"] = name;
    if (isa<IntegerType>(types[i])) {
      info["type"] = "integer";
      info["width"] = types[i].getIntOrFloatBitWidth();
    } else if (isa<FloatType>(types[i])) {
      info["type"] = "float";
      info["width"] = types[i].getIntOrFloatBitWidth();
    } else if (auto memrefTy = dyn_cast<mlir::MemRefType>(types[i])) {
      info["type"] = "memref";
      info["shape"] = std::vector<int64_t>(memrefTy.getShape());
      info["ports"] = getMemPortInfo(name, argAttrs[i]);
      llvm::json::Object elementInfo;
      if (isa<IntegerType>(memrefTy.getElementType()))
        elementInfo["type"] = "integer";
      else if (isa<FloatType>(memrefTy.getElementType()))
        elementInfo["type"] = "float";
      elementInfo["width"] = memrefTy.getElementType().getIntOrFloatBitWidth();
      info["element"] = std::move(elementInfo);
    } else {
      assert(false && "Unsupported type for serialization.");
    }
    argInfo.push_back(std::move(info));
  }
  funcInfo["args"] = std::move(argInfo);
  funcInfo["probes"] = std::move(probeStack);
  cosimInfo[op.getSymName()] = std::move(funcInfo);
  return success();
}

LogicalResult CPUModuleBuilder::visitOp(func::FuncOp op) {
  if (!op->hasAttrOfType<UnitAttr>("hwAccel"))
    return success();

  if (failed(writeInfoToJson(op, this->cosimInfo, probeStack)))
    return failure();
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

LogicalResult CPUModuleBuilder::visitOp(hir::ProbeOp op) {
  llvm::json::Object signal;
  signal["name"] = op.getVerilogName();
  signal["id"] = op->getAttrOfType<IntegerAttr>("id").getInt();
  this->probeStack.push_back(std::move(signal));
  OpBuilder builder(op);
  builder.create<mlir::func::CallOp>(
      op.getLoc(), TypeRange(),
      SymbolRefAttr::get(builder.getStringAttr("record")), op.getInput());
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
void CPUModuleBuilder::printJSON(llvm::raw_ostream &os) {
  os << llvm::json::Value(std::move(cosimInfo));
}