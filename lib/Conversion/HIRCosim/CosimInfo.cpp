#include "CosimInfo.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include <mlir/IR/Visitors.h>

using namespace mlir;
using namespace circt;
namespace {
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
} // namespace

CosimInfo::CosimInfo(ModuleOp mod) { this->mod = mod; }

LogicalResult CosimInfo::visitOp(func::FuncOp op) {
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
  funcInfo["probes"] = std::move(this->probes);
  this->probes.clear();
  this->cosimInfo[op.getSymName()] = std::move(funcInfo);
  return success();
}

LogicalResult CosimInfo::visitOp(hir::ProbeOp op) {
  llvm::json::Object probe;
  probe["name"] = op.getVerilogName();
  probe["id"] = op->getAttrOfType<IntegerAttr>("id").getInt();
  this->probes.push_back(std::move(probe));
  return success();
}

LogicalResult CosimInfo::walk() {
  auto walkResult = this->mod.walk([this](Operation *operation) {
    if (auto op = dyn_cast<func::FuncOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    } else if (auto op = dyn_cast<hir::ProbeOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  return success();
}
void CosimInfo::print(llvm::raw_ostream &os) {
  os << llvm::json::Value(std::move(cosimInfo));
}