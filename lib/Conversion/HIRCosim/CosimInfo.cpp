#include "CosimInfo.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "llvm/Support/FormatVariadicDetails.h"
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
    if (isa<IntegerType>(types[i])) {
      llvm::json::Object arg;
      arg["name"] = name;
      arg["width"] = types[i].getIntOrFloatBitWidth();
      info["Int"] = std::move(arg);
    } else if (isa<FloatType>(types[i])) {
      llvm::json::Object arg;
      arg["name"] = name;
      arg["width"] = types[i].getIntOrFloatBitWidth();
      info["Float"] = std::move(arg);
    } else if (auto memrefTy = dyn_cast<mlir::MemRefType>(types[i])) {
      llvm::json::Object arg;
      arg["name"] = name;
      arg["shape"] = std::vector<int64_t>(memrefTy.getShape());
      arg["ports"] = getMemPortInfo(name, argAttrs[i]);
      llvm::json::Object elementInfo;
      if (isa<IntegerType>(memrefTy.getElementType()))
        elementInfo["Int"] = memrefTy.getElementType().getIntOrFloatBitWidth();
      else if (isa<FloatType>(memrefTy.getElementType()))
        elementInfo["Float"] =
            memrefTy.getElementType().getIntOrFloatBitWidth();
      arg["element"] = std::move(elementInfo);
      info["Memref"] = std::move(arg);
    } else {
      assert(false && "Unsupported type for serialization.");
    }
    argInfo.push_back(std::move(info));
  }
  llvm::json::Object funcInfo;
  funcInfo["name"] = op.getSymName();
  funcInfo["type"] = "function";
  funcInfo["args"] = std::move(argInfo);
  funcInfo["probes"] = std::move(this->probes);
  this->probes.clear();
  this->cosimInfo.push_back(std::move(funcInfo));
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
  auto v = llvm::json::Value(std::move(cosimInfo));
  llvm::format_provider<llvm::json::Value>::format(v, os, "2");
}
