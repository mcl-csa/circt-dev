#include "PragmaHandler.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
using namespace mlir;
using namespace circt;
using DimKind = MemrefPragmaHandler::DimKind;
FuncExternPragmaHandler::FuncExternPragmaHandler(mlir::func::CallOp op) {
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  auto funcOp = dyn_cast_or_null<hir::FuncExternOp>(
      moduleOp.lookupSymbol(op.getCallee()));
  if (!funcOp) {
    op.emitError("Expected callee to be hir FuncExternOp.");
  }
  assert(funcOp);
  auto hirFuncTy = funcOp.getFuncType();
  for (auto attr : hirFuncTy.getInputAttrs()) {
    auto delay = helper::getHIRDelayAttr(attr);
    if (delay)
      argDelays.push_back(*delay);
    else
      argDelays.push_back(llvm::None);
  }
  for (auto attr : hirFuncTy.getResultAttrs()) {
    auto delay = helper::getHIRDelayAttr(attr);
    if (delay)
      resultDelays.push_back(*delay);
    else
      resultDelays.push_back(llvm::None);
  }
}

llvm::Optional<size_t> FuncExternPragmaHandler::getArgDelay(size_t i) {
  return argDelays[i];
}

llvm::Optional<size_t> FuncExternPragmaHandler::getResultDelay(size_t i) {
  return resultDelays[i];
}

MemrefPragmaHandler::MemrefPragmaHandler(Value memref) : ramKind(SMP) {
  assert(memref.getType().isa<mlir::MemRefType>());

  ArrayAttr portAttrs;
  ArrayAttr isBankedArr;

  if (auto *definingOp = memref.getDefiningOp()) {
    portAttrs = definingOp->getAttrOfType<ArrayAttr>("hir.memref.ports");
    isBankedArr = definingOp->getAttrOfType<ArrayAttr>("hir.bank_dims");
  } else {
    auto funcOpArgs =
        dyn_cast<mlir::func::FuncOp>(memref.getParentRegion()->getParentOp())
            .getBody()
            .getArguments();
    for (size_t i = 0; i < funcOpArgs.size(); i++) {
      if (funcOpArgs[i] == memref) {
        auto funcOp = dyn_cast<mlir::func::FuncOp>(
            memref.getParentRegion()->getParentOp());
        portAttrs = funcOp.getArgAttrOfType<ArrayAttr>(i, "hir.memref.ports");
        isBankedArr = funcOp.getArgAttrOfType<ArrayAttr>(i, "hir.bank_dims");
        break;
      }
    }
  }

  // ports
  int64_t portID = 0;
  for (auto portAttr : portAttrs) {
    auto rdLatAttr = portAttr.dyn_cast<DictionaryAttr>().get("rd_latency");
    auto wrLatAttr = portAttr.dyn_cast<DictionaryAttr>().get("wr_latency");
    assert(rdLatAttr || wrLatAttr);

    if (rdLatAttr) {
      if (this->rdLatency)
        assert(this->rdLatency == rdLatAttr.dyn_cast<IntegerAttr>().getInt());
      this->rdLatency = rdLatAttr.dyn_cast<IntegerAttr>().getInt();
    }
    if (wrLatAttr) {
      if (this->wrLatency)
        assert(this->wrLatency == wrLatAttr.dyn_cast<IntegerAttr>().getInt());
      this->wrLatency = wrLatAttr.dyn_cast<IntegerAttr>().getInt();
    }

    if (rdLatAttr && wrLatAttr) {
      this->ports.push_back(READ_WRITE);
      this->rdPorts.push_back(portID);
      this->wrPorts.push_back(portID);
      // All ports must be read-write XOR (read-only OR write-only)
      assert(portID == 0 | ramKind == TMP);
      ramKind = TMP;
    } else if (rdLatAttr) {
      this->ports.push_back(READ_ONLY);
      this->rdPorts.push_back(portID);
      // All ports must be read-write XOR (read-only OR write-only)
      assert(ramKind == SMP);
    } else if (wrLatAttr) {
      this->ports.push_back(WRITE_ONLY);
      this->wrPorts.push_back(portID);
      // All ports must be read-write XOR (read-only OR write-only)
      assert(ramKind == SMP);
    }
    portID++;
  }

  // DimKind.
  size_t nDims = memref.getType().dyn_cast<MemRefType>().getShape().size();
  for (size_t i = 0; i < nDims; i++) {
    if (!isBankedArr)
      this->dimKinds.push_back(DimKind::ADDR);
    else if (isBankedArr[i].dyn_cast<mlir::BoolAttr>().getValue())
      this->dimKinds.push_back(DimKind::BANK);
    else
      this->dimKinds.push_back(DimKind::ADDR);
  }
}

int64_t MemrefPragmaHandler::getRdPortID(int64_t n) { return rdPorts[n]; }
int64_t MemrefPragmaHandler::getWrPortID(int64_t n) { return wrPorts[n]; }
size_t MemrefPragmaHandler::getNumRdPorts() { return rdPorts.size(); }
size_t MemrefPragmaHandler::getNumWrPorts() { return wrPorts.size(); }
MemrefPragmaHandler::PortKind MemrefPragmaHandler::getPortKind(int portNum) {
  return ports[portNum];
}
MemrefPragmaHandler::RAMKind MemrefPragmaHandler::getRAMKind() {
  assert(this->ramKind == MemrefPragmaHandler::RAMKind::SMP ||
         this->ramKind == MemrefPragmaHandler::RAMKind::TMP);
  return this->ramKind;
}

size_t MemrefPragmaHandler::getNumDims() { return dimKinds.size(); }
DimKind MemrefPragmaHandler::getDimKind(size_t i) { return dimKinds[i]; }

int64_t MemrefPragmaHandler::getRdLatency() {
  return this->rdLatency.getValue();
}
int64_t MemrefPragmaHandler::getWrLatency() {
  return this->wrLatency.getValue();
}

int64_t AffineForPragmaHandler::getII() {
  assert(ii > 0);
  return ii;
}
AffineForPragmaHandler::AffineForPragmaHandler(mlir::AffineForOp op) {
  this->ii = op->getAttrOfType<IntegerAttr>("II").getInt();
}