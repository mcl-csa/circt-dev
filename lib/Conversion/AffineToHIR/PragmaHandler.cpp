#include "PragmaHandler.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
using namespace mlir;

MemrefPragmaHandler::MemrefPragmaHandler(Value memref) : ramKind(SMP) {
  assert(memref.getType().isa<mlir::MemRefType>());

  std::string memKindStr;
  ArrayAttr portAttrs;

  if (auto *definingOp = memref.getDefiningOp()) {
    memKindStr = definingOp->getAttrOfType<StringAttr>("mem_kind").str();
    portAttrs = definingOp->getAttrOfType<ArrayAttr>("hir.memref.ports");
  } else {
    auto funcOpArgs =
        dyn_cast<mlir::func::FuncOp>(memref.getParentRegion()->getParentOp())
            .getBody()
            .getArguments();
    for (size_t i = 0; i < funcOpArgs.size(); i++) {
      if (funcOpArgs[i] == memref) {
        portAttrs = dyn_cast<mlir::func::FuncOp>(
                        memref.getParentRegion()->getParentOp())
                        .getArgAttrOfType<ArrayAttr>(i, "hir.memref.ports");
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
}

int64_t MemrefPragmaHandler::getRdPortID(int64_t n) { return rdPorts[n]; }
int64_t MemrefPragmaHandler::getWrPortID(int64_t n) { return wrPorts[n]; }
size_t MemrefPragmaHandler::getNumRdPorts() { return rdPorts.size(); }
size_t MemrefPragmaHandler::getNumWrPorts() { return wrPorts.size(); }
MemrefPragmaHandler::PortKind MemrefPragmaHandler::getPortKind(int portNum) {
  return ports[portNum];
}
MemrefPragmaHandler::RAMKind MemrefPragmaHandler::getRAMKind() {
  return ramKind;
}
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