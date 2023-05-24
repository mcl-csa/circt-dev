#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/Optional.h"
struct MemrefPragmaHandler {
  enum PortKind { READ_ONLY, WRITE_ONLY, READ_WRITE };
  MemrefPragmaHandler(mlir::Value memref);
  int64_t getNumPorts();
  PortKind getPortKind(int portNum);
  int64_t getRdLatency();
  int64_t getWrLatency();
  size_t getNumRdPorts();
  size_t getNumWrPorts();

private:
  llvm::SmallVector<PortKind, 2> ports;
  llvm::Optional<int64_t> rdLatency;
  llvm::Optional<int64_t> wrLatency;
  // Read-Write ports counted in both;
  size_t numRdPorts;
  size_t numWrPorts;
};

struct AffineForPragmaHandler {
  int64_t getII();
  AffineForPragmaHandler(mlir::AffineForOp op);

private:
  int64_t ii;
};