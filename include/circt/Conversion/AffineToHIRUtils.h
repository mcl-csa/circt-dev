#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <cstdint>
using namespace mlir;

struct HIRValue {
  HIRValue() = default;
  HIRValue(Value);
  HIRValue(Value value, Value timeVar, int64_t offset);
  bool isConstant() const;
  bool isContainerType() const;
  Value getValue() const;
  Value getTimeVar() const;
  int64_t getOffset() const;

private:
  Value value;
  Value timeVar;
  int64_t offset;
};

/// Identifies all the block local HIR values.
struct BlockArgManager {
  BlockArgManager(mlir::func::FuncOp);
  ArrayRef<Value> getCapturedValues(Block *);

private:
  void addCapturedOperandsToAllParentBlocks(Operation *);

private:
  llvm::DenseMap<Block *, SmallVector<Value>> mapBlockToCapturedValues;
};

/// Manages conversion of Values from affine dialect to hir dialect.
struct ValueConverter {
  HIRValue getDelayedBlockLocalValue(OpBuilder &builder, Value mlirValue,
                                     Value timeVar, int64_t offset);
  HIRValue getBlockLocalValue(OpBuilder &builder, Value mlirValue, Block *blk);
  Value getMemref(Value mlirMemRef);
  SmallVector<HIRValue, 4> getBlockLocalValues(OpBuilder &builder,
                                               ValueRange mlirValues,
                                               Value timeVar, int64_t offset);
  SmallVector<HIRValue, 4> getBlockLocalValues(OpBuilder &builder,
                                               ValueRange mlirValues, Block *);

  void mapValueToHIRValue(Value mlirValue, HIRValue, Block *blk);
  void mapMemref(Value mlirMemRef, Value hirMemref);

private:
  std::optional<HIRValue> getDelayedValue(OpBuilder &builder, Location errLoc,
                                          HIRValue hirValue, Value destTimeVar,
                                          int64_t destOffset);

private:
  llvm::DenseMap<std::pair<Block *, Value>, HIRValue>
      mapValueToBlockLocalHIRValue;
  llvm::DenseMap<Value, Value> mapMemrefToHIR;
};
