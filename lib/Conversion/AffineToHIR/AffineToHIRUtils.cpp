//===- AffineToHIRUtils.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Conversion/AffineToHIR.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <set>
#include <stack>

using namespace mlir;
using namespace circt;
using namespace hir;

// HIRValue class.
HIRValue::HIRValue(Value value) : value(value) {
  auto *definingOp = value.getDefiningOp();
  assert(definingOp);
  assert(isa<hw::ConstantOp>(definingOp) ||
         isa<mlir::arith::ConstantOp>(definingOp));
}

HIRValue::HIRValue(Value value, Value timeVar, int64_t offset)
    : value(value), timeVar(timeVar), offset(offset) {
  assert(value.getType().isIntOrFloat());
}
bool HIRValue::isConstant() const { return !isContainerType() && !timeVar; }
bool HIRValue::isContainerType() const {
  return value.getType().isa<circt::hir::MemrefType>();
}
Value HIRValue::getValue() const {
  assert(value);
  return value;
}
Value HIRValue::getTimeVar() const {
  assert(timeVar);
  return timeVar;
}
int64_t HIRValue::getOffset() const {
  assert(timeVar);
  return offset;
}

// BlockArgManager.

BlockArgManager::BlockArgManager(mlir::func::FuncOp funcOp) {
  funcOp->walk([this](Operation *operation) {
    if (operation->getNumOperands() > 0) {
      addCapturedOperandsToAllParentBlocks(operation);
    }
    return WalkResult::advance();
  });
}
ArrayRef<Value> BlockArgManager::getCapturedValues(Block *blk) {
  return mapBlockToCapturedValues[blk];
}

// FIXME: Use DenseSet instead of this function.
void insertIfNotPresent(SmallVectorImpl<Value> &array, Value value) {
  bool alreadyPresent = false;
  for (auto v : array) {
    if (v == value)
      alreadyPresent = true;
  }
  if (!alreadyPresent)
    array.push_back(value);
}

void BlockArgManager::addCapturedOperandsToAllParentBlocks(
    Operation *operation) {

  for (Value operand : operation->getOperands()) {
    if (operand.getType().isa<mlir::MemRefType>())
      continue;
    assert(operand.getType().isIntOrIndexOrFloat());
    auto *parentBlk = operation->getBlock();
    while (parentBlk != operand.getParentBlock()) {
      insertIfNotPresent(mapBlockToCapturedValues[parentBlk], operand);
      parentBlk = parentBlk->getParentOp()->getBlock();
    }
  }
}

// ValueConverter class.
HIRValue ValueConverter::getBlockLocalValue(OpBuilder &builder, Value mlirValue,
                                            Block *blk) {
  assert(mlirValue.getType().isIntOrIndexOrFloat());
  auto iter = mapValueToBlockLocalHIRValue.find(std::make_pair(blk, mlirValue));
  assert(iter != mapValueToBlockLocalHIRValue.end());
  return iter->second;
}

HIRValue ValueConverter::getDelayedBlockLocalValue(OpBuilder &builder,
                                                   Value mlirValue,
                                                   Value timeVar,
                                                   int64_t offset) {
  assert(mlirValue.getType().isIntOrIndexOrFloat());
  assert(timeVar);
  auto *blk = timeVar.getParentBlock();
  auto blkLocalValue = getBlockLocalValue(builder, mlirValue, blk);
  return getDelayedValue(builder, mlirValue.getLoc(), blkLocalValue, timeVar,
                         offset)
      .getValue();
}

Value ValueConverter::getMemref(Value mlirMemRef) {
  assert(mlirMemRef.getType().isa<mlir::MemRefType>());
  auto iter = mapMemrefToHIR.find(mlirMemRef);
  assert(iter != mapMemrefToHIR.end());
  return iter->second;
}

SmallVector<HIRValue, 4>
ValueConverter::getBlockLocalValues(OpBuilder &builder, ValueRange mlirValues,
                                    Value timeVar, int64_t offset) {
  SmallVector<HIRValue, 4> hirValues;
  for (auto v : mlirValues) {
    hirValues.push_back(getDelayedBlockLocalValue(builder, v, timeVar, offset));
  }
  return hirValues;
}

SmallVector<HIRValue, 4>
ValueConverter::getBlockLocalValues(OpBuilder &builder, ValueRange mlirValues,
                                    Block *blk) {
  SmallVector<HIRValue, 4> hirValues;
  for (auto v : mlirValues) {
    hirValues.push_back(getBlockLocalValue(builder, v, blk));
  }
  return hirValues;
}

void ValueConverter::mapValueToHIRValue(Value mlirValue, HIRValue hirValue,
                                        Block *blk) {
  assert(mapValueToBlockLocalHIRValue.find(std::make_pair(blk, mlirValue)) ==
         mapValueToBlockLocalHIRValue.end());
  mapValueToBlockLocalHIRValue[std::make_pair(blk, mlirValue)] = hirValue;
}

void ValueConverter::mapMemref(Value mlirMemRef, Value hirMemref) {
  assert(mapMemrefToHIR.find(mlirMemRef) == mapMemrefToHIR.end());
  mapMemrefToHIR[mlirMemRef] = hirMemref;
}

Optional<HIRValue> ValueConverter::getDelayedValue(OpBuilder &builder,
                                                   Location errLoc,
                                                   HIRValue hirValue,
                                                   Value destTimeVar,
                                                   int64_t destOffset) {
  assert(hirValue.getValue());
  assert(!hirValue.isContainerType());

  if (hirValue.isConstant())
    return hirValue;
  assert(hirValue.getValue().getType().isIntOrFloat());
  auto srcTimeVar = hirValue.getTimeVar();
  auto srcOffset = hirValue.getOffset();
  assert(srcTimeVar == destTimeVar);

  auto delay = destOffset - srcOffset;
  if (delay < 0) {
    emitError(
        errLoc,
        "Wrong schedule: Time offset of destination is less than the source.");
    return hirValue;
  }
  Value delayedValue;
  if (delay > 0) {
    auto delayAttr = builder.getI64IntegerAttr(delay);
    delayedValue = builder.create<hir::DelayOp>(
        builder.getUnknownLoc(), hirValue.getValue().getType(),
        hirValue.getValue(), delayAttr, srcTimeVar,
        builder.getI64IntegerAttr(srcOffset));
  } else {
    delayedValue = hirValue.getValue();
  }
  return HIRValue(delayedValue, destTimeVar, destOffset);
}
