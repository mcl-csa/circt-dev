//===- AffineToHIR.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToHIR.h"
#include "../PassDetail.h"
#include "AffineToHIRUtils.h"
#include "SchedulingAnalysis.h"
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
#include <stack>

using namespace mlir;
using namespace circt;
using namespace hir;

namespace {

struct AffineToHIR : public AffineToHIRBase<AffineToHIR> {
  void runOnOperation() override;
};

class AffineToHIRImpl {
public:
  AffineToHIRImpl(mlir::func::FuncOp funcOp)
      : mlirFuncOp(funcOp), builder(funcOp), blkArgManager(funcOp) {}
  void runOnOperation();

private:
  void pushInsertionBlk(Block &);
  void popInsertionBlk();
  SmallVector<Value> getFlattenedHIRIndices(OperandRange, AffineMap,
                                            hir::MemrefType, Value, int64_t);

private:
  LogicalResult visitOperation(Operation *);
  LogicalResult visitOp(mlir::func::FuncOp);
  LogicalResult visitOp(mlir::func::ReturnOp);
  LogicalResult visitOp(mlir::AffineForOp);
  LogicalResult visitOp(mlir::AffineLoadOp);
  LogicalResult visitOp(mlir::AffineStoreOp);
  LogicalResult visitOp(mlir::AffineYieldOp);
  LogicalResult visitOp(mlir::arith::ConstantOp);
  LogicalResult visitOp(mlir::memref::AllocaOp);
  LogicalResult visitFFIOp(Operation *);

private:
  mlir::func::FuncOp mlirFuncOp;
  OpBuilder builder;
  std::unique_ptr<SchedulingAnalysis> schedulingAnalysis;
  BlockArgManager blkArgManager;
  ValueConverter valueConverter;
  std::stack<OpBuilder::InsertionGuard> insertionGuards;
  llvm::DenseMap<std::pair<Value, Region *>, Value> mapValueToRegionArg;
};

} // namespace

// Helper functions.
SmallVector<DimKind> getDimKinds(int numDims, DictionaryAttr attr) {
  SmallVector<DimKind> out;
  if (auto dimKinds = attr.getNamed("hir.bank_dims")) {
    auto bankDims = dimKinds.getValue().getValue().dyn_cast<ArrayAttr>();
    for (auto isBanked : bankDims) {
      if (isBanked.dyn_cast<mlir::BoolAttr>().getValue() == true) {
        out.push_back(DimKind::BANK);
      } else {
        out.push_back(DimKind::ADDR);
      }
    }
  } else {
    out.append(numDims, DimKind::ADDR);
  }
  return out;
}

Value emitI64Expr(OpBuilder &builder, ArrayRef<HIRValue> i64Values,
                  ArrayRef<int64_t> flattenedExprCoeffs, Value destTimeVar,
                  int64_t destOfffset) {

  auto uLoc = builder.getUnknownLoc();
  int64_t constCoeff = flattenedExprCoeffs.back();
  Value exprResult = builder.create<circt::hw::ConstantOp>(
      uLoc, builder.getI64IntegerAttr(constCoeff));
  for (size_t n = 0; n < i64Values.size(); n++) {
    auto coeff = flattenedExprCoeffs[n];
    Value vCoeff = builder.create<circt::hw::ConstantOp>(
        uLoc, builder.getI64IntegerAttr(coeff));
    Value value = i64Values[n].getValue();
    assert(value.getType().isa<IntegerType>() &&
           value.getType().getIntOrFloatBitWidth() == 64);
    Value prod = builder.create<circt::comb::MulOp>(uLoc, vCoeff, value);
    exprResult = builder.create<circt::comb::AddOp>(uLoc, exprResult, prod);
  }
  return exprResult;
}
SmallVector<Value> AffineToHIRImpl::getFlattenedHIRIndices(
    OperandRange indices, AffineMap affineMap, hir::MemrefType memrefTy,
    Value timeVar, int64_t offset) {
  SmallVector<Value> hirIndices;
  auto hirValues =
      valueConverter.getBlockLocalValues(builder, indices, timeVar, offset);
  auto dimKinds = memrefTy.getDimKinds();
  auto shape = memrefTy.getShape();
  for (size_t i = 0; i < dimKinds.size(); i++) {
    SmallVector<int64_t> flattenedExprCoeffs;
    assert(getFlattenedAffineExpr(affineMap.getResult(i),
                                  affineMap.getNumDims(), 0,
                                  &flattenedExprCoeffs)
               .succeeded());

    assert(flattenedExprCoeffs.size() == hirValues.size() + 1);
    Value finalIdx;
    if (dimKinds[i] == DimKind::ADDR) {
      finalIdx =
          emitI64Expr(builder, hirValues, flattenedExprCoeffs, timeVar, offset);
      auto idxWidth = helper::clog2(shape[i]);
      if (idxWidth != 64) {
        assert(idxWidth < 64);
        auto idxTy = builder.getIntegerType(helper::clog2(shape[i]));
        finalIdx = builder.create<circt::comb::ExtractOp>(
            builder.getUnknownLoc(), idxTy, finalIdx,
            builder.getI32IntegerAttr(0));
      }
    } else {
      int idxLoc = -1;
      int64_t constCoeff = flattenedExprCoeffs.back();
      for (size_t loc = 0; loc < flattenedExprCoeffs.size(); loc++) {
        if (flattenedExprCoeffs[loc] != 0) {
          idxLoc = loc;
          break;
        }
      }
      assert(idxLoc == -1 || constCoeff == 0);
      if (idxLoc > -1) {
        assert(flattenedExprCoeffs[idxLoc] == 1);
        finalIdx = hirValues[idxLoc].getValue();
      } else {
        finalIdx = builder.create<mlir::arith::ConstantOp>(
            builder.getUnknownLoc(), builder.getIndexAttr(constCoeff));
      }
    }
    hirIndices.push_back(finalIdx);
  }
  return hirIndices;
}

Type getHIRValueType(Type ty) {
  if (ty.isa<IntegerType>())
    return ty;
  if (ty.isa<FloatType>()) {
    return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth());
  }
  assert(false && "Type must be a simple type like int or float.");
}
SmallVector<Type> getHIRValueTypes(TypeRange types) {
  SmallVector<Type> hirTypes;
  for (auto ty : types) {
    hirTypes.push_back(getHIRValueType(ty));
  }
  return hirTypes;
}

Type getHIRType(Type ty, DictionaryAttr attr) {
  if (auto memrefTy = ty.dyn_cast<mlir::MemRefType>()) {
    if (memrefTy.getNumElements() == 1) {
      assert(memrefTy.getShape().size() == 1);
      // If the memref is of size one then it must be a banked dimension so that
      // we can index it.
      return hir::MemrefType::get(ty.getContext(), memrefTy.getShape(),
                                  getHIRValueType(memrefTy.getElementType()),
                                  SmallVector<DimKind>({DimKind::BANK}));
    }

    return hir::MemrefType::get(ty.getContext(), memrefTy.getShape(),
                                getHIRValueType(memrefTy.getElementType()),
                                getDimKinds(memrefTy.getShape().size(), attr));
  }
  return getHIRValueType(ty);
}

DictionaryAttr getParamAttr(MLIRContext *context, ArrayAttr argAttr, int i) {
  if (!argAttr)
    return DictionaryAttr();
  return argAttr[i].dyn_cast_or_null<DictionaryAttr>();
}

// AffineToHIR methods.
void AffineToHIRImpl::pushInsertionBlk(Block &blk) {
  insertionGuards.push(OpBuilder::InsertionGuard(builder));
  builder.setInsertionPointToStart(&blk);
}

void AffineToHIRImpl::popInsertionBlk() {
  assert(!insertionGuards.empty());
  insertionGuards.pop();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::func::FuncOp op) {
  auto functionTy = op.getFunctionType();
  auto argAttrs = op->getAttrOfType<ArrayAttr>("arg_attrs");
  auto resAttrs = op->getAttrOfType<ArrayAttr>("res_attrs");
  SmallVector<Type> inputTypes;
  SmallVector<DictionaryAttr> inputAttrs;
  SmallVector<Type> resultTypes;
  SmallVector<DictionaryAttr> resultAttrs;
  for (size_t i = 0; i < functionTy.getNumInputs(); i++) {
    DictionaryAttr attr = getParamAttr(builder.getContext(), argAttrs, i);
    if (!attr)
      return op->emitError(
                 "affine-to-hir pass: Can't get Dictionary attribute for "
                 "input parameter ")
             << i << ".";
    Type ty = getHIRType(functionTy.getInput(i), attr);
    inputTypes.push_back(ty);
    inputAttrs.push_back(attr);
  }
  for (size_t i = 0; i < functionTy.getNumResults(); i++) {
    DictionaryAttr attr = getParamAttr(builder.getContext(), resAttrs, i);
    if (!attr)
      return op->emitError("affine-to-hir pass: Can't convert attribute for "
                           "return parameter ")
             << i << ".";
    Type ty = getHIRType(functionTy.getResult(i), attr);
    resultTypes.push_back(ty);
    resultAttrs.push_back(attr);
  }

  auto funcTy = FuncType::get(builder.getContext(), inputTypes, inputAttrs,
                              resultTypes, resultAttrs);

  builder.setInsertionPoint(op);
  auto argNamesAttr = op->getAttrOfType<ArrayAttr>("argNames");

  if (!argNamesAttr && op.getNumArguments() > 0)
    return op->emitError("Could not find argNames attr.");
  if (argNamesAttr && argNamesAttr.size() != op.getNumArguments())
    return op->emitError("Wrong number of argNames.");

  auto resultNamesAttr = op->getAttrOfType<ArrayAttr>("resultNames");
  if (!resultNamesAttr && op.getNumResults() > 0)
    return op->emitError("Could not find resultNames attr.");
  if (resultNamesAttr && resultNamesAttr.size() != op.getNumResults())
    return op->emitError("Wrong number of resultNames.");

  SmallVector<Attribute> argNames;
  for (auto name : argNamesAttr) {
    argNames.push_back(name);
  }
  argNames.push_back(builder.getStringAttr("t"));
  auto funcOp = builder.create<hir::FuncOp>(
      op->getLoc(), op.getSymName(), funcTy, builder.getArrayAttr(argNames),
      resultNamesAttr);

  // Map function arguments to hir.
  auto mlirFuncOpArguments = op.getBody().front().getArguments();
  auto *bodyBlk = &funcOp.getFuncBody().front();
  for (size_t i = 0; i < mlirFuncOpArguments.size(); i++) {
    auto mlirArg = mlirFuncOpArguments[i];
    auto hirArg = bodyBlk->getArgument(i);
    if (hirArg.getType().isa<hir::MemrefType>()) {
      valueConverter.mapMemref(mlirArg, hirArg);
    } else {
      valueConverter.mapValueToHIRValue(
          mlirArg, HIRValue(hirArg, funcOp.getRegionTimeVar(), 0), bodyBlk);
    }
  }

  // Set the builder insertion location.
  pushInsertionBlk(funcOp.getFuncBody().front());
  return success();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::func::ReturnOp op) {
  auto hirValues = valueConverter.getBlockLocalValues(
      builder, op->getOperands(), builder.getInsertionBlock());

  SmallVector<Value> returnOperands;
  for (auto v : hirValues)
    returnOperands.push_back(v.getValue());
  builder.create<hir::ReturnOp>(op->getLoc(), returnOperands);
  popInsertionBlk();
  return success();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::AffineForOp op) {
  auto lb = builder.create<circt::hw::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getI64IntegerAttr(
          op.getLowerBound().getMap().getSingleConstantResult()));
  auto ub = builder.create<circt::hw::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getI64IntegerAttr(
          op.getUpperBound().getMap().getSingleConstantResult()));
  auto step = builder.create<circt::hw::ConstantOp>(
      builder.getUnknownLoc(), builder.getI64IntegerAttr(op.getStep()));

  Value tRegion = builder.getInsertionBlock()->getArguments().back();
  auto loopII = schedulingAnalysis->getLoopII(op);
  auto offset = schedulingAnalysis->getTimeOffset(op);
  auto offsetAttr = builder.getI64IntegerAttr(offset);

  auto capturedValues =
      blkArgManager.getCapturedValues(op.getInductionVar().getParentBlock());
  SmallVector<Value> iterArgOperands;
  SmallVector<Value> mlirNonConstantValues;
  SmallVector<Value> mlirConstantValues;
  SmallVector<HIRValue> hirConstantValues;

  for (auto mlirValue : capturedValues) {
    auto hirValue = valueConverter.getDelayedBlockLocalValue(builder, mlirValue,
                                                             tRegion, offset);
    if (hirValue.isConstant()) {
      hirConstantValues.push_back(hirValue);
      mlirConstantValues.push_back(mlirValue);
    } else {
      iterArgOperands.push_back(hirValue.getValue());
      mlirNonConstantValues.push_back(mlirValue);
    }
  }
  auto forOp = builder.create<hir::ForOp>(
      op.getLoc(), lb, ub, step, iterArgOperands, tRegion, offsetAttr,
      [loopII, this, &mlirConstantValues, &mlirNonConstantValues,
       &hirConstantValues](OpBuilder &builder, Value iv,
                           ArrayRef<Value> iterArgs, Value tLoopBody) {
        for (size_t i = 0; i < iterArgs.size(); i++) {
          auto iterArg = HIRValue(iterArgs[i], tLoopBody, 0);
          valueConverter.mapValueToHIRValue(mlirNonConstantValues[i], iterArg,
                                            iv.getParentBlock());
        }
        for (size_t i = 0; i < mlirConstantValues.size(); i++) {
          auto mlirValue = mlirConstantValues[i];
          auto hirValue = hirConstantValues[i];
          valueConverter.mapValueToHIRValue(mlirValue, hirValue,
                                            iv.getParentBlock());
        }
        SmallVector<Value> yieldOperands;
        for (auto v : mlirNonConstantValues) {
          yieldOperands.push_back(
              valueConverter
                  .getDelayedBlockLocalValue(builder, v, tLoopBody, loopII)
                  .getValue());
        }
        auto nextIterOp = builder.create<hir::NextIterOp>(
            builder.getUnknownLoc(), Value(), yieldOperands, tLoopBody,
            builder.getI64IntegerAttr(loopII));
        return nextIterOp;
      });

  auto *forOpBodyBlk = forOp.getInductionVar().getParentBlock();
  valueConverter.mapValueToHIRValue(
      op.getInductionVar(),
      HIRValue(forOp.getInductionVar(), forOp.getIterTimeVar(), 0),
      forOpBodyBlk);
  pushInsertionBlk(forOp.getLoopBody().front());
  return success();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::AffineLoadOp op) {
  auto hirMem = valueConverter.getMemref(op.getMemRef());

  auto portAndDelay = schedulingAnalysis->getPortNumAndDelayForMemoryOp(op);
  auto port = builder.getI64IntegerAttr(portAndDelay.first);
  auto delay = builder.getI64IntegerAttr(portAndDelay.second);
  auto offset = schedulingAnalysis->getTimeOffset(op);
  auto offsetAttr = builder.getI64IntegerAttr(offset);
  auto tRegion = builder.getInsertionBlock()->getArguments().back();
  assert(tRegion);
  auto hirIndices = getFlattenedHIRIndices(
      op.getIndices(), op.getAffineMap(),
      hirMem.getType().dyn_cast<hir::MemrefType>(), tRegion, offset);
  auto loadOp = builder.create<hir::LoadOp>(
      op->getLoc(), getHIRValueType(op.getResult().getType()), hirMem,
      hirIndices, port, delay, tRegion, offsetAttr);
  auto resultDelay = op->getAttrOfType<ArrayAttr>("result_delays")[0]
                         .dyn_cast<IntegerAttr>()
                         .getInt();
  valueConverter.mapValueToHIRValue(
      op.getResult(), {loadOp.getResult(), tRegion, offset + resultDelay},
      loadOp->getBlock());
  return success();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::AffineStoreOp op) {

  Value hirMem = valueConverter.getMemref(op.getMemRef());

  auto portAndDelay = schedulingAnalysis->getPortNumAndDelayForMemoryOp(op);
  auto port = builder.getI64IntegerAttr(portAndDelay.first);
  auto delay = builder.getI64IntegerAttr(portAndDelay.second);

  int64_t offset = schedulingAnalysis->getTimeOffset(op);
  auto offsetAttr = builder.getI64IntegerAttr(offset);
  auto tRegion = builder.getInsertionBlock()->getArguments().back();
  assert(tRegion);
  auto hirIndices = getFlattenedHIRIndices(
      op.getIndices(), op.getAffineMap(),
      hirMem.getType().dyn_cast<hir::MemrefType>(), tRegion, offset);

  auto hirValue = valueConverter.getDelayedBlockLocalValue(
      builder, op.getValue(), tRegion, offset);

  builder.create<hir::StoreOp>(op->getLoc(), hirValue.getValue(), hirMem,
                               hirIndices, port, delay, tRegion, offsetAttr);
  return success();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::AffineYieldOp op) {
  popInsertionBlk();
  return success();
}

LogicalResult AffineToHIRImpl::visitOp(mlir::arith::ConstantOp op) {
  auto ty = op.getResult().getType();
  assert(!ty.isSignedInteger());
  if (ty.isIndex()) {
    Value v = builder.clone(*op.getOperation())->getResult(0);
    valueConverter.mapValueToHIRValue(op.getResult(), HIRValue(v),
                                      v.getParentBlock());

    return success();
  }
  if (ty.isSignlessInteger()) {
    Value v = builder.create<circt::hw::ConstantOp>(
        op->getLoc(), op.getValue().dyn_cast<IntegerAttr>());
    valueConverter.mapValueToHIRValue(op.getResult(), HIRValue(v),
                                      v.getParentBlock());

    return success();
  }
  if (ty.isa<FloatType>()) {
    auto bits =
        op.getValue().dyn_cast<mlir::FloatAttr>().getValue().bitcastToAPInt();
    auto bitsAttr =
        IntegerAttr::get(builder.getIntegerType(bits.getBitWidth()), bits);
    Value v = builder.create<circt::hw::ConstantOp>(op->getLoc(), bitsAttr);
    valueConverter.mapValueToHIRValue(op.getResult(), HIRValue(v),
                                      v.getParentBlock());
    return success();
  }
  return op->emitError("Only signless integer and floats are supported.");
}

MemKindEnumAttr toMemKindAttr(StringAttr memKindStrAttr) {
  auto str = memKindStrAttr.strref();
  if (str == "reg")
    return MemKindEnumAttr::get(memKindStrAttr.getContext(), MemKindEnum::reg);
  if (str == "bram")
    return MemKindEnumAttr::get(memKindStrAttr.getContext(), MemKindEnum::bram);
  if (str == "lutram")
    return MemKindEnumAttr::get(memKindStrAttr.getContext(),
                                MemKindEnum::lutram);
  assert(false);
}

LogicalResult AffineToHIRImpl::visitOp(mlir::memref::AllocaOp op) {
  DictionaryAttr attr = op->getAttrDictionary();
  auto memKindStrAttr = attr.getAs<StringAttr>("mem_kind");
  auto ports = helper::extractMemrefPortsFromDict(attr);

  if (!ports.hasValue())
    return op->emitError("Could not find hir.memref.ports attr.");
  if (!memKindStrAttr)
    return op->emitError("Could not find mem_kind attr.");

  auto hirTy = getHIRType(op.memref().getType(), attr);

  auto allocaOp = builder.create<hir::AllocaOp>(
      op->getLoc(), hirTy, toMemKindAttr(memKindStrAttr), ports.getValue());

  if (op->hasAttrOfType<ArrayAttr>("names"))
    allocaOp->setAttr("names", op->getAttr("names"));
  valueConverter.mapMemref(op.getResult(), allocaOp.getResult());

  return success();
}

LogicalResult AffineToHIRImpl::visitFFIOp(Operation *operation) {
  auto hirFuncAttr =
      operation->getAttrOfType<mlir::FlatSymbolRefAttr>("hir_function");
  if (!hirFuncAttr)
    return operation->emitError("Could not find hir_function attribute");

  auto topLevelModuleOp = operation->getParentOfType<mlir::ModuleOp>();
  assert(topLevelModuleOp);
  auto *hirFuncExternOperation = topLevelModuleOp.lookupSymbol(hirFuncAttr);
  if (!hirFuncExternOperation)
    return operation->emitError("Could not find declaration of hir function ")
           << hirFuncAttr << ".";

  auto hirFuncExternOp = dyn_cast<hir::FuncExternOp>(hirFuncExternOperation);
  assert(hirFuncExternOp);
  Value tRegion = builder.getInsertionBlock()->getArguments().back();
  auto offset = schedulingAnalysis->getTimeOffset(operation);
  auto offsetAttr = builder.getI64IntegerAttr(offset);
  auto hirOperands = valueConverter.getBlockLocalValues(
      builder, operation->getOperands(), tRegion, offset);

  SmallVector<Value> operands;
  for (auto hirOperand : hirOperands) {
    operands.push_back(hirOperand.getValue());
  }
  auto callOp = builder.create<hir::CallOp>(
      operation->getLoc(), getHIRValueTypes(operation->getResultTypes()),
      hirFuncAttr, builder.getStringAttr(hirFuncAttr.getValue()),
      hirFuncExternOp.funcTyAttr(), operands, tRegion, offsetAttr);
  assert(callOp->getNumResults() == operation->getNumResults());
  auto resultDelays = operation->getAttrOfType<ArrayAttr>("result_delays");
  if (!resultDelays && callOp->getNumResults() > 0) {
    return operation->emitError("Could not find result_delays");
  }
  if (resultDelays.size() != operation->getNumResults()) {
    return operation->emitError("Number of result delays is wrong.");
  }
  for (size_t i = 0; i < callOp->getNumResults(); i++) {
    auto resultDelay = resultDelays[i].dyn_cast<IntegerAttr>().getInt();
    valueConverter.mapValueToHIRValue(
        operation->getResult(i),
        HIRValue(callOp->getResult(i), tRegion, offset + resultDelay),
        callOp->getBlock());
  }
  return success();
}

LogicalResult AffineToHIRImpl::visitOperation(Operation *operation) {
  if (auto op = dyn_cast<mlir::func::FuncOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::func::ReturnOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::AffineForOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::AffineLoadOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::AffineStoreOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::AffineYieldOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::arith::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::memref::AllocaOp>(operation))
    return visitOp(op);
  if (isa<arith::ArithmeticDialect>(operation->getDialect()))
    return visitFFIOp(operation);
  if (isa<mlir::func::CallOp>(operation))
    return visitFFIOp(operation);
  return operation->emitError("Unknown operation for affine-to-hir pass.");
}

void AffineToHIRImpl::runOnOperation() {
  std::string logFile = "/dev/null";
  schedulingAnalysis = std::make_unique<SchedulingAnalysis>(
      SchedulingAnalysis(mlirFuncOp, logFile));
  if (!schedulingAnalysis->hasSolution()) {
    mlirFuncOp->emitError("Failed to find a schedule.");
    return;
  }
  mlirFuncOp.walk<WalkOrder::PreOrder>([this](Operation *operation) {
    if (failed(visitOperation(operation))) {
      operation->emitError("Could not convert this to hir.");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
}

//-----------------------------------------------------------------------------
// AffineToHIR methods.
//-----------------------------------------------------------------------------
void AffineToHIR::runOnOperation() {
  getOperation()->walk([](Operation *operation) {
    if (auto funcOp = dyn_cast<mlir::func::FuncOp>(operation)) {
      if (funcOp->getAttrOfType<mlir::UnitAttr>("hwAccel")) {
        AffineToHIRImpl impl(funcOp);
        impl.runOnOperation();
      }
      funcOp->erase();
    }
    return WalkResult::advance();
  });
}
//-----------------------------------------------------------------------------

std::unique_ptr<mlir::Pass> circt::createAffineToHIR() {
  return std::make_unique<AffineToHIR>();
}
