//===----- HIRToHW.cpp - HIR To HW Conversion Pass-------*-C++-*-===//
//
// This pass converts HIR to HW, Comb and SV dialect.
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HIRToHW.h"
#include "../PassDetail.h"
#include "HIRToHWUtils.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
using namespace circt;

class HIRToHWPass : public HIRToHWBase<HIRToHWPass> {
public:
  void runOnOperation() override;

private:
  void updateHIRToHWMapForFuncInputs(hw::HWModuleOp,
                                     mlir::Block::BlockArgListType,
                                     FuncToHWModulePortMap);

  std::string getUniquePostfix() { return "_" + std::to_string(uniqueInt++); }

  LogicalResult visitHWOp(Operation *);
  LogicalResult visitOperation(Operation *);
  LogicalResult visitOp(hir::BusAssignOp);
  LogicalResult visitOp(hir::BusBroadcastOp);
  LogicalResult visitOp(hir::BusMapOp);
  LogicalResult visitOp(hir::BusOp);
  LogicalResult visitOp(hir::BusRecvOp);
  LogicalResult visitOp(hir::BusSendOp);
  LogicalResult visitOp(hir::BusTensorAssignOp);
  LogicalResult visitOp(hir::BusTensorAssignElementOp);
  LogicalResult visitOp(hir::BusTensorGetElementOp);
  LogicalResult visitOp(hir::BusTensorInsertElementOp);
  LogicalResult visitOp(hir::BusTensorMapOp);
  LogicalResult visitOp(hir::BusTensorOp);
  LogicalResult visitOp(hir::CallOp);
  LogicalResult visitOp(hir::CastOp);
  LogicalResult visitOp(hir::CommentOp);
  LogicalResult visitOp(mlir::arith::ConstantOp);
  LogicalResult visitOp(hir::DelayOp);
  LogicalResult visitOp(hir::FuncExternOp);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::GetClockOp);
  LogicalResult visitOp(hir::GetResetOp);
  LogicalResult visitOp(hir::IsFirstIterOp);
  LogicalResult visitOp(hir::NextIterOp);
  LogicalResult visitOp(hir::ProbeOp);
  LogicalResult visitOp(hir::ReturnOp);
  LogicalResult visitOp(hir::TimeOp);
  LogicalResult visitOp(hir::WhileOp);
  LogicalResult visitOp(hir::WireOp);
  LogicalResult visitOp(hir::DriveOp);
  LogicalResult visitRegion(mlir::Region &);

private:
  Operation *getEmittedHWModuleOp(StringRef);
  hw::InstanceOp getOrCreateHWInstanceOp(Location loc, Operation *hwModuleOp,
                                         StringRef instanceName,
                                         ArrayRef<Value> hwInputs,
                                         ArrayAttr hwParams, Value tstart);

private:
  std::optional<OpBuilder> builder;
  HIRToHWMapping mapHIRToHWValue;
  llvm::DenseMap<StringRef, uint64_t> mapFuncNameToInstanceCount;
  Value clk;
  Value reset;
  hw::HWModuleOp hwModuleOp;
  mlir::ModuleOp mlirModuleOp; // Enclosing module{}
  size_t uniqueInt = 0;
  DenseMap<Value, SmallVector<Value>> mapArrayToElements;
  DenseMap<StringRef, Operation *> mapNameToHWModuleOp;
  DenseMap<StringRef, hw::InstanceOp> mapNameToHWInstanceOp;
};

Value emitMux(OpBuilder &builder, Value select, Value t, Value f) {
  auto combinedArrayOfValues = builder.create<hw::ArrayCreateOp>(
      builder.getUnknownLoc(), SmallVector<Value>({t, f}));

  return builder.create<hw::ArrayGetOp>(builder.getUnknownLoc(),
                                        combinedArrayOfValues, select);
}

LogicalResult HIRToHWPass::visitOp(hir::BusMapOp op) {
  SmallVector<Value> hwOperands;
  for (auto operand : op.getOperands())
    hwOperands.push_back(mapHIRToHWValue.lookup(operand));
  auto results = insertBusMapLogic(*builder, op.getBody().front(), hwOperands);
  for (size_t i = 0; i < op.getNumResults(); i++)
    mapHIRToHWValue.map(op.getResult(i), results[i]);

  return success();
}

Operation *HIRToHWPass::getEmittedHWModuleOp(StringRef hwModuleName) {
  auto *operation = mapNameToHWModuleOp[hwModuleName];
  assert(operation);
  auto hwModuleOp = dyn_cast<hw::HWModuleOp>(operation);
  auto hwModuleExternOp = dyn_cast<hw::HWModuleExternOp>(operation);
  assert(hwModuleOp || hwModuleExternOp);
  if (hwModuleOp)
    return hwModuleOp;
  return hwModuleExternOp;
}

hw::InstanceOp HIRToHWPass::getOrCreateHWInstanceOp(
    Location loc, Operation *hwModuleOp, StringRef instanceName,
    ArrayRef<Value> hwInputs, ArrayAttr hwParams, Value tstart) {
  auto it = mapNameToHWInstanceOp.find(instanceName);
  // If the instance has not been previously generated then create a new
  // instance.
  if (it == mapNameToHWInstanceOp.end()) {
    auto instanceOp = builder->create<hw::InstanceOp>(
        loc, hwModuleOp, instanceName, hwInputs, hwParams);
    mapNameToHWInstanceOp[instanceName] = instanceOp;
    return instanceOp;
  }
  auto instanceOp = it->getSecond();
  assert(instanceOp->getOperands().size() == hwInputs.size());
  // Replace all the prev inputs of the op with mux of hwInputs and prev inputs;
  for (size_t i = 0; i < hwInputs.size(); i++) {
    auto &opOperand = instanceOp->getOpOperand(i);
    opOperand.set(emitMux(*builder, tstart, hwInputs[i], opOperand.get()));
  }
  return it->getSecond();
}

LogicalResult HIRToHWPass::visitOp(hir::BusOp op) {
  // Add a placeholder SSA Var for the buses. CallOp visit will replace them.
  // We need to do this because HW dialect does not have SSA dominance.
  auto *constantXOp = constantX(*builder, op.getType());
  auto placeHolderSSAVar = constantXOp->getResult(0);
  auto name = helper::getOptionalName(constantXOp, 0);
  if (name)
    helper::setNames(constantXOp, {name.value()});
  mapHIRToHWValue.map(op.getRes(), placeHolderSSAVar);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorMapOp op) {
  auto uLoc = builder->getUnknownLoc();
  SmallVector<Value> hwOperandTensors;
  for (auto operand : op.getOperands())
    hwOperandTensors.push_back(mapHIRToHWValue.lookup(operand));
  auto hwArrayTy = hwOperandTensors[0].getType().dyn_cast<hw::ArrayType>();

  // If the tensor has only one element then treat this like BusMapOp.
  if (!hwArrayTy) {
    auto results =
        insertBusMapLogic(*builder, op.getBody().front(), hwOperandTensors);
    for (size_t i = 0; i < op.getNumResults(); i++)
      mapHIRToHWValue.map(op.getResult(i), results[i]);
    return success();
  }

  // Copy the bus map logic as many times as there are elements in the tensor.
  // Save the outputs in the results[array-index][result-number] array.
  SmallVector<SmallVector<Value>> results;
  for (size_t arrayIdx = 0; arrayIdx < hwArrayTy.getSize(); arrayIdx++) {
    SmallVector<Value> hwOperands;
    for (auto hwOperandT : hwOperandTensors)
      hwOperands.push_back(
          insertConstArrayGetLogic(*builder, hwOperandT, arrayIdx));
    results.push_back(
        insertBusMapLogic(*builder, op.getBody().front(), hwOperands));
  }

  // For each result (i.e. results[*][result-num]), create a hw array from the
  // elements.
  for (size_t resultNum = 0; resultNum < op.getNumResults(); resultNum++) {
    SmallVector<Value> resultElements;
    for (size_t arrayIdx = 0; arrayIdx < hwArrayTy.getSize(); arrayIdx++)
      resultElements.push_back(results[arrayIdx][resultNum]);
    mapHIRToHWValue.map(
        op.getResult(resultNum),
        builder->create<hw::ArrayCreateOp>(uLoc, resultElements));
  }
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorOp op) {
  auto *constantXOp =
      getConstantXArray(*builder, op.getType(), mapArrayToElements);
  auto placeHolderSSAVar = constantXOp->getResult(0);
  auto name = helper::getOptionalName(constantXOp, 0);
  if (name)
    helper::setNames(constantXOp, {name.value()});
  mapHIRToHWValue.map(op.getRes(), placeHolderSSAVar);
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::CommentOp op) {
  builder->create<sv::VerbatimOp>(builder->getUnknownLoc(),
                                  "//COMMENT: " + op.getComment());
  return success();
}

LogicalResult HIRToHWPass::visitOp(mlir::arith::ConstantOp op) {
  if (op.getResult().getType().isa<mlir::IntegerType>())
    return op.emitError()
           << "hir-to-hw pass supports only hw.constant op for integer types.";
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::CallOp op) {
  assert(op.getOffset() == 0);
  auto filteredOperands = filterCallOpArgs(op.getFuncType(), op.getOperands());

  // Get the mapped inputs and create the input types for instance op.
  SmallVector<Value> hwInputs;
  for (auto input : filteredOperands.first) {
    auto hwInput = mapHIRToHWValue.lookup(input);
    hwInputs.push_back(hwInput);
  }
  assert(op.getOffset() == 0);

  Value const tstart = mapHIRToHWValue.lookup(op.getTstart());
  hwInputs.push_back(tstart);
  hwInputs.push_back(this->clk);
  hwInputs.push_back(this->reset);

  auto hwInputTypes = helper::getTypes(hwInputs);

  auto sendBuses = filteredOperands.second;
  auto sendBusTypes = helper::getTypes(sendBuses);

  // Create instance op result types.
  SmallVector<Type> hwResultTypes;
  for (auto ty : sendBusTypes)
    hwResultTypes.push_back(*helper::convertToHWType(ty));

  for (auto ty : op.getResultTypes())
    hwResultTypes.push_back(*helper::convertToHWType(ty));

  auto instanceName = op.getInstanceNameAttr();
  assert(instanceName);
  auto *calleeHWModule = getEmittedHWModuleOp(op.getCallee());
  hw::InstanceOp instanceOp =
      getOrCreateHWInstanceOp(op.getLoc(), calleeHWModule, instanceName,
                              hwInputs, getHWParams(op), tstart);
  copyHIRAttrs(op, instanceOp);

  // Map CallOp input send buses to the results of the instance op and replace
  // all prev uses of the placeholder hw ssa vars corresponding to these send
  // buses.
  uint64_t i;
  for (i = 0; i < sendBuses.size(); i++) {
    auto placeHolderSSAVar = mapHIRToHWValue.lookup(sendBuses[i]);
    mapHIRToHWValue.replaceAllHWUses(placeHolderSSAVar,
                                     instanceOp.getResult(i));
  }

  // Map the CallOp return vars to instance op return vars.
  for (uint64_t j = 0; i + j < instanceOp.getNumResults(); j++)
    mapHIRToHWValue.map(op.getResult(j), instanceOp.getResult(i + j));

  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::CastOp op) {
  if (op.getInput().getType().isa<mlir::IndexType>()) {
    assert(op.getRes().getType().isa<mlir::IntegerType>());
    auto constantOp =
        dyn_cast<mlir::arith::ConstantOp>(op.getInput().getDefiningOp());
    assert(constantOp);
    auto value = constantOp.getValue().dyn_cast<mlir::IntegerAttr>().getInt();
    mapHIRToHWValue.map(
        op.getRes(),
        builder->create<hw::ConstantOp>(
            builder->getUnknownLoc(),
            IntegerAttr::get(*helper::convertToHWType(op.getRes().getType()),
                             value)));
    return success();
  }

  assert(*helper::convertToHWType(op.getInput().getType()) ==
         *helper::convertToHWType(op.getRes().getType()));
  mapHIRToHWValue.map(op.getRes(), mapHIRToHWValue.lookup(op.getInput()));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::TimeOp op) {
  auto tIn = mapHIRToHWValue.lookup(op.getTimevar());
  if (!tIn.getType().isa<mlir::IntegerType>())
    return tIn.getDefiningOp()->emitError()
           << "Expected converted type to be i1.";
  auto name = helper::getOptionalName(op, 0);
  Value const res =
      getDelayedValue(*builder, tIn, op.getStartTime().getOffset(), name,
                      op.getLoc(), clk, reset);
  assert(res.getType().isa<mlir::IntegerType>());
  mapHIRToHWValue.map(op.getRes(), res);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::WhileOp op) {
  auto uLoc = builder->getUnknownLoc();
  assert(op.getOffset() == 0);
  auto &bb = op.getBody().front();
  assert(op.getOffset() == 0);

  auto conditionBegin = mapHIRToHWValue.lookup(op.getCondition());
  auto tstartBegin = mapHIRToHWValue.lookup(op.getTstart());

  assert(conditionBegin && tstartBegin);

  auto conditionTrue = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(builder->getI1Type(), 1));

  // Placeholder values.
  auto tstartNextIterOp =
      constantX(*builder, builder->getI1Type())->getResult(0);
  auto conditionNextIterOp =
      constantX(*builder, builder->getI1Type())->getResult(0);
  auto continueCondition = builder->create<comb::XorOp>(
      builder->getUnknownLoc(), conditionNextIterOp, conditionTrue);
  auto tNextBegin = builder->create<comb::AndOp>(builder->getUnknownLoc(),
                                                 tstartBegin, conditionBegin);
  auto tNextIter = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartNextIterOp, continueCondition);
  Value const iterTimeVar =
      builder->create<comb::OrOp>(op.getLoc(), tNextBegin, tNextIter);
  auto notConditionBegin = builder->create<comb::XorOp>(
      builder->getUnknownLoc(), conditionBegin, conditionTrue);
  auto tLastBegin = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartBegin, notConditionBegin);
  auto tLastIter = builder->create<comb::AndOp>(
      builder->getUnknownLoc(), tstartNextIterOp, conditionNextIterOp);
  auto tLast = builder->create<comb::OrOp>(op.getLoc(), tLastBegin, tLastIter);
  auto tLastName = helper::getOptionalName(op, 0);
  if (tLastName)
    tLast->setAttr("name", builder->getStringAttr(tLastName.value()));

  SmallVector<Value> placeholderIterArgs;
  for (size_t i = 0; i < op.getIterArgs().size(); i++) {
    auto iterArg = op.getIterArgs()[i];
    auto backwardIterArg = constantX(*builder, iterArg.getType())->getResult(0);
    placeholderIterArgs.push_back(backwardIterArg);
    auto forwardIterArg = mapHIRToHWValue.lookup(iterArg);
    mapHIRToHWValue.map(op.getBody().front().getArgument(i),
                        builder->create<comb::MuxOp>(
                            uLoc, tNextBegin, forwardIterArg, backwardIterArg));
  }

  mapHIRToHWValue.map(op.getIterTimeVar(), iterTimeVar);

  auto visitResult = visitRegion(op.getBody());

  auto nextIterOp = dyn_cast<hir::NextIterOp>(bb.back());
  assert(nextIterOp);

  // Replace placeholder values.
  for (size_t i = 0; i < op.getIterArgs().size(); i++) {
    auto forwardIterArg = mapHIRToHWValue.lookup(op.getIterArgs()[i]);
    auto backwardIterArg = mapHIRToHWValue.lookup(nextIterOp.getIterArgs()[i]);
    placeholderIterArgs[i].replaceAllUsesWith(backwardIterArg);
    auto iterResult = builder->create<comb::MuxOp>(
        uLoc, tLastBegin, forwardIterArg, backwardIterArg);
    mapHIRToHWValue.map(op.getIterResults()[i], iterResult);
  }
  mapHIRToHWValue.map(op.getTEnd(), tLast);
  tstartNextIterOp.replaceAllUsesWith(
      mapHIRToHWValue.lookup(nextIterOp.getTstart()));
  conditionNextIterOp.replaceAllUsesWith(
      mapHIRToHWValue.lookup(nextIterOp.getCondition()));

  if (failed(visitResult))
    return failure();
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::IsFirstIterOp op) {
  auto isFirstIter =
      mapHIRToHWValue.lookup(op->getParentOfType<hir::WhileOp>().getTstart());
  mapHIRToHWValue.map(op.getRes(), isFirstIter);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::NextIterOp op) {
  assert(op.getOffset() == 0);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::ProbeOp op) {
  auto input = mapHIRToHWValue.lookup(op.getInput());
  assert(input);
  auto wire = builder
                  ->create<sv::WireOp>(builder->getUnknownLoc(),
                                       input.getType(), op.getVerilogNameAttr())
                  .getResult();
  builder->create<sv::AssignOp>(builder->getUnknownLoc(), wire, input);
  builder->create<sv::VerbatimOp>(builder->getUnknownLoc(),
                                  builder->getStringAttr("//PROBE: {{0}}"),
                                  wire, builder->getArrayAttr({}));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusRecvOp op) {
  assert(op.getOffset() == 0);
  mapHIRToHWValue.map(op.getRes(), mapHIRToHWValue.lookup(op.getBus()));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::ReturnOp op) {
  auto funcOp = op->getParentOfType<hir::FuncOp>();

  auto portMap = getHWModulePortMap(
      *builder, op.getLoc(), funcOp.getFuncType(), funcOp.getArgNames(),
      funcOp.getResultNames().value_or(ArrayAttr()));
  auto funcArgs = funcOp.getFuncBody().front().getArguments();

  // hwOutputs are the outputs to be returned in the hw module.
  SmallVector<Value> hwOutputs;

  // Insert 'send' buses in the input args of hir.func. These buses are outputs
  // in the hw dialect.
  for (size_t i = 0; i < funcArgs.size(); i++) {
    auto modulePortInfo = portMap.getPortInfoForFuncInput(i);
    if (modulePortInfo.isOutput()) {
      hwOutputs.push_back(mapHIRToHWValue.lookup((Value)funcArgs[i]));
    }
  }

  // Insert the hir.func outputs.
  for (Value const funcResult : op.getOperands()) {
    hwOutputs.push_back(mapHIRToHWValue.lookup(funcResult));
  }

  auto *oldOutputOp = hwModuleOp.getBodyBlock()->getTerminator();
  oldOutputOp->replaceAllUsesWith(
      builder->create<hw::OutputOp>(builder->getUnknownLoc(), hwOutputs));
  oldOutputOp->erase();
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusSendOp op) {
  assert(op.getOffset() == 0);
  auto value = mapHIRToHWValue.lookup(op.getValue());
  if (!value)
    return op.emitError() << "Could not find mapped value.";
  auto placeHolderBus = mapHIRToHWValue.lookup(op.getBus());
  auto tstart = mapHIRToHWValue.lookup(op.getTstart());
  Value defaultValue;
  if (auto defaultAttr = op->getAttr("default")) {
    assert(defaultAttr.isa<IntegerAttr>());
    defaultValue = builder->create<hw::ConstantOp>(
        builder->getUnknownLoc(), defaultAttr.dyn_cast<IntegerAttr>());
  } else {
    defaultValue = constantX(*builder, value.getType())->getResult(0);
  }
  auto newBus = builder->create<comb::MuxOp>(
      builder->getUnknownLoc(), value.getType(), tstart, value, defaultValue);
  mapHIRToHWValue.replaceAllHWUses(placeHolderBus, newBus);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::FuncExternOp op) {
  auto portMap = getHWModulePortMap(*builder, op.getLoc(), op.getFuncType(),
                                    op.getArgNames(),
                                    op.getResultNames().value_or(ArrayAttr()));
  auto name = builder->getStringAttr(op.getName());
  auto verilogNameAttr = op->getAttrOfType<StringAttr>("verilogName");
  auto hwOp = builder->create<hw::HWModuleExternOp>(
      op.getLoc(), name, portMap.getPortInfoList(),
      verilogNameAttr ? verilogNameAttr.getValue() : op.getName(),
      getHWParams(op, true));
  mapNameToHWModuleOp[op.getName()] = hwOp;
  return success();
}

void HIRToHWPass::updateHIRToHWMapForFuncInputs(
    hw::HWModuleOp hwModuleOp, mlir::Block::BlockArgListType funcArgs,
    FuncToHWModulePortMap portMap) {
  for (size_t i = 0; i < funcArgs.size(); i++) {
    auto modulePortInfo = portMap.getPortInfoForFuncInput(i);
    if (modulePortInfo.isInput()) {
      auto hwArg =
          hwModuleOp.getBodyBlock()->getArgument(modulePortInfo.argNum);
      mapHIRToHWValue.map(funcArgs[i], hwArg);
    } else {
      if (funcArgs[i].getType().isa<hir::BusType>())
        mapHIRToHWValue.map(
            funcArgs[i],
            constantX(*builder, funcArgs[i].getType())->getResult(0));
      else if (funcArgs[i].getType().isa<hir::BusTensorType>())
        mapHIRToHWValue.map(funcArgs[i],
                            getConstantXArray(*builder, funcArgs[i].getType(),
                                              mapArrayToElements)
                                ->getResult(0));
      else
        assert(false);
    }
  }
}

LogicalResult HIRToHWPass::visitOp(hir::FuncOp op) {
  auto portMap = getHWModulePortMap(*builder, op.getLoc(), op.getFuncType(),
                                    op.getArgNames(),
                                    op.getResultNames().value_or(ArrayAttr()));
  auto name = builder->getStringAttr(op.getNameAttr().getValue().str());

  this->hwModuleOp = builder->create<hw::HWModuleOp>(op.getLoc(), name,
                                                     portMap.getPortInfoList());
  mapNameToHWModuleOp[op.getName()] = hwModuleOp;
  OpBuilder::InsertionGuard const guard(*this->builder);
  this->builder->setInsertionPointToStart(hwModuleOp.getBodyBlock());

  updateHIRToHWMapForFuncInputs(
      hwModuleOp, op.getFuncBody().front().getArguments(), portMap);

  this->clk = getClkFromHWModule(hwModuleOp);
  this->reset = getResetFromHWModule(hwModuleOp);
  auto visitResult = visitRegion(op.getFuncBody());

  return visitResult;
}

LogicalResult HIRToHWPass::visitOp(hir::GetClockOp op) {
  op.getResult().replaceAllUsesWith(this->clk);
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::GetResetOp op) {
  op.getResult().replaceAllUsesWith(this->reset);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorGetElementOp op) {
  auto busTensorTy = op.getTensor().getType().dyn_cast<hir::BusTensorType>();
  auto shape = busTensorTy.getShape();
  SmallVector<Value> indices;
  auto hwTensor = mapHIRToHWValue.lookup(op.getTensor());
  assert(hwTensor);
  if (!hwTensor.getType().isa<hw::ArrayType>()) {
    mapHIRToHWValue.map(op.getRes(), hwTensor);
    return success();
  }

  for (auto idx : op.getIndices()) {
    indices.push_back(idx);
  }

  auto linearIdxValue = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(busTensorTy.getNumElements())),
          helper::calcLinearIndex(indices, shape).value()));
  mapHIRToHWValue.map(
      op.getRes(), builder->create<hw::ArrayGetOp>(builder->getUnknownLoc(),
                                                   hwTensor, linearIdxValue));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorInsertElementOp op) {
  // left = hw.slice
  // right = hw.slice
  // element = hw.array_create op.element
  // hw.array_concat left, element, right
  // calc left slice.

  auto busTensorTy = op.getTensor().getType().dyn_cast<hir::BusTensorType>();
  assert(op.getElement().getType().dyn_cast<hir::BusType>().getElementType() ==
         busTensorTy.getElementType());
  auto hwTensor = mapHIRToHWValue.lookup(op.getTensor());
  if (!hwTensor.getType().isa<hw::ArrayType>()) {
    mapHIRToHWValue.map(op.getRes(), mapHIRToHWValue.lookup(op.getElement()));
    return success();
  }

  assert(busTensorTy.getNumElements() > 1);
  SmallVector<Value> indices;
  for (auto idx : op.getIndices()) {
    indices.push_back(idx);
  }
  // Note that hw dialect arranges arrays right to left - index zero goes to
  // rightmost end syntactically.
  // Strangely, the operand vectors passed to the hw op builders are printed
  // left to right. So operand[0] would actually be the last
  auto idx = helper::calcLinearIndex(indices, busTensorTy.getShape()).value();
  auto leftWidth = busTensorTy.getNumElements() - idx - 1;
  auto rightWidth = idx;

  auto leftIdx = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(busTensorTy.getNumElements())),
          idx + 1));
  auto rightIdx = builder->create<hw::ConstantOp>(
      builder->getUnknownLoc(),
      builder->getIntegerAttr(
          IntegerType::get(builder->getContext(),
                           helper::clog2(busTensorTy.getNumElements())),
          0));

  auto leftSliceTy = hw::ArrayType::get(
      *helper::convertToHWType(busTensorTy.getElementType()), leftWidth);
  auto rightSliceTy = hw::ArrayType::get(
      *helper::convertToHWType(busTensorTy.getElementType()), rightWidth);

  auto leftSlice = builder->create<hw::ArraySliceOp>(
      builder->getUnknownLoc(), leftSliceTy, hwTensor, leftIdx);
  auto rightSlice = builder->create<hw::ArraySliceOp>(
      builder->getUnknownLoc(), rightSliceTy, hwTensor, rightIdx);
  auto hwElement = mapHIRToHWValue.lookup(op.getElement());
  if (*helper::convertToHWType(busTensorTy.getElementType()) !=
      hwElement.getType()) {
    return op.emitError() << "Incompatible hw types "
                          << *helper::convertToHWType(
                                 busTensorTy.getElementType())
                          << " and " << hwElement.getType();
  }
  auto elementAsArray =
      builder->create<hw::ArrayCreateOp>(builder->getUnknownLoc(), hwElement);

  assert(leftSlice.getType().isa<hw::ArrayType>());
  assert(rightSlice.getType().isa<hw::ArrayType>());
  assert(elementAsArray.getType().isa<hw::ArrayType>());
  assert(helper::getElementType(leftSliceTy) ==
         helper::getElementType(elementAsArray.getType()));
  assert(helper::getElementType(rightSlice.getType()) ==
         helper::getElementType(elementAsArray.getType()));
  mapHIRToHWValue.map(
      op.getRes(),
      builder->create<hw::ArrayConcatOp>(
          builder->getUnknownLoc(),
          SmallVector<Value>({leftSlice, elementAsArray, rightSlice})));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusAssignOp op) {
  auto hwDest = mapHIRToHWValue.lookup(op.getDest());
  assert(isa<sv::ConstantXOp>(hwDest.getDefiningOp()));
  auto hwSrc = mapHIRToHWValue.lookup(op.getSrc());
  mapHIRToHWValue.replaceAllHWUses(hwDest, hwSrc);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusTensorAssignOp op) {
  // dest is predefined using a placeholder.
  mapHIRToHWValue.replaceAllHWUses(mapHIRToHWValue.lookup(op.getDest()),
                                   mapHIRToHWValue.lookup(op.getSrc()));
  return success();
}
LogicalResult HIRToHWPass::visitOp(hir::BusTensorAssignElementOp op) {
  auto hwArray = mapHIRToHWValue.lookup(op.getTensor());
  auto hwElement = mapHIRToHWValue.lookup(op.getBus());
  if (!hwArray.getType().isa<hw::ArrayType>()) {
    mapHIRToHWValue.replaceAllHWUses(hwArray, hwElement);
    mapHIRToHWValue.map(op.getTensor(), hwElement);
    return success();
  }
  auto arrayElements = mapArrayToElements[hwArray];
  SmallVector<Value> indices;
  for (auto idx : op.getIndices()) {
    indices.push_back(idx);
  }
  auto idx =
      helper::calcLinearIndex(
          indices,
          op.getTensor().getType().dyn_cast<hir::BusTensorType>().getShape())
          .value();
  arrayElements[arrayElements.size() - idx - 1].replaceAllUsesWith(hwElement);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusBroadcastOp op) {
  auto busTensorTy = op.getRes().getType().dyn_cast<hir::BusTensorType>();
  auto hwBus = mapHIRToHWValue.lookup(op.getBus());
  if (busTensorTy.getNumElements() == 1) {
    mapHIRToHWValue.map(op.getRes(), hwBus);
    return success();
  }

  SmallVector<Value> replicatedArray;
  for (size_t i = 0; i < busTensorTy.getNumElements(); i++) {
    replicatedArray.push_back(hwBus);
  }

  mapHIRToHWValue.map(
      op.getRes(), builder->create<hw::ArrayCreateOp>(builder->getUnknownLoc(),
                                                      replicatedArray));
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::DelayOp op) {
  auto input = mapHIRToHWValue.lookup(op.getInput());
  auto name = helper::getOptionalName(op, 0);
  mapHIRToHWValue.map(op.getRes(),
                      getDelayedValue(*builder, input, op.getDelay(), name,
                                      op.getLoc(), clk, reset));
  return success();
}

LogicalResult HIRToHWPass::visitHWOp(Operation *operation) {
  auto operandMap = mapHIRToHWValue.getBlockAndValueMapping();
  auto *clonedOperation = builder->clone(*operation, operandMap);
  for (size_t i = 0; i < operation->getNumResults(); i++) {
    mapHIRToHWValue.map(operation->getResult(i), clonedOperation->getResult(i));
  }
  return success();
}

LogicalResult HIRToHWPass::visitRegion(mlir::Region &region) {
  Block &bb = *region.begin();
  for (Operation &operation : bb)
    if (failed(visitOperation(&operation)))
      return failure();
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::WireOp op) {
  auto z = builder->create<sv::ConstantZOp>(
      builder->getUnknownLoc(),
      op.getResult().getType().dyn_cast<hir::WireType>().getElementType());
  mapHIRToHWValue.map(op.getRes(), z);
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::DriveOp op) {
  auto wire = mapHIRToHWValue.lookup(op.getWire());
  auto value = mapHIRToHWValue.lookup(op.getValue());
  mapHIRToHWValue.replaceAllHWUses(wire, value);
  return success();
}

LogicalResult HIRToHWPass::visitOperation(Operation *operation) {
  if (auto op = dyn_cast<hir::BusMapOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusBroadcastOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorMapOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CommentOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::arith::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CallOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::DelayOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::GetClockOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::GetResetOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::TimeOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::WhileOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::IsFirstIterOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusSendOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusRecvOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorGetElementOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorInsertElementOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusAssignOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorAssignOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusTensorAssignElementOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ReturnOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::NextIterOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CastOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::ProbeOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::WireOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::DriveOp>(operation))
    return visitOp(op);
  if (auto *dialect = operation->getDialect();
      isa<comb::CombDialect, hw::HWDialect, sv::SVDialect>(dialect))
    return visitHWOp(operation);

  return operation->emitError() << "Unsupported operation for hir-to-hw pass.";
}

void HIRToHWPass::runOnOperation() {
  this->mlirModuleOp = getOperation();
  this->builder = OpBuilder(mlirModuleOp.getLoc().getContext());
  this->builder->setInsertionPointToStart(mlirModuleOp.getBody(0));
  WalkResult const result =
      mlirModuleOp.walk([this](Operation *operation) -> WalkResult {
        if (auto op = dyn_cast<hir::FuncOp>(operation)) {
          if (failed(visitOp(op)))
            return WalkResult::interrupt();
        } else if (auto op = dyn_cast<hir::FuncExternOp>(operation)) {
          if (failed(visitOp(op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // erase unnecessary ops.
  SmallVector<Operation *> opsToErase;
  for (auto &operation : getOperation()) {
    if (!isa<hw::HWDialect, sv::SVDialect, comb::CombDialect>(
            operation.getDialect()))
      opsToErase.push_back(&operation);
  }
  helper::eraseOps(opsToErase);
}

/// hir-to-hw pass Constructor
std::unique_ptr<mlir::Pass> circt::createHIRToHWPass() {
  return std::make_unique<HIRToHWPass>();
}
