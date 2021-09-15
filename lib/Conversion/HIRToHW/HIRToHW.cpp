//===----- HIRToHW.cpp - HIR To HW Conversion Pass-------*-C++-*-===//
//
// This pass converts HIR to HW, Comb and SV dialect.
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "HIRToHWUtils.h"
#include "circt/Conversion/HIRToHW/HIRToHW.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iostream>
using namespace circt;

class HIRToHWPass : public HIRToHWBase<HIRToHWPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult visitRegion(mlir::Region &);
  LogicalResult visitOperation(Operation *);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(mlir::ConstantOp);
  LogicalResult visitOp(hir::BusOp);
  LogicalResult visitOp(hir::CommentOp);
  LogicalResult visitOp(hir::CallOp);

private:
  OpBuilder *builder;
  llvm::DenseMap<Value, Value> mapHIRToHWValue;
  llvm::DenseMap<StringRef, uint64_t> mapFuncNameToInstanceCount;
};

LogicalResult HIRToHWPass::visitOp(mlir::ConstantOp op) {
  if (op.getType().isa<mlir::IndexType>())
    return success();
  if (!op.getType().isa<mlir::IntegerType>())
    return op.emitError(
        "hir-to-hw pass only supports IntegerType/IndexType constants.");

  builder->create<hw::ConstantOp>(op.getLoc(),
                                  op.value().dyn_cast<IntegerAttr>());
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::BusOp op) {
  // Add a placeholder SSA Var for the buses. CallOp visit will replace them.
  // We need to do this because HW dialect does not have SSA dominance.
  auto *constantXOp = getConstantX(builder, op.getType());
  auto placeHolderSSAVar = constantXOp->getResult(0);
  auto name = helper::getOptionalName(constantXOp, 0);
  if (name)
    helper::setNames(constantXOp, {name.getValue()});
  mapHIRToHWValue[op.res()] = placeHolderSSAVar;
  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::CommentOp) { return success(); }

LogicalResult HIRToHWPass::visitOp(hir::CallOp op) {
  auto filteredOperands = filterCallOpArgs(op.getFuncType(), op.operands());

  // Get the mapped inputs and create the input types for instance op.
  SmallVector<Value> hwInputs;
  for (auto input : filteredOperands.first) {
    auto hwInput = mapHIRToHWValue[input];
    // FIXME: Once all the op visitors are written, replace this with
    // assert(hwInput).
    if (!hwInput)
      return success();
    hwInputs.push_back(hwInput);
  }
  auto hwInputTypes = helper::getTypes(hwInputs);

  auto sendBuses = filteredOperands.second;
  auto sendBusTypes = helper::getTypes(sendBuses);

  // Create instance op result types.
  SmallVector<Type> hwResultTypes;
  for (auto ty : sendBusTypes)
    hwResultTypes.push_back(convertType(ty));

  for (auto ty : op.getResultTypes())
    hwResultTypes.push_back(convertType(ty));

  auto instanceName = builder->getStringAttr(
      op.callee().str() + "_" +
      std::to_string(mapFuncNameToInstanceCount[op.callee()]++));

  auto instanceOp = builder->create<hw::InstanceOp>(
      op.getLoc(), hwResultTypes, instanceName, op.calleeAttr(), hwInputs,
      op->getAttr("params").dyn_cast_or_null<DictionaryAttr>(), StringAttr());

  uint64_t i;

  // Replace the placeholder HW SSA var (for HIR buses) with the results of the
  // instance op.
  for (i = 0; i < sendBuses.size(); i++) {
    auto placeHolderSSAVar = mapHIRToHWValue[sendBuses[i]];
    assert(placeHolderSSAVar);
    placeHolderSSAVar.replaceAllUsesWith(instanceOp.getResult(i));
  }

  // Map the HIR SSA vars to HW SSA vars.
  for (uint64_t j = 0; i + j < instanceOp.getNumResults(); j++)
    mapHIRToHWValue[op.getResult(j)] = instanceOp.getResult(i + j);

  return success();
}

LogicalResult HIRToHWPass::visitOp(hir::FuncOp op) {
  builder = new OpBuilder(op);
  builder->setInsertionPoint(op);
  auto portInfoList = getHWModulePortInfoList(
      *builder, op.getFuncType(), op.getInputNames(), op.getResultNames());
  auto name = builder->getStringAttr("hw_" + op.getNameAttr().getValue().str());
  auto hwModuleOp =
      builder->create<hw::HWModuleOp>(op.getLoc(), name, portInfoList);
  builder->setInsertionPointToStart(hwModuleOp.getBodyBlock());
  auto visitResult = visitRegion(op.getFuncBody());
  delete (builder);
  return visitResult;
}

LogicalResult HIRToHWPass::visitRegion(mlir::Region &region) {
  Block &bb = *region.begin();
  for (Operation &operation : bb)
    if (failed(visitOperation(&operation)))
      return failure();
  return success();
}

LogicalResult HIRToHWPass::visitOperation(Operation *operation) {
  if (auto op = dyn_cast<hir::FuncOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<mlir::ConstantOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::BusOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CommentOp>(operation))
    return visitOp(op);
  if (auto op = dyn_cast<hir::CallOp>(operation))
    return visitOp(op);

  // operation->emitRemark() << "Unsupported operation for hir-to-hw pass.";
  return success();
}

void HIRToHWPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  WalkResult result = moduleOp.walk([this](Operation *operation) -> WalkResult {
    if (auto op = dyn_cast<hir::FuncOp>(operation)) {
      if (failed(visitOp(op)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

/// hir-to-hw pass Constructor
std::unique_ptr<mlir::Pass> circt::createHIRToHWPass() {
  return std::make_unique<HIRToHWPass>();
}
