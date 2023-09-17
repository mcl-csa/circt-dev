#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
using namespace circt;
using namespace hir;

void BusMapOp::build(
    OpBuilder &builder, OperationState &result, ArrayRef<Value> operands,
    std::function<hir::YieldOp(OpBuilder &, ArrayRef<Value>)> bodyCtor) {
  OpBuilder::InsertionGuard guard(builder);

  auto *bb = builder.createBlock(result.addRegion());
  SmallVector<Value> blockArgs;
  for (auto operand : operands) {
    assert(operand.getType().isa<hir::BusType>());
    blockArgs.push_back(bb->addArgument(
        operand.getType().dyn_cast<hir::BusType>().getElementType(),
        builder.getUnknownLoc()));
  }

  hir::YieldOp terminator = bodyCtor(builder, blockArgs);
  auto results = terminator.getOperands();
  SmallVector<Type> resultTypes;
  for (auto res : results) {
    resultTypes.push_back(
        hir::BusType::get(builder.getContext(), res.getType()));
  }
  result.addOperands(operands);
  result.addTypes(resultTypes);
}

void BusTensorMapOp::build(
    OpBuilder &builder, OperationState &result, ArrayRef<Value> operands,
    std::function<hir::YieldOp(OpBuilder &, ArrayRef<Value>)> bodyCtor) {

  OpBuilder::InsertionGuard guard(builder);
  auto *bb = builder.createBlock(result.addRegion());
  SmallVector<Value> blockArgs;
  for (auto operand : operands) {
    assert(operand.getType().isa<hir::BusTensorType>());
    blockArgs.push_back(bb->addArgument(
        operand.getType().dyn_cast<hir::BusTensorType>().getElementType(),
        builder.getUnknownLoc()));
  }

  hir::YieldOp terminator = bodyCtor(builder, blockArgs);
  auto results = terminator.getOperands();
  SmallVector<Type> resultTypes;
  auto shape = operands[0].getType().dyn_cast<hir::BusTensorType>().getShape();
  for (auto res : results) {
    resultTypes.push_back(
        hir::BusTensorType::get(builder.getContext(), shape, res.getType()));
  }
  result.addOperands(operands);
  result.addTypes(resultTypes);
}

void ForOp::build(
    OpBuilder &builder, OperationState &result, Value lb, Value ub, Value step,
    ArrayRef<Value> iterOperands, Value tstart, IntegerAttr offset,
    std::function<hir::NextIterOp(OpBuilder &, Value, ArrayRef<Value>, Value)>
        bodyCtor) {
  result.addOperands(lb);
  result.addOperands(ub);
  result.addOperands(step);
  if (iterOperands.size() > 0) {
    result.addOperands(iterOperands);
    SmallVector<int64_t> delays(iterOperands.size(), 0);
    result.addAttribute("iter_arg_delays", builder.getI64ArrayAttr(delays));
  }

  if (tstart)
    result.addOperands(tstart);
  if (offset)
    result.addAttribute(getOffsetAttrName(result.name), offset);

  OpBuilder::InsertionGuard guard(builder);
  auto *bb = builder.createBlock(result.addRegion());
  SmallVector<Value> iterArgs;
  for (auto v : iterOperands) {
    iterArgs.push_back(bb->addArgument(v.getType(), builder.getUnknownLoc()));
  }
  Value iv = bb->addArgument(lb.getType(), builder.getUnknownLoc());
  auto hirTimeTy = hir::TimeType::get(lb.getContext());
  Value tRegion = bb->addArgument(hirTimeTy, builder.getUnknownLoc());

  bodyCtor(builder, iv, iterArgs, tRegion);
  SmallVector<Type> resultTypes;
  for (auto iterArg : iterArgs) {
    resultTypes.push_back(iterArg.getType());
  }
  resultTypes.push_back(hirTimeTy);
  result.addTypes(resultTypes);
}

void BusOp::build(OpBuilder &builder, OperationState &result, Type resTy) {
  assert(resTy.isa<hir::BusType>());
  result.addTypes(resTy);
}

void hir::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringRef symName, hir::FuncType funcTy,
                        ArrayAttr argNames, ArrayAttr resultNames) {

  if (!argNames) {
    SmallVector<Attribute> argNamesVec;
    for (size_t i = 0; i < funcTy.getInputTypes().size(); i++) {
      argNamesVec.push_back(builder.getStringAttr("arg" + std::to_string(i)));
    }
    argNamesVec.push_back(builder.getStringAttr("t"));
    argNames = ArrayAttr::get(builder.getContext(), argNamesVec);
  }
  if (!resultNames && funcTy.getResultTypes().size() > 0) {
    SmallVector<Attribute> resultNamesVec;
    for (size_t i = 0; i < funcTy.getResultTypes().size(); i++) {
      resultNamesVec.push_back(builder.getStringAttr("r" + std::to_string(i)));
    }
    resultNames = ArrayAttr::get(builder.getContext(), resultNamesVec);
  }
  SmallVector<Attribute> functionArgAttrs;
  SmallVector<Attribute> functionResultAttrs;

  // Insert the arg_attrs and res_attrs.
  // arg_attrs also contains the attr for %t.
  auto funcArgAttrs = funcTy.getInputAttrs();
  for (auto attr : funcArgAttrs)
    functionArgAttrs.push_back(attr);
  functionArgAttrs.push_back(DictionaryAttr::get(
      builder.getContext(), SmallVector<NamedAttribute>({})));

  auto funcResultAttrs = funcTy.getResultAttrs();
  for (auto attr : funcResultAttrs)
    functionResultAttrs.push_back(attr);

  assert(argNames.size() == funcTy.getInputTypes().size() + 1);
  auto functionTy = funcTy.getFunctionType();
  FuncOp::build(builder, result, functionTy, symName, funcTy, argNames,
                resultNames, builder.getArrayAttr(functionArgAttrs),
                builder.getArrayAttr(functionResultAttrs));
  auto &region = result.regions[0];
  auto *bb = new Block();
  bb->addArguments(functionTy.getInputs(),
                   SmallVector<Location>(functionTy.getNumInputs(),
                                         builder.getUnknownLoc()));
  region->push_back(bb);
}

void hir::FuncExternOp::build(OpBuilder &builder, OperationState &result,
                              StringRef symName, hir::FuncType funcTy,
                              ArrayAttr argNames, ArrayAttr resultNames) {

  if (!argNames) {
    SmallVector<Attribute> argNamesVec;
    for (size_t i = 0; i < funcTy.getInputTypes().size(); i++) {
      argNamesVec.push_back(builder.getStringAttr("arg" + std::to_string(i)));
    }
    argNamesVec.push_back(builder.getStringAttr("t"));
    argNames = ArrayAttr::get(builder.getContext(), argNamesVec);
  }
  if (!resultNames && funcTy.getResultTypes().size() > 0) {
    SmallVector<Attribute> resultNamesVec;
    for (size_t i = 0; i < funcTy.getResultTypes().size(); i++) {
      resultNamesVec.push_back(builder.getStringAttr("r" + std::to_string(i)));
    }
    resultNames = ArrayAttr::get(builder.getContext(), resultNamesVec);
  }
  SmallVector<Attribute> functionArgAttrs;
  SmallVector<Attribute> functionResultAttrs;

  // Insert the arg_attrs and res_attrs.
  // arg_attrs also contains the attr for %t.
  auto funcArgAttrs = funcTy.getInputAttrs();
  for (auto attr : funcArgAttrs)
    functionArgAttrs.push_back(attr);
  functionArgAttrs.push_back(DictionaryAttr::get(
      builder.getContext(), SmallVector<NamedAttribute>({})));
  result.addAttribute("arg_attrs",
                      ArrayAttr::get(builder.getContext(), functionArgAttrs));

  auto funcResultAttrs = funcTy.getResultAttrs();
  for (auto attr : funcResultAttrs)
    functionResultAttrs.push_back(attr);

  result.addAttribute(
      "res_attrs", ArrayAttr::get(builder.getContext(), functionResultAttrs));
  assert(argNames.size() == funcTy.getInputTypes().size() + 1);

  auto functionTy = funcTy.getFunctionType();
  FuncExternOp::build(builder, result, functionTy, symName, funcTy, argNames,
                      resultNames);
  OpBuilder::InsertionGuard guard(builder);
  auto *bb = builder.createBlock(&*result.regions[0]);
  bb->addArguments(functionTy.getInputs(),
                   SmallVector<Location>(functionTy.getNumInputs(),
                                         builder.getUnknownLoc()));

  builder.create<hir::ReturnOp>(builder.getUnknownLoc(),
                                SmallVector<Value>({}));
}
