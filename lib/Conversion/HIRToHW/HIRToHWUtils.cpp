#include "HIRToHWUtils.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

void FuncToHWModulePortMap::addFuncInput(StringAttr name,
                                         hw::PortInfo::Direction direction,
                                         Type type) {
  assert(name);
  assert(type);
  assert((direction == hw::PortInfo::Direction::Input) ||
         (direction == hw::PortInfo::Direction::Output));

  size_t const argNum = (direction == hw::PortInfo::Direction::Input)
                            ? (hwModuleInputArgNum++)
                            : (hwModuleResultArgNum++);

  hw::PortInfo portInfo = hw::PortInfo{{name, type, direction}, argNum};
  portInfoList.push_back(portInfo);

  mapFuncInputToHWPortInfo.push_back(portInfo);
}

void FuncToHWModulePortMap::addClk(OpBuilder &builder) {
  auto clkName = builder.getStringAttr("clk");

  hw::PortInfo portInfo = hw::PortInfo{
      {clkName, builder.getI1Type(), hw::PortInfo::Direction::Input},
      hwModuleInputArgNum++};
  portInfoList.push_back(portInfo);
}

void FuncToHWModulePortMap::addReset(OpBuilder &builder) {
  auto resetName = builder.getStringAttr("rst");
  hw::PortInfo portInfo = hw::PortInfo{
      {resetName, builder.getI1Type(), hw::PortInfo::Direction::Input},
      hwModuleInputArgNum++};
  portInfoList.push_back(portInfo);
}

void FuncToHWModulePortMap::addFuncResult(StringAttr name, Type type) {
  assert(name);
  assert(type);
  hw::PortInfo portInfo = hw::PortInfo{
      {name, type, hw::PortInfo::Direction::Output}, hwModuleResultArgNum++};
  portInfoList.push_back(portInfo);
}

bool isSendBus(DictionaryAttr busAttr) {
  return helper::extractBusPortFromDict(busAttr) == "send";
}

ArrayRef<hw::PortInfo> FuncToHWModulePortMap::getPortInfoList() {
  return portInfoList;
}

hw::PortInfo
FuncToHWModulePortMap::getPortInfoForFuncInput(size_t inputArgNum) {
  auto modulePortInfo = mapFuncInputToHWPortInfo[inputArgNum];
  assert(!modulePortInfo.isInOut());
  return modulePortInfo;
}

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, SmallVector<Value, 4> args) {
  SmallVector<Value> inputs;
  SmallVector<Value> results;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    if (helper::isBusLikeType(ty) && isSendBus(funcTy.getInputAttrs()[i])) {
      results.push_back(args[i]);
      continue;
    }
    inputs.push_back(args[i]);
  }

  return std::make_pair(inputs, results);
}

FuncToHWModulePortMap getHWModulePortMap(OpBuilder &builder,
                                         mlir::Location errorLoc,
                                         hir::FuncType funcTy,
                                         ArrayAttr inputNames,
                                         ArrayAttr resultNames) {
  FuncToHWModulePortMap portMap;

  // filter the input and output types and names.
  uint64_t i;
  for (i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto originalTy = funcTy.getInputTypes()[i];
    auto hwTy = helper::convertToHWType(originalTy);
    if (!hwTy)
      emitError(errorLoc) << "Type " << originalTy
                          << "could not be converted to a compatible hw type.";
    auto attr = funcTy.getInputAttrs()[i];
    auto name = inputNames[i].dyn_cast<StringAttr>();
    if (helper::isBusLikeType(originalTy) && isSendBus(attr)) {
      portMap.addFuncInput(name, hw::PortInfo::Direction::Output, *hwTy);
    } else {
      portMap.addFuncInput(name, hw::PortInfo::Direction::Input, *hwTy);
    }
  }

  // Add time input arg.
  auto timeVarName = inputNames[i].dyn_cast<StringAttr>();
  portMap.addFuncInput(timeVarName, hw::PortInfo::Direction::Input,
                       builder.getI1Type());

  // Add clk input arg.
  portMap.addClk(builder);
  // Add reset input arg.
  portMap.addReset(builder);

  // Add hir.func results.
  for (uint64_t i = 0; i < funcTy.getResultTypes().size(); i++) {
    auto hwTy = *helper::convertToHWType(funcTy.getResultTypes()[i]);
    auto name = resultNames[i].dyn_cast<StringAttr>();
    portMap.addFuncResult(name, hwTy);
  }

  return portMap;
}

void copyHIRAttrs(hir::CallOp srcOp, hw::InstanceOp destOp) {
  if (destOp->hasAttr("hir_attrs"))
    return;

  if (auto attr = srcOp->getAttr("hir_attrs"))
    destOp->setAttr("hir_attrs", attr);
}

Operation *
getConstantXArray(OpBuilder &builder, Type hirTy,
                  DenseMap<Value, SmallVector<Value>> &mapArrayToElements) {
  assert(hirTy.isa<hir::BusTensorType>());
  auto hwTy = *helper::convertToHWType(hirTy);
  auto hwArrayTy = hwTy.dyn_cast<hw::ArrayType>();
  // If its a bus_tensor of size 1 then do not update mapArrayToElements;
  if (!hwArrayTy) {
    return constantX(builder, hwTy);
  }
  SmallVector<Value> constXCopies;
  for (uint64_t i = 0; i < hwArrayTy.getSize(); i++) {
    Value const constXValue =
        constantX(builder, hir::BusType::get(builder.getContext(),
                                             hwArrayTy.getElementType()))
            ->getResult(0);
    constXCopies.push_back(constXValue);
  }
  auto arrOp =
      builder.create<hw::ArrayCreateOp>(builder.getUnknownLoc(), constXCopies);
  mapArrayToElements[arrOp.getResult()] = constXCopies;
  return arrOp;
}

Operation *constantX(OpBuilder &builder, Type hirTy) {
  auto hwTy = *helper::convertToHWType(hirTy);
  assert(hwTy.isa<IntegerType>());
  return builder.create<sv::ConstantXOp>(builder.getUnknownLoc(), hwTy);
}

ArrayAttr getHWParams(Attribute paramsAttr, bool ignoreValues) {
  if (!paramsAttr)
    return ArrayAttr();

  auto params = paramsAttr.dyn_cast<DictionaryAttr>();
  assert(params);

  Builder builder(params.getContext());
  SmallVector<Attribute> hwParams;
  for (const NamedAttribute &param : params) {
    auto name = builder.getStringAttr(param.getName().strref());
    hw::ParamDeclAttr hwParam;
    mlir::TypedAttr value = dyn_cast<mlir::TypedAttr>(param.getValue());
    if (ignoreValues || !value)
      hwParam = hw::ParamDeclAttr::get(
          name, mlir::NoneType::get(builder.getContext()));
    else
      hwParam = hw::ParamDeclAttr::get(name, value);
    hwParams.push_back(hwParam);
  }
  return ArrayAttr::get(builder.getContext(), hwParams);
}

Value getDelayedValue(OpBuilder &builder, Value input, int64_t delay,
                      std::optional<StringRef> name, Location loc, Value clk,
                      Value reset) {
  assert(input.getType().isa<mlir::IntegerType>() ||
         input.getType().isa<hw::ArrayType>());
  auto nameAttr = name ? builder.getStringAttr(name.value()) : StringAttr();

  Type regTy;
  if (delay > 1)
    regTy = hw::ArrayType::get(input.getType(), delay);
  else
    regTy = input.getType();

  auto reg =
      builder.create<sv::RegOp>(builder.getUnknownLoc(), regTy, nameAttr);

  auto regOutput =
      builder.create<sv::ReadInOutOp>(builder.getUnknownLoc(), reg);

  Value regInput;
  if (delay > 1) {
    auto c0 =
        helper::materializeIntegerConstant(builder, 0, helper::clog2(delay));
    auto sliceTy = hw::ArrayType::get(input.getType(), delay - 1);
    auto regSlice = builder.create<hw::ArraySliceOp>(builder.getUnknownLoc(),
                                                     sliceTy, regOutput, c0);
    regInput = builder.create<hw::ArrayConcatOp>(
        builder.getUnknownLoc(),
        ArrayRef<Value>({regSlice, builder.create<hw::ArrayCreateOp>(
                                       builder.getUnknownLoc(), input)}));
  } else {
    regInput = input;
  }
  auto bodyCtor = [&builder, &reg, &regInput] {
    builder.create<sv::PAssignOp>(builder.getUnknownLoc(), reg.getResult(),
                                  regInput);
  };
  Value regResetValue;
  auto zeroBit = builder.create<hw::ConstantOp>(
      builder.getUnknownLoc(), IntegerAttr::get(input.getType(), 0));
  if (delay > 1)
    regResetValue = builder.create<hw::ArrayCreateOp>(
        builder.getUnknownLoc(), SmallVector<Value>(delay, zeroBit));
  else
    regResetValue = zeroBit;

  auto resetCtor = [&builder, &reg, &regResetValue] {
    builder.create<sv::PAssignOp>(builder.getUnknownLoc(), reg.getResult(),
                                  regResetValue);
  };

  builder.create<sv::AlwaysFFOp>(
      builder.getUnknownLoc(), sv::EventControl::AtPosEdge, clk,
      ResetType::SyncReset, sv::EventControl::AtPosEdge, reset, bodyCtor,
      resetCtor);

  Value output;
  if (delay > 1) {
    auto cEnd = helper::materializeIntegerConstant(builder, delay - 1,
                                                   helper::clog2(delay));
    output = builder.create<hw::ArrayGetOp>(loc, regOutput, cEnd).getResult();
  } else {
    output = regOutput;
  }

  assert(input.getType() == output.getType());
  return output;
}

Value convertToNamedValue(OpBuilder &builder, StringRef name, Value val) {
  assert(val.getType().isa<mlir::IntegerType>() ||
         val.getType().isa<hw::ArrayType>());
  auto wire = builder.create<sv::WireOp>(builder.getUnknownLoc(), val.getType(),
                                         builder.getStringAttr(name));
  builder.create<sv::AssignOp>(builder.getUnknownLoc(), wire, val);
  return builder.create<sv::ReadInOutOp>(builder.getUnknownLoc(), wire);
}

Value convertToOptionalNamedValue(OpBuilder &builder,
                                  std::optional<StringRef> name, Value val) {

  assert(val.getType().isa<mlir::IntegerType>() ||
         val.getType().isa<hw::ArrayType>());
  if (name) {
    return convertToNamedValue(builder, name.value(), val);
  }
  return val;
}

SmallVector<Value> insertBusMapLogic(OpBuilder &builder, Block &bodyBlock,
                                     ArrayRef<Value> operands) {
  IRMapping operandMap;
  SmallVector<Value> results;
  for (size_t i = 0; i < operands.size(); i++) {
    operandMap.map(bodyBlock.getArgument(i), operands[i]);
  }
  for (auto &operation : bodyBlock) {
    if (auto yieldOp = dyn_cast<hir::YieldOp>(operation)) {
      for (size_t i = 0; i < yieldOp.getNumOperands(); i++) {
        results.push_back(operandMap.lookup(yieldOp.getOperand(i)));
      }
    } else
      builder.clone(operation, operandMap);
  }
  return results;
}

Value insertConstArrayGetLogic(OpBuilder &builder, Value arr, int idx) {
  auto uLoc = builder.getUnknownLoc();
  auto arrayTy = arr.getType().dyn_cast<hw::ArrayType>();
  assert(arrayTy);
  assert(arrayTy.getSize() > 1);
  auto cIdx = builder.create<hw::ConstantOp>(
      uLoc, IntegerAttr::get(
                builder.getIntegerType(helper::clog2(arrayTy.getSize())), idx));
  return builder.create<hw::ArrayGetOp>(uLoc, arr, cIdx);
}

Value getClkFromHWModule(hw::HWModuleOp op) {
  auto idxClk = op.getBodyBlock()->getNumArguments() - 2;
  return op.getBodyBlock()->getArguments()[idxClk];
}

Value getResetFromHWModule(hw::HWModuleOp op) {
  auto idxReset = op.getBodyBlock()->getNumArguments() - 1;
  return op.getBodyBlock()->getArguments()[idxReset];
}
