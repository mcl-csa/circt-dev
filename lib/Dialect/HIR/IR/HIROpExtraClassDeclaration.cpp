//=========- HIROpExtraClassDecl.cpp - extraClassDeclarations for Ops -----===//
//
// This file implements the extraClassDeclarations for HIR ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace circt;
using namespace hir;

namespace {
SmallVector<Value> filterIndices(DimKind idxKind, OperandRange indices,
                                 ArrayRef<DimKind> dimKinds) {
  SmallVector<Value> addrIndices;
  for (size_t i = 0; i < indices.size(); i++) {
    if (dimKinds[i] == idxKind) {
      auto idx = indices[i];
      addrIndices.push_back(idx);
    }
  }
  return addrIndices;
}
} // namespace
SmallVector<Value> LoadOp::filterIndices(DimKind idxKind) {

  OperandRange indices = this->getIndices();
  auto dimKinds =
      this->getMem().getType().dyn_cast<hir::MemrefType>().getDimKinds();
  return ::filterIndices(idxKind, indices, dimKinds);
}

SmallVector<Value> StoreOp::filterIndices(DimKind idxKind) {

  OperandRange indices = this->getIndices();
  auto dimKinds =
      this->getMem().getType().dyn_cast<hir::MemrefType>().getDimKinds();
  return ::filterIndices(idxKind, indices, dimKinds);
}

SmallVector<Value, 4> hir::FuncOp::getOperands() {
  SmallVector<Value, 4> operands;

  auto &entryBlock = this->getFuncBody().front();
  for (Value arg :
       entryBlock.getArguments().slice(0, entryBlock.getNumArguments() - 1))
    operands.push_back(arg);
  return operands;
}

hir::FuncType hir::FuncOp::getFuncType() {
  return getFuncTy().dyn_cast<hir::FuncType>();
}

ArrayRef<Type> hir::FuncOp::getArgumentTypes() {
  return this->getFunctionType().dyn_cast<mlir::FunctionType>().getInputs();
}

ArrayRef<Type> hir::FuncOp::getResultTypes() {
  return this->getFunctionType().dyn_cast<mlir::FunctionType>().getResults();
}

// unsigned hir::FuncExternOp::getNumFunctionArguments() {
//  return this->type().dyn_cast<mlir::FunctionType>().getNumInputs();
//}
//
// unsigned hir::FuncExternOp::getNumFunctionResults() {
//  return this->type().dyn_cast<mlir::FunctionType>().getNumResults();
//}

void hir::FuncOp::updateArguments(ArrayRef<DictionaryAttr> inputAttrs) {

  auto &entryBlock = this->getFuncBody().front();
  SmallVector<Type> inputTypes;
  for (uint64_t i = 0; i < entryBlock.getNumArguments() - 1; i++) {
    auto ty = entryBlock.getArgumentTypes()[i];
    inputTypes.push_back(ty);
  }
  assert(inputTypes.size() == inputAttrs.size() ||
         succeeded(this->emitError("Mismatch in number of types and attrs")));

  auto newFuncTy =
      hir::FuncType::get(this->getContext(), inputTypes, inputAttrs,
                         this->getFuncType().getResultTypes(),
                         this->getFuncType().getResultAttrs());

  this->setFunctionTypeAttr(TypeAttr::get(newFuncTy.getFunctionType()));
  this->setFuncTyAttr(TypeAttr::get(newFuncTy));

  SmallVector<Attribute> functionArgAttrs;
  for (auto attr : inputAttrs)
    functionArgAttrs.push_back(attr);
  functionArgAttrs.push_back(
      DictionaryAttr::get(this->getContext(), ArrayRef<NamedAttribute>({})));

  this->setArgAttrsAttr(ArrayAttr::get(this->getContext(), functionArgAttrs));
}

void hir::FuncExternOp::updateArguments(ArrayRef<DictionaryAttr> inputAttrs) {
  auto &entryBlock = this->getFuncBody().front();
  SmallVector<Type> inputTypes;
  for (uint64_t i = 0; i < entryBlock.getNumArguments() - 1; i++) {
    auto ty = entryBlock.getArgumentTypes()[i];
    inputTypes.push_back(ty);
  }
  assert(inputTypes.size() == inputAttrs.size() ||
         succeeded(this->emitError("Mismatch in number of types and attrs")));

  auto newFuncTy =
      hir::FuncType::get(this->getContext(), inputTypes, inputAttrs,
                         this->getFuncType().getResultTypes(),
                         this->getFuncType().getResultAttrs());

  this->setFuncTyAttr(TypeAttr::get(newFuncTy));
}

SmallVector<Value, 4> hir::CallOp::getOperands() {
  SmallVector<Value, 4> operands;
  for (Value arg : this->getHirOperands().slice(0, this->getNumOperands() - 1))
    operands.push_back(arg);
  return operands;
}

SmallVector<Value> ForOp::getCapturedValues() {
  SmallVector<Value> capturedValues;
  mlir::visitUsedValuesDefinedAbove(
      getBody(), [&capturedValues](OpOperand *operand) {
        if (helper::isBuiltinSizedType(operand->get().getType()))
          capturedValues.push_back(operand->get());
        return;
      });
  return capturedValues;
}

Value ForOp::getIterArgOperand(unsigned int index) {
  return getOperand(index + 3);
}
void ForOp::setIterArgOperand(unsigned int index, Value v) {
  setOperand(index + 3, v);
  auto iterArgs = this->getIterArgs();
  assert(iterArgs.size() > index);
  iterArgs[index].setType(v.getType());
}

Block *ForOp::addEntryBlock(MLIRContext *context, Type inductionVarTy) {
  Block *entry = new Block;
  Builder builder(this->getContext());
  for (Value iterArg : this->getIterArgs())
    entry->addArgument(iterArg.getType(), builder.getUnknownLoc());
  entry->addArgument(inductionVarTy, builder.getUnknownLoc()); // induction var
  entry->addArgument(hir::TimeType::get(context),
                     builder.getUnknownLoc()); // iter time
  getLoopBody().push_back(entry);
  return entry;
}

Block *WhileOp::addEntryBlock() {
  auto *context = this->getContext();
  Builder builder(this->getContext());
  Block *entry = new Block;
  for (Value iterArg : this->getIterArgs())
    entry->addArgument(iterArg.getType(), builder.getUnknownLoc());
  entry->addArgument(hir::TimeType::get(context),
                     builder.getUnknownLoc()); // iter time
  getBody().push_back(entry);
  return entry;
}

SmallVector<Value> WhileOp::getCapturedValues() {
  SmallVector<Value> capturedValues;
  mlir::visitUsedValuesDefinedAbove(
      getBody(), [&capturedValues](OpOperand *operand) {
        if (helper::isBuiltinSizedType(operand->get().getType()))
          capturedValues.push_back(operand->get());
        return;
      });
  return capturedValues;
}

Operation *CallOp::getCalleeDecl() {
  auto topLevelModuleOp = (*this)->getParentOfType<mlir::ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  return topLevelModuleOp.lookupSymbol(getCallee());
}

Value ForOp::getInductionVar() {
  return getBody().getArgument(getBody().getNumArguments() - 2);
}

Value ForOp::setInductionVar(Type ty) {
  Builder builder(this->getContext());
  int ivPosition = getBody().getNumArguments() - 2;
  getBody().getArgument(ivPosition).setType(ty);
  return getInductionVar();
}

SmallVector<Value> ForOp::getIterArgArguments() {
  SmallVector<Value> iterArgs;
  for (size_t i = 0; i < getBody().getArguments().size() - 2; i++)
    iterArgs.push_back(getBody().getArgument(i));
  return iterArgs;
}

Value ForOp::getIterTimeVar() { return getBody().getArguments().back(); }
StringRef ForOp::getInductionVarName() {
  return (*this)
      ->getAttr("argNames")
      .dyn_cast<ArrayAttr>()[0]
      .dyn_cast<StringAttr>()
      .getValue();
}
StringRef ForOp::getIterTimeVarName() {
  return (*this)
      ->getAttr("argNames")
      .dyn_cast<ArrayAttr>()[1]
      .dyn_cast<StringAttr>()
      .getValue();
}
std::optional<int64_t> ForOp::getTripCount() {
  auto lb = helper::getConstantIntValue(this->getLb());
  if (!lb)
    return std::nullopt;
  auto ub = helper::getConstantIntValue(this->getUb());
  if (!ub)
    return std::nullopt;
  if (*ub <= *lb) {
    this->emitError("Upper bound should be greater than lower bound.")
        << "ub=" << *ub << "lb=" << *lb;
    assert(false && "ub>lb");
  }
  auto step = helper::getConstantIntValue(this->getStep());
  if (!step)
    return std::nullopt;
  return (*ub - *lb) / (*step);
}

// ScheduledOp interface.
hir::Time CallOp::getStartTime() { return hir::Time(getTstart(), getOffset()); }

hir::Time hir::ForOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::WhileOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::IsFirstIterOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::NextIterOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::IfOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::DelayOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::TimeOp::getStartTime() {
  return hir::Time(getTimevar(), getOffset());
}
hir::Time hir::LoadOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::StoreOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::BusSendOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}
hir::Time hir::BusRecvOp::getStartTime() {
  return hir::Time(getTstart(), getOffset());
}

void CallOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}

void ForOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void WhileOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void IsFirstIterOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void NextIterOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}

void IfOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}

void DelayOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}

void TimeOp::setStartTime(hir::Time time) {
  this->getTimevarMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void LoadOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void StoreOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void BusSendOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}
void BusRecvOp::setStartTime(hir::Time time) {
  this->getTstartMutable().assign(time.getTimeVar());
  this->setOffsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                       time.getOffset()));
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
CallOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  auto funcTy = this->getFuncTy().dyn_cast<hir::FuncType>();
  for (size_t i = 0; i < this->getNumResults(); i++) {
    Value res = this->getResult(i);
    Type resTy = res.getType();
    if (helper::isBuiltinSizedType(resTy)) {
      DictionaryAttr attrDict = funcTy.getResultAttrs()[i];
      uint64_t delay = *helper::getHIRDelayAttr(attrDict);
      Time time = this->getStartTime().addOffset(delay);
      output.push_back(std::make_pair(res, time));
    } else if (resTy.isa<TimeType>()) {
      Time time = Time(res, 0);
      output.push_back(std::make_pair(res, time));
    } else {
      llvm_unreachable("CallOp output should be a value type.");
    }
  }
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
ForOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  Value endTimeVar = this->getResults().back();
  auto tripCount = this->getTripCount();
  auto ii = this->getInitiationInterval();
  std::optional<hir::Time> endTime;
  if (tripCount && ii) {
    hir::Time startTime = this->getStartTime();
    endTime = startTime.addOffset(tripCount.value() * ii.value());
  } else {
    endTime = Time(endTimeVar, 0);
  }
  if (!endTime.has_value()) {
    output.push_back(std::make_pair(endTimeVar, endTime));
    return output;
  }

  if (this->getIterArgDelays()) {
    auto iterArgDelays = this->getIterArgDelays().value();
    for (size_t i = 0; i < iterArgDelays.size(); i++) {
      auto delay = iterArgDelays[i].dyn_cast<mlir::IntegerAttr>().getInt();
      auto result = this->getResult(i);
      output.push_back(std::make_pair(result, endTime->addOffset(delay)));
    }
  }

  output.push_back(std::make_pair(endTimeVar, endTime));
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
WhileOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  Value endTimeVar = this->getResults().back();
  if (this->getIterArgDelays()) {
    auto iterArgDelays = this->getIterArgDelays().value();
    for (size_t i = 0; i < iterArgDelays.size(); i++) {
      auto delay = iterArgDelays[i].dyn_cast<mlir::IntegerAttr>().getInt();
      auto result = this->getResult(i);
      output.push_back(std::make_pair(result, Time(endTimeVar, delay)));
    }
  }
  output.push_back(std::make_pair(endTimeVar, Time(endTimeVar, 0)));
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
IsFirstIterOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(this->getResult(), this->getStartTime()));
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
NextIterOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
IfOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  auto resultAttrs = this->getResultAttrs();
  for (size_t i = 0; i < this->getNumResults(); i++) {
    Value res = this->getResult(i);
    Type resTy = res.getType();
    if (helper::isBuiltinSizedType(resTy)) {
      uint64_t delay = resultAttrs.value()[i].dyn_cast<IntegerAttr>().getInt();
      Time time = this->getStartTime().addOffset(delay);
      output.push_back(std::make_pair(res, time));
    } else if (resTy.isa<TimeType>()) {
      Time time = Time(res, 0);
      output.push_back(std::make_pair(res, time));
    } else {
      llvm_unreachable("IfOp output should be value type.");
    }
  }
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
DelayOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(
      this->getResult(), this->getStartTime().addOffset(this->getDelay())));
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
TimeOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(this->getResult(), this->getStartTime()));
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
LoadOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(
      this->getResult(), this->getStartTime().addOffset(this->getDelay())));
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
StoreOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
BusSendOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  return output;
}

SmallVector<std::pair<Value, std::optional<hir::Time>>, 4>
BusRecvOp::getResultsWithTime() {
  SmallVector<std::pair<Value, std::optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(this->getResult(), this->getStartTime()));
  return output;
}

// ScheduledRegionOp interface.
SmallVector<Value> ForOp::getRegionTimeVars() {
  SmallVector<Value> regionTimeVars;
  regionTimeVars.push_back(this->getIterTimeVar());
  return regionTimeVars;
}

SmallVector<Value> WhileOp::getRegionTimeVars() {
  SmallVector<Value> regionTimeVars;
  regionTimeVars.push_back(this->getIterTimeVar());
  return regionTimeVars;
}

SmallVector<Value> FuncOp::getRegionTimeVars() {
  SmallVector<Value> regionTimeVars;
  regionTimeVars.push_back(this->getRegionTimeVar());
  return regionTimeVars;
}

std::optional<int64_t> FuncOp::getRegionII() {
  // FIXME: Currently we are assuming FuncOp is not pipelined.
  return INT64_MAX;
}

std::optional<int64_t> ForOp::getRegionII() { return getInitiationInterval(); }

std::optional<int64_t> WhileOp::getRegionII() { return std::nullopt; }
