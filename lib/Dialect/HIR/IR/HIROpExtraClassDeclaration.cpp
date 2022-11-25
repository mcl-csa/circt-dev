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

  OperandRange indices = this->indices();
  auto dimKinds =
      this->mem().getType().dyn_cast<hir::MemrefType>().getDimKinds();
  return ::filterIndices(idxKind, indices, dimKinds);
}

SmallVector<Value> StoreOp::filterIndices(DimKind idxKind) {

  OperandRange indices = this->indices();
  auto dimKinds =
      this->mem().getType().dyn_cast<hir::MemrefType>().getDimKinds();
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

mlir::FunctionType hir::FuncOp::getFunctionType() {
  return this->function_type().dyn_cast<mlir::FunctionType>();
}
hir::FuncType hir::FuncOp::getFuncType() {
  return funcTy().dyn_cast<hir::FuncType>();
}

ArrayRef<Type> hir::FuncOp::getArgumentTypes() {
  return this->function_type().dyn_cast<mlir::FunctionType>().getInputs();
}

ArrayRef<Type> hir::FuncOp::getResultTypes() {
  return this->function_type().dyn_cast<mlir::FunctionType>().getResults();
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

  this->function_typeAttr(TypeAttr::get(newFuncTy.getFunctionType()));
  this->funcTyAttr(TypeAttr::get(newFuncTy));

  SmallVector<Attribute> functionArgAttrs;
  for (auto attr : inputAttrs)
    functionArgAttrs.push_back(attr);
  functionArgAttrs.push_back(
      DictionaryAttr::get(this->getContext(), ArrayRef<NamedAttribute>({})));

  this->setAllArgAttrs(functionArgAttrs);
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

  this->funcTyAttr(TypeAttr::get(newFuncTy));
}

SmallVector<Value, 4> hir::CallOp::getOperands() {
  SmallVector<Value, 4> operands;
  for (Value arg : this->operands().slice(0, this->getNumOperands() - 1))
    operands.push_back(arg);
  return operands;
}

SmallVector<Value> ForOp::getCapturedValues() {
  SmallVector<Value> capturedValues;
  mlir::visitUsedValuesDefinedAbove(
      body(), [&capturedValues](OpOperand *operand) {
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
  for (Value iterArg : this->iter_args())
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
  for (Value iterArg : this->iter_args())
    entry->addArgument(iterArg.getType(), builder.getUnknownLoc());
  entry->addArgument(hir::TimeType::get(context),
                     builder.getUnknownLoc()); // iter time
  body().push_back(entry);
  return entry;
}

SmallVector<Value> WhileOp::getCapturedValues() {
  SmallVector<Value> capturedValues;
  mlir::visitUsedValuesDefinedAbove(
      body(), [&capturedValues](OpOperand *operand) {
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

  return topLevelModuleOp.lookupSymbol(callee());
}

Value ForOp::getInductionVar() {
  return getBody()->getArgument(getBody()->getNumArguments() - 2);
}

Value ForOp::setInductionVar(Type ty) {
  Builder builder(this->getContext());
  int ivPosition = getBody()->getNumArguments() - 2;
  getBody()->getArgument(ivPosition).setType(ty);
  return getInductionVar();
}

SmallVector<Value> ForOp::getIterArgs() {
  SmallVector<Value> iterArgs;
  for (size_t i = 0; i < getBody()->getArguments().size() - 2; i++)
    iterArgs.push_back(getBody()->getArgument(i));
  return iterArgs;
}

Value ForOp::getIterTimeVar() { return getBody()->getArguments().back(); }
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
Optional<int64_t> ForOp::getTripCount() {
  auto lb = helper::getConstantIntValue(this->lb());
  if (!lb.hasValue())
    return llvm::None;
  auto ub = helper::getConstantIntValue(this->ub());
  if (!ub.hasValue())
    return llvm::None;
  auto step = helper::getConstantIntValue(this->step());
  if (!step.hasValue())
    return llvm::None;
  return (*ub - *lb) / (*step);
}

Optional<int64_t> ForOp::getInitiationInterval() {
  auto nextIterOp = dyn_cast<hir::NextIterOp>(
      this->getLoopBody().getBlocks().front().getTerminator());
  auto time = nextIterOp.getStartTime();
  if (time.getTimeVar() != this->getIterTimeVar()) {
    return llvm::None;
  }
  return time.getOffset();
}

// ScheduledOp interface.
hir::Time CallOp::getStartTime() { return hir::Time(tstart(), offset()); }

hir::Time hir::ForOp::getStartTime() { return hir::Time(tstart(), offset()); }
hir::Time hir::WhileOp::getStartTime() { return hir::Time(tstart(), offset()); }
hir::Time hir::IsFirstIterOp::getStartTime() {
  return hir::Time(tstart(), offset());
}
hir::Time hir::NextIterOp::getStartTime() {
  return hir::Time(tstart(), offset());
}
hir::Time hir::IfOp::getStartTime() { return hir::Time(tstart(), offset()); }
hir::Time hir::DelayOp::getStartTime() { return hir::Time(tstart(), offset()); }
hir::Time hir::TimeOp::getStartTime() { return hir::Time(timevar(), offset()); }
hir::Time hir::LoadOp::getStartTime() { return hir::Time(tstart(), offset()); }
hir::Time hir::StoreOp::getStartTime() { return hir::Time(tstart(), offset()); }
hir::Time hir::BusSendOp::getStartTime() {
  return hir::Time(tstart(), offset());
}
hir::Time hir::BusRecvOp::getStartTime() {
  return hir::Time(tstart(), offset());
}

void CallOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}

void ForOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void WhileOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void IsFirstIterOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void NextIterOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}

void IfOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}

void DelayOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}

void TimeOp::setStartTime(hir::Time time) {
  this->timevarMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void LoadOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void StoreOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void BusSendOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}
void BusRecvOp::setStartTime(hir::Time time) {
  this->tstartMutable().assign(time.getTimeVar());
  this->offsetAttr(IntegerAttr::get(IntegerType::get(this->getContext(), 64),
                                    time.getOffset()));
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
CallOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  auto funcTy = this->funcTy().dyn_cast<hir::FuncType>();
  for (size_t i = 0; i < this->getNumResults(); i++) {
    Value res = this->getResult(i);
    Type resTy = res.getType();
    if (helper::isBuiltinSizedType(resTy)) {
      DictionaryAttr attrDict = funcTy.getResultAttrs()[i];
      uint64_t delay = helper::extractDelayFromDict(attrDict).getValue();
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

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
ForOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  Value endTimeVar = this->getResults().back();
  auto tripCount = this->getTripCount();
  auto ii = this->getInitiationInterval();
  Optional<hir::Time> endTime;
  if (tripCount.hasValue() && ii.hasValue()) {
    hir::Time startTime = this->getStartTime();
    endTime = startTime.addOffset(tripCount.getValue() * ii.getValue());
  } else {
    endTime = Time(endTimeVar, 0);
  }
  if (!endTime.has_value()) {
    output.push_back(std::make_pair(endTimeVar, endTime));
    return output;
  }

  if (this->iter_arg_delays()) {
    auto iterArgDelays = this->iter_arg_delays().getValue();
    for (size_t i = 0; i < iterArgDelays.size(); i++) {
      auto delay = iterArgDelays[i].dyn_cast<mlir::IntegerAttr>().getInt();
      auto result = this->getResult(i);
      output.push_back(std::make_pair(result, endTime->addOffset(delay)));
    }
  }

  output.push_back(std::make_pair(endTimeVar, endTime));
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
WhileOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  Value endTimeVar = this->getResults().back();
  if (this->iter_arg_delays()) {
    auto iterArgDelays = this->iter_arg_delays().getValue();
    for (size_t i = 0; i < iterArgDelays.size(); i++) {
      auto delay = iterArgDelays[i].dyn_cast<mlir::IntegerAttr>().getInt();
      auto result = this->getResult(i);
      output.push_back(std::make_pair(result, Time(endTimeVar, delay)));
    }
  }
  output.push_back(std::make_pair(endTimeVar, Time(endTimeVar, 0)));
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
IsFirstIterOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(this->getResult(), this->getStartTime()));
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
NextIterOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
IfOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  auto resultAttrs = this->result_attrs();
  for (size_t i = 0; i < this->getNumResults(); i++) {
    Value res = this->getResult(i);
    Type resTy = res.getType();
    if (helper::isBuiltinSizedType(resTy)) {
      uint64_t delay =
          resultAttrs.getValue()[i].dyn_cast<IntegerAttr>().getInt();
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

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
DelayOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(
      this->getResult(), this->getStartTime().addOffset(this->delay())));
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
TimeOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(this->getResult(), this->getStartTime()));
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
LoadOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  output.push_back(std::make_pair(
      this->getResult(), this->getStartTime().addOffset(this->delay())));
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
StoreOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
BusSendOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
  return output;
}

SmallVector<std::pair<Value, Optional<hir::Time>>, 4>
BusRecvOp::getResultsWithTime() {
  SmallVector<std::pair<Value, Optional<hir::Time>>, 4> output;
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

Optional<int64_t> FuncOp::getRegionII() {
  // FIXME: Currently we are assuming FuncOp is not pipelined.
  return INT64_MAX;
}

Optional<int64_t> ForOp::getRegionII() { return getInitiationInterval(); }

Optional<int64_t> WhileOp::getRegionII() { return llvm::None; }