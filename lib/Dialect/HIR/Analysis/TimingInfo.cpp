#include "circt/Dialect/HIR/Analysis/TimingInfo.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HW/HWOps.h"

using namespace circt;
using namespace hir;

//-----------------------------------------------------------------------------
// EquivalentTimeMap class methods.
//-----------------------------------------------------------------------------
void EquivalentTimeMap::registerEquivalentTime(Value timeVar, Time time) {
  assert(timeVar.getType().isa<hir::TimeType>());
  if (time.getTimeVar() == timeVar)
    return;
  mapTimeVarToOffsetAndEquivalentTimeVar[time.getTimeVar()].push_back(
      std::make_pair(time.getOffset(), timeVar));
}

unsigned int EquivalentTimeMap::getLexicalOrder(ScheduledOp op) {
  auto it = mapOpToLexicalOrder.find(op);
  assert(it != mapOpToLexicalOrder.end());
  return it->getSecond();
}

unsigned int EquivalentTimeMap::getLexicalOrder(Value op) {
  return getLexicalOrder(dyn_cast<hir::ScheduledOp>(op.getDefiningOp()));
}

Time EquivalentTimeMap::getEquivalentTimeWithSmallerOffset(ScheduledOp op) {
  auto time = op.getStartTime();
  auto equivalentTimeVarForDifferentOffsets =
      mapTimeVarToOffsetAndEquivalentTimeVar[time.getTimeVar()];

  // Search for the closest timevar equivalent to time.
  int64_t offset = 0;
  Value newTimeVar;
  for (std::pair<int64_t, Value> offsetAndEquivalentTimeVar :
       equivalentTimeVarForDifferentOffsets) {
    if (offsetAndEquivalentTimeVar.first < offset)
      continue;
    if (offsetAndEquivalentTimeVar.first > time.getOffset())
      continue;
    if (getLexicalOrder(op) <=
        getLexicalOrder(offsetAndEquivalentTimeVar.second))
      continue;
    offset = offsetAndEquivalentTimeVar.first;
    newTimeVar = offsetAndEquivalentTimeVar.second;
  }

  // If no equivalent timevar is found then return the original time.
  if (!newTimeVar)
    return time;

  return Time(newTimeVar, time.getOffset() - offset);
}

//-----------------------------------------------------------------------------
// TimingInfo class methods.
//-----------------------------------------------------------------------------

TimingInfo::TimingInfo(FuncOp op)
    : equivalentTimeMap(mapOpToLexicalOrder), currentLexicalPos(0) {
  auto walkResult =
      op.walk<mlir::WalkOrder::PreOrder>([this](Operation *operation) {
        if (auto scheduledOp = dyn_cast<ScheduledOp>(operation)) {
          if (failed(visitOp(scheduledOp)))
            return WalkResult::interrupt();
        } else if (auto hwConstantOp =
                       dyn_cast<circt::hw::ConstantOp>(operation)) {
          if (failed(visitOp(hwConstantOp)))
            return WalkResult::interrupt();
        } else if (auto arithConstantOp =
                       dyn_cast<mlir::arith::ConstantOp>(operation)) {
          if (failed(visitOp(arithConstantOp)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  assert(!walkResult.wasInterrupted());
}

void TimingInfo::registerValue(Value value, Time time) {
  if (time.getTimeVar() == value) {
    assert(time.getOffset() == 0);
    return;
  }
  if (value.getType().isa<hir::TimeType>()) {
    equivalentTimeMap.registerEquivalentTime(value, time);
  } else
    mapValueToTime[value] = time;
}

LogicalResult TimingInfo::visitOp(ScheduledOp op) {
  mapOpToLexicalOrder[op] = currentLexicalPos++;
  auto resultsWithTime = op.getResultsWithTime();
  for (auto resultWithTime : resultsWithTime) {
    Value result = resultWithTime.first;
    // Time variables generated using TimeOp, WhileOp or CallOp are root level
    // time variables.
    if ((result.getType().isa<hir::TimeType>() &&
         isa<hir::ForOp>(op.getOperation())) ||
        helper::isBuiltinSizedType(result.getType())) {
      if (auto time = resultWithTime.second)
        registerValue(result, *time);
    }
  }
  return success();
}

LogicalResult TimingInfo::visitOp(circt::hw::ConstantOp op) {
  setOfConstants.insert(op.getResult());
  return success();
}

LogicalResult TimingInfo::visitOp(mlir::arith::ConstantOp op) {
  setOfConstants.insert(op.getResult());
  return success();
}

bool TimingInfo::isValidAtTime(Value v, hir::Time time) {
  if (isAlwaysValid(v))
    return true;
  if (time == getTime(v))
    return true;
  return false;
}

bool TimingInfo::isAlwaysValid(Value v) {
  if (setOfConstants.find(v) != setOfConstants.end())
    return true;
  return false;
}

Time TimingInfo::getTime(Value v) {
  auto timeIter = mapValueToTime.find(v);
  assert(timeIter != mapValueToTime.end());
  return timeIter->second;
}

Time TimingInfo::getOptimizedTime(hir::ScheduledOp op) {
  return equivalentTimeMap.getEquivalentTimeWithSmallerOffset(op);
}
