//=========- HIROpCanonicalizer.cpp - Canonicalize HIR Ops ----------------===//
//
// This file implements op canonicalizers for HIR dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR//helper.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/IR/IRMapping.h>

using namespace circt;
using namespace hir;
using namespace llvm;

template <typename OPTYPE>
static LogicalResult splitOffsetIntoSeparateOp(OPTYPE op,
                                               PatternRewriter &rewriter) {
  auto *context = rewriter.getContext();
  if (!op.getOffset())
    return failure();
  if (op.getOffset() == 0)
    return failure();

  Value tstart =
      rewriter.create<hir::TimeOp>(op.getLoc(), helper::getTimeType(context),
                                   op.getTstart(), op.getOffsetAttr());

  op.getTstartMutable().assign(tstart);
  op.setOffsetAttr(rewriter.getI64IntegerAttr(0));
  return success();
}

LogicalResult LoadOp::canonicalize(LoadOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult WhileOp::canonicalize(WhileOp op,
                                    ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult StoreOp::canonicalize(StoreOp op,
                                    ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult BusSendOp::canonicalize(BusSendOp op,
                                      ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult BusRecvOp::canonicalize(BusRecvOp op,
                                      ::mlir::PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult ForOp::canonicalize(ForOp op, PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult CallOp::canonicalize(CallOp op, PatternRewriter &rewriter) {
  return splitOffsetIntoSeparateOp(op, rewriter);
}

LogicalResult IfOp::canonicalize(IfOp op, mlir::PatternRewriter &rewriter) {
  LogicalResult result = splitOffsetIntoSeparateOp(op, rewriter);

  auto constantOp = dyn_cast_or_null<mlir::arith::ConstantOp>(
      op.getCondition().getDefiningOp());
  if (!constantOp)
    return result;

  int condition = constantOp.getValue().dyn_cast<IntegerAttr>().getInt();
  IRMapping operandMap;
  SmallVector<Value> yieldedValues;
  operandMap.map(op.getRegionTimeVar(), op.getTstart());
  Region &selectedRegion = condition ? op.getIfRegion() : op.getElseRegion();
  for (Operation &operation : selectedRegion.front()) {
    if (auto yieldOp = dyn_cast<hir::YieldOp>(operation)) {
      assert(yieldOp.getOperands().size() == op.getResults().size());
      for (Value value : yieldOp.getOperands())
        yieldedValues.push_back(helper::lookupOrOriginal(operandMap, value));
      continue;
    }
    rewriter.clone(operation, operandMap);
  }
  if (!yieldedValues.empty())
    rewriter.replaceOp(op, yieldedValues);
  return success();
}

LogicalResult NextIterOp::canonicalize(NextIterOp op,
                                       mlir::PatternRewriter &rewriter) {

  return splitOffsetIntoSeparateOp(op, rewriter);
}

OpFoldResult TimeOp::fold(TimeOp::FoldAdaptor a) {
  auto startTime = this->getStartTime();
  if (startTime.getOffset() == 0)
    return startTime.getTimeVar();
  return {};
}
