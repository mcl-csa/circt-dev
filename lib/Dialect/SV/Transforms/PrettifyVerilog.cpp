//===- PrettifyVerilog.cpp - Transformations to improve Verilog quality ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass contains elective transformations that improve the quality of
// SystemVerilog generated by the ExportVerilog library.  This pass is not
// compulsory: things that are required for ExportVerilog to be correct should
// be included as part of the ExportVerilog pass itself to make sure it is self
// contained.  This allows the ExportVerilog pass to be simpler.
//
// PrettifyVerilog is run prior to Verilog emission but must be aware of the
// options in LoweringOptions.  It shouldn't introduce invalid constructs that
// aren't present in the IR already: this isn't a general "raising" pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// PrettifyVerilogPass
//===----------------------------------------------------------------------===//

namespace {
struct PrettifyVerilogPass
    : public sv::PrettifyVerilogBase<PrettifyVerilogPass> {
  void runOnOperation() override;

private:
  void processPostOrder(Block &block);
  bool prettifyUnaryOperator(Operation *op);
  void sinkOrCloneOpToUses(Operation *op);
  void sinkExpression(Operation *op);

  bool anythingChanged;
};
} // end anonymous namespace

/// Return true if this is something that will get printed as a unary operator
/// by the Verilog printer.
static bool isVerilogUnaryOperator(Operation *op) {
  if (isa<comb::ParityOp>(op))
    return true;

  if (auto xorOp = dyn_cast<comb::XorOp>(op))
    return xorOp.isBinaryNot();

  if (auto icmpOp = dyn_cast<comb::ICmpOp>(op))
    return icmpOp.isEqualAllOnes() || icmpOp.isNotEqualZero();

  return false;
}

/// Sink an operation into the same block where it is used.  This will clone the
/// operation so it can be sunk into multiple blocks. If there are no more uses
/// in the current block, the op will be removed.
void PrettifyVerilogPass::sinkOrCloneOpToUses(Operation *op) {
  assert(mlir::MemoryEffectOpInterface::hasNoEffect(op) &&
         "Op with side effects cannot be sunk to its uses.");
  auto block = op->getBlock();
  // This maps a block to the block local instance of the op.
  SmallDenseMap<Block *, Value, 8> blockLocalValues;
  for (auto &use : llvm::make_early_inc_range(op->getUses())) {
    // If the current use is in the same block as the operation, there is
    // nothing to do.
    auto localBlock = use.getOwner()->getBlock();
    if (block == localBlock)
      continue;
    // Find the block local clone of the operation. If there is not one already,
    // the op will be cloned in to the block.
    auto &localValue = blockLocalValues[localBlock];
    if (!localValue) {
      // Clone the operation and insert it to the beginning of the block.
      localValue = OpBuilder::atBlockBegin(localBlock).clone(*op)->getResult(0);
    }
    // Replace the current use, removing it from the use list.
    use.set(localValue);
    anythingChanged = true;
  }
  // If this op is no longer used, drop it.
  if (op->use_empty()) {
    op->erase();
    anythingChanged = true;
  }
}

/// This is called on unary operators.  This returns true if the operator is
/// moved.
bool PrettifyVerilogPass::prettifyUnaryOperator(Operation *op) {
  // If this is a multiple use unary operator, duplicate it and move it into the
  // block corresponding to the user.  This avoids emitting a temporary just for
  // a unary operator.  Instead of:
  //
  //    tmp1 = ^(thing+thing);
  //         = tmp1 + 42
  //
  // we get:
  //
  //    tmp2 = thing+thing;
  //         = ^tmp2 + 42
  //
  // This is particularly helpful when the operand of the unary op has multiple
  // uses as well.
  if (op->use_empty() || op->hasOneUse())
    return false;

  // Duplicating unary operations can move them across blocks (down the region
  // tree).  Make sure to keep referenced constants local.
  auto cloneConstantOperandsIfNeeded = [&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      auto constant = operand.get().getDefiningOp<hw::ConstantOp>();
      if (!constant)
        continue;

      // If the constant is in a different block, clone or move it into the
      // block.
      if (constant->getBlock() != op->getBlock()) {
        if (constant->hasOneUse())
          constant->moveBefore(op);
        else
          operand.set(OpBuilder(op).clone(*constant)->getResult(0));
      }
    }
  };

  while (!op->hasOneUse()) {
    OpOperand &use = *op->use_begin();
    Operation *user = use.getOwner();

    // Clone the operation and insert before this user.
    auto *cloned = OpBuilder(user).clone(*op);
    cloneConstantOperandsIfNeeded(cloned);

    // Update user's operand to the new value.
    use.set(cloned->getResult(0));
  }

  // There is exactly one user left, so move this before it.
  Operation *user = *op->user_begin();
  op->moveBefore(user);
  cloneConstantOperandsIfNeeded(op);

  anythingChanged = true;
  return true;
}

// Return the depth of the specified block in the region tree, stopping at
// 'topBlock'.
static unsigned getBlockDepth(Block *block, Block *topBlock) {
  unsigned result = 0;
  while (block != topBlock) {
    block = block->getParentOp()->getBlock();
    ++result;
  }
  return result;
}

/// This method is called on expressions to see if we can sink them down the
/// region tree.  This is a good thing to do to reduce scope of the expression.
void PrettifyVerilogPass::sinkExpression(Operation *op) {
  // Ignore expressions with no users.
  if (op->use_empty())
    return;

  Block *curOpBlock = op->getBlock();

  // Single-used expressions are the most common and simple to handle.
  if (op->hasOneUse()) {
    if (curOpBlock != op->user_begin()->getBlock()) {
      op->moveBefore(*op->user_begin());
      anythingChanged = true;
    }
    return;
  }

  // Find the nearest common ancestor of all the users.
  auto userIt = op->user_begin();
  Block *ncaBlock = userIt->getBlock();
  ++userIt;
  unsigned ncaBlockDepth = getBlockDepth(ncaBlock, curOpBlock);
  if (ncaBlockDepth == 0)
    return; // Have a user in the current block.

  for (auto e = op->user_end(); userIt != e; ++userIt) {
    auto *userBlock = userIt->getBlock();
    if (userBlock == curOpBlock)
      return; // Op has a user in it own block, can't sink it.
    if (userBlock == ncaBlock)
      continue;

    // Get the region depth of the user block so we can march up the region tree
    // to a common ancestor.
    unsigned userBlockDepth = getBlockDepth(userBlock, curOpBlock);
    while (userBlock != ncaBlock) {
      if (ncaBlockDepth < userBlockDepth) {
        userBlock = userBlock->getParentOp()->getBlock();
        --userBlockDepth;
      } else if (userBlockDepth < ncaBlockDepth) {
        ncaBlock = ncaBlock->getParentOp()->getBlock();
        --ncaBlockDepth;
      } else {
        userBlock = userBlock->getParentOp()->getBlock();
        --userBlockDepth;
        ncaBlock = ncaBlock->getParentOp()->getBlock();
        --ncaBlockDepth;
      }
    }

    if (ncaBlockDepth == 0)
      return; // Have a user in the current block.
  }

  // Ok, we found a common ancestor between all the users that is deeper than
  // the current op.  Sink it into the start of that block.
  assert(ncaBlock != curOpBlock && "should have bailed out earlier");
  op->moveBefore(&ncaBlock->front());
  anythingChanged = true;
}

void PrettifyVerilogPass::processPostOrder(Block &body) {
  SmallVector<Operation *> instances;

  // Walk the block bottom-up, processing the region tree inside out.
  for (auto &op :
       llvm::make_early_inc_range(llvm::reverse(body.getOperations()))) {
    if (op.getNumRegions()) {
      for (auto &region : op.getRegions())
        for (auto &regionBlock : region.getBlocks())
          processPostOrder(regionBlock);
    }

    // Sink and duplicate unary operators.
    if (isVerilogUnaryOperator(&op) && prettifyUnaryOperator(&op))
      continue;

    // Sink or duplicate constant ops and invisible "free" ops into the same
    // block as their use.  This will allow the verilog emitter to inline
    // constant expressions and avoids ReadInOutOp from preventing motion.
    if (matchPattern(&op, mlir::m_Constant()) ||
        isa<sv::ReadInOutOp, sv::ArrayIndexInOutOp>(op)) {
      sinkOrCloneOpToUses(&op);
      continue;
    }

    // Sink normal expressions down the region tree if they aren't used within
    // their current block.  This allows them to be folded into the using
    // expression inline in the best case, and better scopes the temporary wire
    // they generate in the worst case.  Our overall traversal order is
    // post-order here which means all users will already be sunk.
    if (hw::isCombinatorial(&op) || sv::isExpression(&op)) {
      sinkExpression(&op);
      continue;
    }

    if (isa<hw::InstanceOp>(op))
      instances.push_back(&op);
  }

  // If we have any instances, keep their relative order but shift them to the
  // end of the module.  Any outputs will be writing to a wire or an output port
  // of the enclosing module anyway, and this allows inputs to be inlined into
  // the operand list as parameters.
  if (!instances.empty()) {
    for (Operation *instance : llvm::reverse(instances)) {
      if (instance != &body.back())
        instance->moveBefore(&body.back());
    }
  }
}

void PrettifyVerilogPass::runOnOperation() {
  hw::HWModuleOp thisModule = getOperation();

  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;

  // Walk the operations in post-order, transforming any that are interesting.
  processPostOrder(*thisModule.getBodyBlock());

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createPrettifyVerilogPass() {
  return std::make_unique<PrettifyVerilogPass>();
}
