#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include <string>

using namespace circt;
using namespace hir;

Value emitReg(OpBuilder &builder, Type elementTy, Value input, Value tstart) {
  auto uLoc = builder.getUnknownLoc();
  auto c0 =
      builder.create<mlir::arith::ConstantOp>(uLoc, builder.getIndexAttr(0));
  auto ivReg = helper::emitRegisterAlloca(builder, elementTy);
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);

  builder.create<hir::StoreOp>(uLoc, input, ivReg, ArrayRef<Value>({c0}),
                               oneAttr, oneAttr, tstart, zeroAttr);
  return builder.create<hir::LoadOp>(uLoc, elementTy, ivReg,
                                     ArrayRef<Value>({c0}), zeroAttr, zeroAttr,
                                     tstart, zeroAttr);
}

static Value getConstantX(OpBuilder &builder, Type ty) {
  assert(ty.isa<IntegerType>());
  return builder.create<sv::ConstantXOp>(builder.getUnknownLoc(), ty);
}

hw::HWModuleOp emitForOPStateMachineModule(OpBuilder &builder, Type ivTy) {
  OpBuilder::InsertionGuard guard(builder);
  auto uLoc = builder.getUnknownLoc();
  llvm::SmallVector<hw::PortInfo, 4> modPorts;
  modPorts.push_back({.name = builder.getStringAttr("lb"),
                      .direction = hw::PortDirection::INPUT,
                      .type = ivTy,
                      .argNum = 0});
  modPorts.push_back({.name = builder.getStringAttr("ub"),
                      .direction = hw::PortDirection::INPUT,
                      .type = ivTy,
                      .argNum = 1});
  modPorts.push_back({.name = builder.getStringAttr("step"),
                      .direction = hw::PortDirection::INPUT,
                      .type = ivTy,
                      .argNum = 2});
  modPorts.push_back({.name = builder.getStringAttr("start"),
                      .direction = hw::PortDirection::INPUT,
                      .type = builder.getI1Type(),
                      .argNum = 3});
  modPorts.push_back({.name = builder.getStringAttr("next"),
                      .direction = hw::PortDirection::INPUT,
                      .type = builder.getI1Type(),
                      .argNum = 4});
  modPorts.push_back({.name = builder.getStringAttr("clk"),
                      .direction = hw::PortDirection::INPUT,
                      .type = builder.getI1Type(),
                      .argNum = 5});
  modPorts.push_back({.name = builder.getStringAttr("rst"),
                      .direction = hw::PortDirection::INPUT,
                      .type = builder.getI1Type(),
                      .argNum = 6});
  modPorts.push_back({.name = builder.getStringAttr("iv"),
                      .direction = hw::PortDirection::OUTPUT,
                      .type = ivTy,
                      .argNum = 0});
  modPorts.push_back({.name = builder.getStringAttr("done"),
                      .direction = hw::PortDirection::OUTPUT,
                      .type = builder.getI1Type(),
                      .argNum = 1});
  builder.setInsertionPointToStart(&builder.getBlock()
                                        ->getParentOp()
                                        ->getParentOfType<mlir::ModuleOp>()
                                        .getBodyRegion()
                                        .getBlocks()
                                        .front());
  static int n = 0;
  auto moduleOp = builder.create<hw::HWModuleOp>(
      uLoc, builder.getStringAttr("ForOp_state_machine" + std::to_string(n++)),
      modPorts);

  auto lb = moduleOp.getArgument(0);
  auto ub = moduleOp.getArgument(1);
  auto step = moduleOp.getArgument(2);
  auto start = moduleOp.getArgument(3);
  auto next = moduleOp.getArgument(4);
  auto clk = moduleOp.getArgument(5);
  auto reset = moduleOp.getArgument(6);

  builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  auto zeroBit = helper::materializeIntegerConstant(builder, 0, 1);
  auto lbWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, lb}));
  auto ubWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, ub}));
  auto stepWide =
      builder.create<comb::ConcatOp>(uLoc, ArrayRef<Value>({zeroBit, step}));

  auto ivReg = builder.create<sv::RegOp>(uLoc, lbWide.getType(),
                                         builder.getStringAttr("iv_reg"));
  auto doneReg = builder.create<sv::RegOp>(uLoc, builder.getI1Type(),
                                           builder.getStringAttr("done_reg"));
  auto ivRegOut = builder.create<sv::ReadInOutOp>(uLoc, ivReg);
  auto ivWide = builder.create<comb::MuxOp>(uLoc, start, lbWide, ivRegOut);
  auto ivNext = builder.create<comb::AddOp>(uLoc, ivWide, stepWide);

  auto nextIterCondition = builder.create<comb::ICmpOp>(
      uLoc,
      ivWide.getType().isSignedInteger() ? comb::ICmpPredicate::sge
                                         : comb::ICmpPredicate::uge,
      ivNext, ubWide);

  auto regEnable =
      builder.create<comb::OrOp>(builder.getUnknownLoc(), start, next);

  auto bodyCtor = [&builder, &ivReg, &doneReg, &ivNext, &nextIterCondition,
                   &regEnable] {
    builder.create<sv::IfOp>(
        builder.getUnknownLoc(), regEnable,
        [&builder, &ivReg, &doneReg, &ivNext, &nextIterCondition] {
          builder.create<sv::PAssignOp>(builder.getUnknownLoc(),
                                        ivReg.getResult(), ivNext);
          builder.create<sv::PAssignOp>(builder.getUnknownLoc(),
                                        doneReg.getResult(), nextIterCondition);
        });
  };

  auto xValue = getConstantX(builder, ivRegOut.getType());
  auto resetCtor = [&builder, &ivReg, &doneReg, &zeroBit, &xValue] {
    builder.create<sv::PAssignOp>(builder.getUnknownLoc(), ivReg.getResult(),
                                  xValue);
    builder.create<sv::PAssignOp>(builder.getUnknownLoc(), doneReg.getResult(),
                                  zeroBit);
  };

  builder.create<sv::AlwaysFFOp>(
      builder.getUnknownLoc(), sv::EventControl::AtPosEdge, clk,
      ResetType::SyncReset, sv::EventControl::AtPosEdge, reset, bodyCtor,
      resetCtor);

  // The ivNext of THIS iteration decides the condition for next iteration.
  // The condition var should be available at the next tstartLoopBody.
  // So we store it in a reg to be read in the next iteration.
  auto done = builder.create<sv::ReadInOutOp>(uLoc, doneReg);
  auto iv = builder.create<comb::ExtractOp>(uLoc, lb.getType(), ivWide,
                                            builder.getI32IntegerAttr(0));

  Operation *oldOutputOp = moduleOp.getBodyBlock()->getTerminator();
  oldOutputOp->replaceAllUsesWith(builder.create<hw::OutputOp>(
      uLoc, llvm::SmallVector<mlir::Value, 2>({iv, done})));
  oldOutputOp->erase();
  return moduleOp;
}

std::pair<Value, Value> insertForOpStateMachine(OpBuilder &builder,
                                                Value isFirstIter, Value lb,
                                                Value ub, Value step,
                                                Value tstartLoopBody) {
  static int n = 0;
  auto module = emitForOPStateMachineModule(builder, lb.getType());
  auto next = builder
                  .create<hir::CastOp>(builder.getUnknownLoc(),
                                       builder.getI1Type(), tstartLoopBody)
                  .getResult();
  auto clk = builder
                 .create<hir::GetClockOp>(builder.getUnknownLoc(),
                                          builder.getI1Type(), tstartLoopBody)
                 .getResult();
  auto reset = builder
                   .create<hir::GetResetOp>(builder.getUnknownLoc(),
                                            builder.getI1Type(), tstartLoopBody)
                   .getResult();
  auto sm = builder.create<hw::InstanceOp>(
      builder.getUnknownLoc(), module, "ForOP_SM" + std::to_string(n++),
      llvm::SmallVector({lb, ub, step, isFirstIter, next, clk, reset}),
      ArrayAttr(), StringAttr());
  return std::make_pair(sm.getResult(1), sm.getResult(0));
}