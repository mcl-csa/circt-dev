#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <string>

using namespace circt;
using namespace hir;
struct HWPort {
  StringAttr name;
  hw::PortInfo::Direction direction;
  Type type;
  size_t argNum;
};

hw::PortInfo getPortInfo(HWPort port) {
  return hw::PortInfo({{port.name, port.type, port.direction}, port.argNum});
}

Value emitReg(OpBuilder &builder, Type elementTy, Value Input, Value tstart) {
  auto uLoc = builder.getUnknownLoc();
  auto c0 =
      builder.create<mlir::arith::ConstantOp>(uLoc, builder.getIndexAttr(0));
  auto ivReg = helper::emitRegisterAlloca(builder, elementTy);
  auto zeroAttr = builder.getI64IntegerAttr(0);
  auto oneAttr = builder.getI64IntegerAttr(1);

  builder.create<hir::StoreOp>(uLoc, Input, ivReg, ArrayRef<Value>({c0}),
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
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("lb"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = ivTy,
                                  .argNum = 0}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("ub"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = ivTy,
                                  .argNum = 1}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("step"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = ivTy,
                                  .argNum = 2}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("start"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = builder.getI1Type(),
                                  .argNum = 3}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("next"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = builder.getI1Type(),
                                  .argNum = 4}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("clk"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = builder.getI1Type(),
                                  .argNum = 5}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("rst"),
                                  .direction = hw::PortInfo::Direction::Input,
                                  .type = builder.getI1Type(),
                                  .argNum = 6}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("iv"),
                                  .direction = hw::PortInfo::Direction::Output,
                                  .type = ivTy,
                                  .argNum = 0}));
  modPorts.push_back(getPortInfo({.name = builder.getStringAttr("done"),
                                  .direction = hw::PortInfo::Direction::Output,
                                  .type = builder.getI1Type(),
                                  .argNum = 1}));
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

  auto lb = moduleOp.getArgumentForInput(0);
  auto ub = moduleOp.getArgumentForInput(1);
  auto step = moduleOp.getArgumentForInput(2);
  auto start = moduleOp.getArgumentForInput(3);
  auto next = moduleOp.getArgumentForInput(4);
  auto clk = moduleOp.getArgumentForInput(5);
  auto reset = moduleOp.getArgumentForInput(6);

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
      llvm::SmallVector<Value>({lb, ub, step, isFirstIter, next, clk, reset}));
  return std::make_pair(sm.getResult(1), sm.getResult(0));
}