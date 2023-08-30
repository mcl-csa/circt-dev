#ifndef CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
#define CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
include "mlir/Pass/PassBase.td"

def FuseHWInst : Pass<"fuse-hw-inst", "hw::HWModuleOp"> {
  let summary = ".";
  let description = [{This pass fuses hw instances.}];

  let constructor = "circt::hir::createFuseHWInstPass()";
  let dependentDialects = ["hw::HWDialect","comb::CombDialect"];
}
def OptBitWidth : Pass<"hir-opt-bitwidth", "hir::FuncOp"> {
  let summary = ".";
  let description = [{This pass reduces bitwidth of various ops.}];

  let constructor = "circt::hir::createOptBitWidthPass()";
  let dependentDialects = ["hir::HIRDialect","mlir::arith::ArithmeticDialect"];
}

def OpFusion : Pass<"hir-fuse-op", "hir::FuncOp"> {
  let summary = ".";
  let description = [{This pass fuses operations.}];
  let constructor = "circt::hir::createOpFusionPass()";
  let dependentDialects = ["hir::HIRDialect"];
}

def OptDelay : Pass<"hir-opt-delay", "hir::FuncOp"> {
  let summary = ".";
  let description = [{This pass hir.delay (and thus shiftreg/LUT) if the enclosing scope has large initiation interval.}];

  let constructor = "circt::hir::createOptDelayPass()";
  let dependentDialects = ["hir::HIRDialect"];
}

def OptTime : Pass<"hir-opt-time", "hir::FuncOp"> {
  let summary = ".";
  let description = [{This pass reduces time offset related shift register usage.}];

  let constructor = "circt::hir::createOptTimePass()";
  let dependentDialects = ["hir::HIRDialect","mlir::arith::ArithmeticDialect"];
}

def SimplifyCtrl : Pass<"hir-simplify-ctrl", "hir::FuncOp"> {
  let summary = "Simplifies ForOp and IfOp.";

  let constructor = "circt::hir::createSimplifyCtrlPass()";
  let dependentDialects = ["circt::hw::HWDialect","circt::sv::SVDialect","circt::comb::CombDialect","mlir::arith::ArithmeticDialect"];
}

def MemrefLowering : Pass<"hir-lower-memref", "mlir::ModuleOp"> {
  let summary = "Lower hir.memref to hir.bus.";
  let description = [{This pass lowers hir.memref to hir.bus for codegen.}];

  let constructor = "circt::hir::createMemrefLoweringPass()";
  let dependentDialects = ["mlir::arith::ArithmeticDialect",
  "circt::hw::HWDialect","circt::comb::CombDialect","hir::HIRDialect"];
}

def VerifySchedule : Pass<"hir-verify-schedule", "hir::FuncOp"> {
  let summary = "Verify that the schedule is correct";
  let description = [{This pass finds anomalies in HIR schedules.}];

  let constructor = "circt::hir::createVerifySchedulePass()";
}



def LoopUnroll : Pass<"hir-loop-unroll", "hir::FuncOp"> {
  let summary = "Unroll loop body";
  let description = [{This pass unrolls the body of the UnrollForOp and removes
  the op.
  }];

  let constructor = "circt::hir::createLoopUnrollPass()";
  let dependentDialects = ["mlir::arith::ArithmeticDialect"];
}
#endif // CIRCT_DIALECT_HIR_TRANSFORMS_PASSES
