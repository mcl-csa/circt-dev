#ifndef HIR_TD
#define HIR_TD

include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

include "circt/Dialect/HIR/IR/HIRDialect.td"
include "circt/Dialect/HIR/IR/HIRTypes.td"
include "circt/Dialect/HIR/IR/HIRAttrs.td"
include "circt/Dialect/HIR/IR/HIROpInterfaces.td"

//Ops
include "circt/Dialect/HIR/IR/HIRFuncOps.td"
include "circt/Dialect/HIR/IR/HIRCtrlFlowOps.td"
include "circt/Dialect/HIR/IR/HIRMemOps.td"
include "circt/Dialect/HIR/IR/HIRTimeOps.td"
include "circt/Dialect/HIR/IR/HIRMiscOps.td"
include "circt/Dialect/HIR/IR/HIRBusOps.td"
include "circt/Dialect/HIR/IR/HIRWireOps.td"

#endif // HIR_TD
