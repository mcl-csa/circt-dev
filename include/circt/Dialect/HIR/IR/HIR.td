#ifndef HIR_TD
#define HIR_TD

include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

include "HIRTypes.td"
include "HIRAttrs.td"
include "HIROpInterfaces.td"

//Ops
include "HIRFuncOps.td"
include "HIRCtrlFlowOps.td"
include "HIRMemOps.td"
include "HIRTimeOps.td"
include "HIRMiscOps.td"
include "HIRBusOps.td"
include "HIRWireOps.td"

#endif // HIR_TD
