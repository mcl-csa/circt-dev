#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/EnumAttr.td"

// Dialect
def HIR_Dialect : Dialect {
  let name = "hir";
  let summary = "An MLIR dialect for Hardware description";
  let description = [{
    This dialect is designed for hardware description. A circuit described in HIR 
      dialect contains the schedule of execution and functional unit binding 
      information. HIR supports high level constructs such as loops and 
      conditionals.
  }];
  let cppNamespace = "::circt::hir";
  let hasConstantMaterializer = 1;
}


class HIR_Op<string mnemonic, list<Trait> traits = []>
: Op<HIR_Dialect, mnemonic, traits>;

include "HIROpInterfaces.td"
include "HIRTypes.td"
include "HIRAttrs.td"

//Ops
include "HIRFuncOps.td"
include "HIRCtrlFlowOps.td"
include "HIRMemOps.td"
include "HIRTimeOps.td"
include "HIRMiscOps.td"
include "HIRBusOps.td"

#endif // HIR_OPS
