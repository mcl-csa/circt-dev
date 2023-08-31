include "mlir/Interfaces/SideEffectInterfaces.td"
include "HIRDialect.td"
include "HIRTypes.td"
include "HIROpInterfaces.td"

def WireOp :HIR_Op<"wire",[Pure]> {
   let summary = "Instantiate a new wire.";

   let description = [{ Instantiate a new !hir.wire 
      The wire is unconnected (Z value in Verilog) when instantiated.
      hir.drive is used to connect the wire with a driver.
      Example: 
         ```%w = hir.wire : !hir.wire<i32>```
   }];

   let arguments = (ins);
   let results = (outs HIR_WireType:$res);

   let assemblyFormat = [{custom<WithSSANames>(attr-dict) `:` type($res) }];
}

def DriveOp :HIR_Op<"drive",[]> {
   let summary = "Connect a wire with its driver.";

   let description = [{ Connects a wire to its driver.
   A wire can have only one driver.
      Example: 
         ```%w = hir.wire : !hir.wire<i32>```
   }];

   let arguments = (ins HIR_WireType:$wire , AnyType:$value);
   let results = (outs);

   let assemblyFormat = [{$wire `value` $value custom<WithSSANames>(attr-dict) `:` type($wire)`,`type($value) }];
} 