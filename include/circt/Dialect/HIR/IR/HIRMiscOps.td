include "mlir/Interfaces/SideEffectInterfaces.td"
include "HIRDialect.td"

def CommentOp : HIR_Op<"comment", []> {
   let summary = "comment";
   let description = [{
      This op is used to add comments. This comments are output by the verilog-generator.```
   }];

   let arguments = (ins StrAttr:$comment );
   let results = (outs );

   let assemblyFormat = [{$comment attr-dict}];
}

def CastOp: HIR_Op<"cast",[Pure]>{
   let summary = "Cast between compatible types.";
   let description = [{
Examples:
      ```mlir %t 
         = hir.cast %b : !hir.time -> !hir.bus<i1>```
   }];

   let arguments = (ins AnyType: $input);
   let results = (outs AnyType: $res);

   let assemblyFormat = [{ $input custom<WithSSANames>(attr-dict) `:` type($input) `->`type($res)}];
   let hasVerifier = 1;
}

def GetClockOp: HIR_Op<"get_clk",[Pure]>{
   let summary = "Get the clock corresponding to a time variable.";
   let description = [{
Example:
      ```mlir %clk = hir.get_clk %t : i1```
   }];

   let arguments = (ins HIR_TimeType: $input);
   let results = (outs I1: $res);

   let assemblyFormat = [{ $input custom<WithSSANames>(attr-dict) `:` type($res)}];
}

def GetResetOp: HIR_Op<"get_reset",[Pure]>{
   let summary = "Get the reset signal corresponding to a time variable.";
   let description = [{
Example:
      ```mlir %rst = hir.get_reset %t : i1```
   }];

   let arguments = (ins HIR_TimeType: $input);
   let results = (outs I1: $res);

   let assemblyFormat = [{ $input custom<WithSSANames>(attr-dict) `:` type($res)}];
}

def ProbeOp: HIR_Op<"probe",[]>{
   let summary = "Preserve name of an SSA var.";
   let description = [{
      This op preserves the name of an input SSA var in the generated Verilog.
         Examples:
         ```mlir %t 
         = hir.probe %x name "verilog_name": !tensor<4xi32>```
   }];

   let arguments = (ins AnyType: $input, StrAttr: $verilog_name);
   let results = (outs );

   let assemblyFormat = [{ $input `name` $verilog_name attr-dict `:` type($input)}];
   let hasVerifier = 1;
}


