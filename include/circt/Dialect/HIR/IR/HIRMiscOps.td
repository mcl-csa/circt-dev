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

def CastOp: HIR_Op<"cast",[NoSideEffect]>{
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


