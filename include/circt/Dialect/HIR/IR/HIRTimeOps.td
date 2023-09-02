include "mlir/Interfaces/SideEffectInterfaces.td"

def DelayOp: HIR_Op<"delay",[NoSideEffect,DeclareOpInterfaceMethods<ScheduledOp>]>{
  let summary = "Delay operation";
  let description = [{
    This op delays the input value by a fixed amount of time.
      Example:
      ```mlir %yy = hir.delay %xx by %1 at %tj```
  }];

  let arguments = (ins AnyType : $input, I64Attr:$delay, 
      HIR_TimeType:$tstart, I64Attr: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{$input `by` $delay `at` custom<TimeAndOffset> 
    ($tstart, $offset) custom<WithSSANames>(attr-dict) `:` type($res) custom<CopyType>(type($input),ref(type($res)))}];
  let hasVerifier = 1;
}

def TimeOp: HIR_Op<"time",[NoSideEffect,DeclareOpInterfaceMethods<ScheduledOp>]>{
  let summary = "Define a new time var";
  let description = [{
    This op defines a new time var that represents a fixed delay from another time var.
      Example:
      ```mlir %t = hir.time %t+3 :!hir.time``
  }];

  let arguments = (ins HIR_TimeType : $timevar, I64Attr:$offset);
  let results = (outs HIR_TimeType : $res);
  let assemblyFormat = [{custom<TimeAndOffset>($timevar, $offset) attr-dict `:` type($res)}];
  let hasFolder = true;
}

def TimeMaxOp: HIR_Op<"time.max",[NoSideEffect]>{
  let summary = "Define a new time var that is the largest of the input time vars.";
  let description = [{
    This op defines a new time var that is the larger of the input time vars.
      Example:
      ```mlir %t = hir.time (%t1, %t2, %t3) :!hir.time``
  }];

  let arguments = (ins Variadic<HIR_TimeType> : $input_timevars);
  let results = (outs HIR_TimeType : $res);
  let assemblyFormat = [{`(` $input_timevars `)` custom<WithSSANames>(attr-dict) `:` type($res)}];
}
