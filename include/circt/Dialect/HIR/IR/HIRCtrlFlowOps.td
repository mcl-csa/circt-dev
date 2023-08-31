include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "HIRDialect.td"
include "HIRTypes.td"
include "HIROpInterfaces.td"

def ForOp : HIR_Op<"for", [SingleBlock, RecursiveMemoryEffects,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>,DeclareOpInterfaceMethods<RegionOp>,DeclareOpInterfaceMethods<ScheduledOp>]> {
       let summary = "for operation";
       let description = [{
          This op implements a sequential for loop.
             ```mlir
             hir.for %i = 0 to %N step 1 at iter_time(%ti = %t offset %1) {
                hir.for %j = %lb to %ub step %s iter_args(%xx = %x at %tj) at iter_time(%tj = %ti offset %1) 
                {   
                   %yy = hir.delay %xx by 1 at %tj
                      hir.yield %yy at %tj offset 1;
                }
                hir.yield at %ti offset 1;
             }
          ```
       }];

       let arguments = (ins AnyType: $lb, AnyType: $ub, AnyType:
             $step,Variadic<AnyType>:$iter_args, OptionalAttr<ArrayAttr>: $iter_arg_delays,
             HIR_TimeType :$tstart, I64Attr : $offset, OptionalAttr<I64Attr> : $initiation_interval);
       let regions = (region SizedRegion<1>:$body);
       let results = (outs Variadic<AnyType>:$iterResults, HIR_TimeType:$t_end);

       let builders = [OpBuilder<(ins "Value":$lb, "Value":$ub,
             "Value":$step,"ArrayRef<Value>":$iterArgs,
             "Value":$tstart, "IntegerAttr":$offset,
             "std::function<hir::NextIterOp(OpBuilder&, Value, ArrayRef<Value>, Value)>":$bodyCtor)>];

       let hasCustomAssemblyFormat = 1;

       let extraClassDeclaration = [{
          Value getInductionVar() ;
          Value setInductionVar(Type ty) ;
          SmallVector<Value> getIterArgArguments();    
          Value getIterTimeVar() ;
          Block* addEntryBlock(MLIRContext *context, Type inductionVarTy);
          SmallVector<Value> getCapturedValues();
          StringRef getInductionVarName();
          StringRef getIterTimeVarName();
          std::optional<int64_t> getTripCount();
          Value getIterArgOperand(unsigned int);
          void setIterArgOperand(unsigned int, Value);
       }];

       let hasCanonicalizeMethod = true; 
       let hasVerifier = 1;
}

def WhileOp : HIR_Op<"while", [SingleBlock, RecursiveMemoryEffects,DeclareOpInterfaceMethods<RegionOp>,DeclareOpInterfaceMethods<ScheduledOp>]> {
   let summary = "for operation";

   let arguments = (ins 
         I1:$condition,
         Variadic<AnyType>:$iter_args, 
         OptionalAttr<ArrayAttr>:$iter_arg_delays, 
         HIR_TimeType:$tstart, 
         I64Attr: $offset);
   let results = (outs Variadic<AnyType>:$iterResults, HIR_TimeType:$t_end);
   let regions = (region SizedRegion<1>:$body);
   let hasCustomAssemblyFormat = 1;

   let extraClassDeclaration = [{
      Value getIterTimeVar() { return getBody().getArguments().back(); }
      SmallVector<Value> getCapturedValues();
      Block* addEntryBlock();

   }];

   let hasCanonicalizeMethod =true;
}

def IsFirstIterOp:HIR_Op<"is_first_iter",[ParentOneOf<["WhileOp","ForOp"]>,DeclareOpInterfaceMethods<ScheduledOp>]>{
   let summary = "Returns true if this is the first iteration.";

   let arguments = (ins 
         HIR_TimeType:$tstart, 
         I64Attr: $offset);
   let results = (outs I1:$res);

   let assemblyFormat = [{ custom<WithSSANames>(attr-dict) `at` custom<TimeAndOffset>($tstart ,$offset)`:` type($res)}];
   let hasVerifier=true;
}

def NextIterOp : HIR_Op<"next_iter", [AttrSizedOperandSegments,
    ParentOneOf<["ForOp","WhileOp"]>,Terminator,DeclareOpInterfaceMethods<ScheduledOp>] > {
       let summary = "specifies the time at which the next iteration of the loop starts.";
       let description = [{
          This op specifies when the next iteration of the loop starts.
             If the current iteration is the last iteration then it specifies the time 
             returned by the op.
             If a condition is provided then the loop breaks when the condition is false;

          This instruction must be the last instruction in for loop body.

             Example:

             ```mlir hir.next_iter break %continue at %t```
             ```mlir hir.next_iter at %t```
       }];

       let arguments = (ins Optional<I1> :$condition, Variadic<AnyType>:$iter_args, HIR_TimeType : $tstart, 
             I64Attr: $offset);

       let results = (outs );
       let hasCanonicalizeMethod = true;
       let assemblyFormat = [{ (`break` $condition^)? (`iter_args` `(`$iter_args^ `)`)? `at` custom<TimeAndOffset>($tstart ,$offset) attr-dict (`:` `(` type($iter_args)^ `)`)? }];
       let hasVerifier=1;

}

def IfOp : HIR_Op<"if", [SingleBlock, RecursiveMemoryEffects, DeclareOpInterfaceMethods<ScheduledOp>]> {
   let summary = "if operation";
   let description = [{
      This op implements an if statement.
         Example:
         ```mlir
         hir.if(%b) at %t {...}        ```
   }];

   let arguments = (ins AnyType : $condition, HIR_TimeType : $tstart, I64Attr: $offset, OptionalAttr<ArrayAttr>:$result_attrs);
   let regions = (region SizedRegion<1>:$if_region, SizedRegion<1>:$else_region);
   let results = (outs Variadic<AnyType>:$results);

   let extraClassDeclaration = [{
      Value getRegionTimeVar(){
         return getIfRegion().getArguments().back();
      }

   }];
   let hasCustomAssemblyFormat = 1;
   let hasVerifier = 1;
   let hasCanonicalizeMethod =true;
}

def YieldOp : HIR_Op<"yield", [ParentOneOf<["IfOp", "BusMapOp","BusTensorMapOp"]>,Terminator] > {
   let summary = "Yield from loops";
   let description = [{
      This Op yields values out of an hir.if.
         This instruction must be the last instruction in if/else bodies.
         Example:

         ```mlir hir.yield (%x, %A) : (f32, !hir.time)```
   }];

   let arguments = (ins Variadic<AnyType> : $operands);
   let results = (outs );

   let builders = [OpBuilder<(ins),[{
      build($_builder,$_state,SmallVector<Value>());
   }]>];
   let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` `(` type($operands) `)`)? }];
   let hasVerifier = 1;
   let hasVerifier=1;
}
