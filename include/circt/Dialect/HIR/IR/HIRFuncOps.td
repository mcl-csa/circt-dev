include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def FuncOp : HIR_Op<"func", 
  [Symbol, SingleBlockImplicitTerminator<"ReturnOp">, FunctionOpInterface, FuncLike,
  IsolatedFromAbove, CallableOpInterface, HasParent<"mlir::ModuleOp">,DeclareOpInterfaceMethods<RegionOp>]> {

  let summary = "Function definition";
  let description = [{
    This Op creates a new function 
      Example:

      ```mlir 
      hir.func @foo (%x, %A) at %t : (i4,!hir.memref<...>) -> (f32,!hir.time)
      ```
  }];

  let arguments = (ins
      TypeAttr:$function_type,
      StrAttr:$sym_name,
      TypeAttr:$funcTy,
      ArrayAttr:$argNames,
      OptionalAttr<ArrayAttr>:$resultNames
      );

  let regions = (region SizedRegion<1>:$body);
  let results = (outs );

  let builders = [OpBuilder<(ins "mlir::StringRef":$sym_name,
  "hir::FuncType":$funcTy, "ArrayAttr":$argNames, "ArrayAttr":$resultNames)>];

  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    private:
      // FunctionOpInterface interface.
      public:
      mlir::FunctionType getFunctionType();
      hir::FuncType getFuncType();
      ArrayRef<Type> getArgumentTypes(); 
      ArrayRef<Type> getResultTypes() ;
      LogicalResult verifyType();
      LogicalResult verifyBody();
      Block *getBodyBlock() { return &body().front(); }
      public:
      //CallableOpInterface
      Region* getCallableRegion(){return &getOperation()->getRegion(0);}
      ArrayRef<Type> getCallableResults(){return getFuncType().getResultTypes();}

    public:
      Region& getFuncBody(){ return body();}
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
      Value getRegionTimeVar(){
        return getFuncBody().getArguments().back();
      }
      void updateArguments(ArrayRef<DictionaryAttr>);
      llvm::Optional<Block*> getOptionalFuncBodyBlock(){ return &getFuncBody().front();}
  }];
  let hasVerifier = 1;
}

def ReturnOp : HIR_Op<"return", [ParentOneOf<["FuncOp","FuncExternOp"]>, Terminator]> {

  let summary = "Return from function";
  let description = [{
    This op returns the output of the function.
    This instruction must be the last instruction in the function.
      ```mlir hir.return (%x, %y): (i32, !hir.const<i1>)```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins),[{
    build($_builder,$_state,SmallVector<Value>());
  }]>];
  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` `(` type($operands) `)`)? }];
}


def FuncExternOp : HIR_Op<"func.extern", 
[Symbol, SingleBlockImplicitTerminator<"ReturnOp">, FuncLike, HasParent<"mlir::ModuleOp">]> {

  let summary = "Function declaration";
  let description = [{
    This Op declares a new external function.
      Example:
      ```mlir 
      hir.func.extern @foo (%x, %A) at %t : (i4,!hir.memref<...>) -> (f32,!hir.time)
      ```
  }];

let arguments = (ins
      TypeAttr:$function_type,
      StrAttr:$sym_name,
      TypeAttr:$funcTy,
      ArrayAttr:$argNames,
      OptionalAttr<ArrayAttr>:$resultNames
      );

  let regions = (region SizedRegion<1>:$body);
  let results = (outs );

  let builders = [OpBuilder<(ins "mlir::StringRef":$sym_name,
  "hir::FuncType":$funcTy, "ArrayAttr":$argNames, "ArrayAttr":$resultNames)>];


  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    public:
      //unsigned getNumFunctionArguments() ;
      //unsigned getNumFunctionResults() ;
      public:
      LogicalResult verifyType();

    public:
    //FuncLike op interface.
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}
      Region& getFuncBody(){ return body();}
      void updateArguments(ArrayRef<DictionaryAttr>);
  }];
  //let hasVerifier = 1;
}

def CallOp : HIR_Op<"call",
[CallOpInterface,DeclareOpInterfaceMethods<ScheduledOp>] > {
  let summary = "Function call";
  let description = [{
    This Op calls a function/module. 
      Example:

      ```mlir 
      %a,%b = hir.call @foo (%x,%A) at %ti : !hir.func<(i32,!hir.memref<...>) -> (i32,i32)>
      ```
  }];

  let arguments = (ins 
      StrAttr:$instance_name,
      FlatSymbolRefAttr:$callee, 
      TypeAttr: $funcTy,
      Variadic<AnyType>:$operands, 
      HIR_TimeType:$tstart,
      I64Attr: $offset
      );
  let results = (outs Variadic<AnyType> : $results);

  let assemblyFormat = [{ $instance_name $callee `(` $operands`)` `at` custom<TimeAndOffset>($tstart ,$offset) custom<WithSSANames>(attr-dict) 
                            `:` custom<FunctionType>($funcTy, type($operands), type($results)) }];
  let hasCanonicalizeMethod = true;
  let hasVerifier = 1;
  let extraClassDeclaration=[{
    public:
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}

    //CallOpInterface

    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType() {return getFuncType().getFunctionType();}

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Lookup the FuncOp/ExternalFuncOp for the callee. This returns null on invalid IR.
    Operation* getCalleeDecl() ;
  }];
}
