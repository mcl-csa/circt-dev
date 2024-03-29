#ifndef HIR_OPINTERFACES
#define HIR_OPINTERFACES

include "circt/Dialect/HIR/IR/HIRDialect.td"

def RegionOp : OpInterface<"RegionOp"> {
let cppNamespace = "circt::hir"; 
  let description = [{
    This is an interface for all hir ops that have a region which takes a timevar.
  }];

  let methods = [
    InterfaceMethod<
      "Get the region timevars for this operation.",
      "SmallVector<Value>", "getRegionTimeVars"    
      >,
    InterfaceMethod<
      "Get the region II for this operation. II= INT64_MAX implies that there is no pipelining.",
      "Optional<int64_t>", "getRegionII"    
      >,
  ];
}

def ScheduledOp : OpInterface<"ScheduledOp"> {
let cppNamespace = "circt::hir"; 
  let description = [{
    This is an interface for all hir ops that have a schedule.
  }];

  let methods = [
    InterfaceMethod<
      "Get the timevar for this operation.",
      "hir::Time", "getStartTime"    
      >,
    InterfaceMethod<
      "Get all results with time.",
      "llvm::SmallVector<std::pair<Value,mlir::Optional<hir::Time>>,4>", "getResultsWithTime"    
      >,
    InterfaceMethod<
      "Set new start time.",
      "void", "setStartTime",(ins "hir::Time":$time)    
      >,
  ];
}

def FuncLike : OpInterface<"FuncLike"> {
let cppNamespace = "circt::hir"; 
  let description = [{
    This is an interface for hir.func like ops.
  }];

  let methods = [
    InterfaceMethod<
      "Get the hir::FuncType for this operation.",
      "hir::FuncType", "getFuncType"
    >,
    InterfaceMethod<
      "Get the Func body .",
      "Region&", "getFuncBody"
    >,
    InterfaceMethod<
      "Update func input args.",
      "void", "updateArguments", (ins "ArrayRef<DictionaryAttr>":$attrs)
    >,
    InterfaceMethod<
      "Get input args names.",
      "ArrayAttr", "argNames"
    >,
    InterfaceMethod<
      "Get input args names.",
      "llvm::Optional<ArrayAttr>", "resultNames"
    >,
  ];
}
#endif
