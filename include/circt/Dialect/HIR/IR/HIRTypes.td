#ifndef HIR_TYPES
#define HIR_TYPES
include "mlir/IR/AttrTypeBase.td"
include "HIRDialect.td"

class HIRDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def HIR_TimeType : HIRDialect_Type<"Time", "time"> {
  let summary = "hir.time type.";
  let description = [{
    This is the type of time variables in hir dialect.
    Syntax: !hir.time
  }];

  let parameters = (ins );
  let assemblyFormat = "";
}

def HIR_MemrefType : HIRDialect_Type<"Memref", "memref"> {
  let summary = "Memref type for hardware";
  let description = [{
    Memref type with support for banked dimensions.
    Syntax: !hir.memref<16x(bank 4)x8xi32>
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape,"Type":$elementType, ArrayRefParameter<"DimKind">:$dimKinds);
  let assemblyFormat = "`<`custom<BankedDimensionList>($shape,$dimKinds) $elementType`>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    int64_t getNumElementsPerBank();
    int64_t getNumBanks();
    SmallVector<int64_t> filterShape(DimKind dimKind);
  }];
}

def HIR_FuncType : HIRDialect_Type<"Func", "func"> {
  let summary = "Type of hir functions.";
  let description = [{
    Syntax: !hir.func<(i32,i32 delay 1)->(i64 delay 2)>
  }];

  let parameters = (ins 
  ArrayRefParameter<"Type">:$inputTypes, 
  ArrayRefParameter<"DictionaryAttr">:$inputAttrs,
  ArrayRefParameter<"Type">:$resultTypes, 
  ArrayRefParameter<"DictionaryAttr">: $resultAttrs);

  let assemblyFormat = "`<`custom<TypedArgList>($inputTypes,$inputAttrs) `->` custom<TypedArgList>($resultTypes,$resultAttrs)`>`";
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    FunctionType getFunctionType();
    unsigned int getNumInputs();
    unsigned int getNumResults();
    Type getInputType(unsigned int);
    DictionaryAttr getInputAttr(unsigned int);
    Type getResultType(unsigned int);
    DictionaryAttr getResultAttr(unsigned int);
  }];
}

def HIR_BusType : HIRDialect_Type<"Bus", "bus"> {
  let summary = "!hir.bus type.";
  let description = [{
    Syntax: !hir.bus<i32>
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<`$elementType`>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    FunctionType getFunctionType();
  }];
}
def HIR_BusTensorType : HIRDialect_Type<"BusTensor", "bus_tensor"> {
  let summary = "!hir.bus_tensor type.";
  let description = [{
    Syntax: !hir.bus_tensor<i32>
  }];

  let parameters = (ins  ArrayRefParameter<"int64_t">:$shape, "Type":$elementType);
  let assemblyFormat = "`<`custom<DimensionList>($shape)$elementType`>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    size_t getNumElements();
  }];
}

def HIR_WireType : HIRDialect_Type<"Wire", "wire"> {
  let summary = "Represents a wire in the hardware.";
  let description = [{
    This type is designed to circumvent the restriction of SSA dominance.
    A wire can have only one driver/writer and multiple readers.
    Wires are not allowed in function arguments/results and loop iter_args.

    Syntax: !hir.wire<i32>
  }];

  let parameters = (ins "IntegerType":$elementType);
  let assemblyFormat = "`<`$elementType`>`";
}
#endif
