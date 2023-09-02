include "mlir/Interfaces/SideEffectInterfaces.td"

def AllocaOp :HIR_Op<"alloca",[]> {
   let summary = "Instantiate new memory.";

   let description = [{

      ```%w = hir.alloc("bram") : !hir.memref<...>```
   }];

   let arguments = (ins MemKindEnumAttr:$mem_kind, ArrayAttr: $ports);
   let results = (outs HIR_MemrefType:$res);

   let assemblyFormat = [{$mem_kind custom<WithSSANames>(attr-dict) 
      `:` type($res) `ports` $ports}];
   let hasVerifier = 1;
}

def LoadOp : HIR_Op<"load", [DeclareOpInterfaceMethods<ScheduledOp>]> {
   let summary = "read from a memory.";
   let description = [{
      This Op reads from a memory interface. 
         Indices for banks should be of index type and indices for address should be
         integers of correct width (based on the size of the dimension).

         Example:

         ```mlir hir.load %A[%x,%y] at %t : !hir.memref<...>```
   }];

   let arguments = (ins HIR_MemrefType:$mem, 
         Variadic<AnyType>:$indices, OptionalAttr<I64Attr>:$port, I64Attr: $delay, HIR_TimeType : $tstart,
         I64Attr: $offset);
   let results = (outs AnyType: $res);
   let assemblyFormat = [{
      $mem (`[` `port` $port^ `]`)? `[`  $indices `]` `at` custom<TimeAndOffset>($tstart ,$offset) 
         custom<WithSSANames>(attr-dict)
         `:`custom<MemrefAndElementType>
         (type($mem),type($indices),type($res)) `delay` $delay
   }];

   let hasCanonicalizeMethod = true;
   let hasVerifier = 1;
   let extraClassDeclaration = [{
      public:
         SmallVector<Value> filterIndices(hir::DimKind);
   }];
}

def StoreOp : HIR_Op<"store", [DeclareOpInterfaceMethods<ScheduledOp>]> {
   let summary = "Write to a memory.";
   let description = [{
      This Op writes to memory. 
         Indices for banks should be of index types and indices for address should be
         integers of correct width (based on the size of the dimension).

         Example:

         ```mlir hir.mem_write %v to %A[%x,%y] at %t + %1 : !hir.memref<...>```
   }];

   let arguments = (ins AnyType:$value, HIR_MemrefType:$mem,
         Variadic<AnyType>:$indices, OptionalAttr<I64Attr>:$port, 
         I64Attr:$delay, HIR_TimeType :$tstart, I64Attr: $offset);
   let results = (outs );
   let assemblyFormat = [{
      $value `to` $mem  (`[` `port` $port^ `]`)? `[` $indices `]` `at` custom<TimeAndOffset>($tstart, $offset) attr-dict `:` 
         custom<MemrefAndElementType>
         (type($mem),type($indices), type($value)) `delay` $delay
   }];

   let hasCanonicalizeMethod = true;
   let hasVerifier = 1;
   let extraClassDeclaration = [{
      public:
         SmallVector<Value> filterIndices(hir::DimKind);
   }];
}

def MemrefExtractOp :HIR_Op<"memref.extract",[NoSideEffect]>{
   let summary = "Extract a port from a memref";
   let description = [{
      This op extracts a port from a memref.
         Examples:
         ```mlir %1 = hir.memref.extract %x[port 1] : hir.memref<1xf32> -> hir.memref<1xf32> ports [#wr]```
   }];

   let arguments = (ins HIR_MemrefType: $mem, OptionalAttr<I64Attr>: $port, ArrayAttr: $ports);
   let results = (outs HIR_MemrefType: $res);

   let assemblyFormat = [{$mem `[` `port` $port `]`  custom<WithSSANames>(attr-dict) `:`
      type($res) `port` $ports  custom<CopyType>(type($mem),ref(type($res)))}];
}
