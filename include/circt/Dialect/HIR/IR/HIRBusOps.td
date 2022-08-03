include "mlir/Interfaces/SideEffectInterfaces.td"
include "HIRDialect.td"
include "HIRTypes.td"
include "HIROpInterfaces.td"

def BusOp :HIR_Op<"bus",[]> {
   let summary = "Instantiate new bus.";

   let description = [{ Instantiate a new !hir.bus 
      Example: 
         ```%b = hir.bus : !hir.bus<i32>```
   }];

   let arguments = (ins);
   let results = (outs HIR_BusType:$res);

   let skipDefaultBuilders = 1;
   let builders = [OpBuilder<(ins "Type":$resTy)>];

   let assemblyFormat = [{custom<WithSSANames>(attr-dict) `:` type($res) }];
}

def BusSendOp : HIR_Op<"bus.send", [DeclareOpInterfaceMethods<ScheduledOp>]> {
   let summary = "write to a bus.";
   let description = [{
      This Op writes to a bus. 
         Example:
         ```mlir hir.send %1 to %v : !hir.bus<...>```
   }];

   let arguments = (ins AnyType:$value, HIR_BusType:$bus,
         HIR_TimeType :$tstart, I64Attr: $offset);
   let results = (outs );
   let assemblyFormat = [{
      $value `to` $bus `at` custom<TimeAndOffset> ($tstart, $offset) attr-dict `:` type($value) `to` type($bus)  
   }];
   let hasCanonicalizeMethod = true;
}

def BusRecvOp : HIR_Op<"bus.recv", [NoSideEffect,DeclareOpInterfaceMethods<ScheduledOp>]> {
   let summary = "Read from a bus.";
   let description = [{
      This Op Reads from a bus. 
         ```mlir %v = hir.recv %x : !hir.bus<...> -> i32```
   }];

   let arguments = (ins HIR_BusType:$bus,
         HIR_TimeType :$tstart, I64Attr: $offset);
   let results = (outs AnyType:$res);

   let assemblyFormat = [{
      $bus `at` custom<TimeAndOffset>($tstart, $offset) custom<WithSSANames>(attr-dict)  `:` type($bus) `->` type($res)  
   }];
   let hasCanonicalizeMethod = true;
}

def BusAssignOp: HIR_Op<"bus.assign",[]>{
   let summary = "Assign one bus to another.";
   let description = [{
      This op assigns one bus to another.
         Examples:
         ```mlir 
         hir.bus.assign %dest, %src: !hir.bus<i32>```
   }];

   let arguments = (ins HIR_BusType: $dest, HIR_BusType:$src);

   let assemblyFormat = [{ $dest `,` $src custom<WithSSANames>(attr-dict) `:`
      type($dest)
         custom<CopyType>(type($src),ref(type($dest))) }];
}

def BusMapOp: HIR_Op<"bus.map",[SingleBlock,IsolatedFromAbove, NoSideEffect]>{
   let summary = "Create new buses from input buses.";
   let description = [{
      This op creates new buses from multiple input buses
         by calculating the output values from the input values every cycle.
         The body of the op can only contain ops from comb dialect and hir.yield
         terminator op.
         Examples:
         ```mlir %out_bus 
         = hir.bus.map (%cond <- %sel , %trueValue <- %in1, %falseValue <- %in2):
            (i1,i32,i32)-> (i32)
      {
         %r = comb.mux %cond, %in1, %in2 :
            hir.yield %r:i32
      } ```
   }];

   let arguments = (ins Variadic<HIR_BusType>: $operands);
   let results = (outs Variadic<HIR_BusType>: $results);
   let regions = (region SizedRegion<1>:$body);

   let skipDefaultBuilders = 1;
   let builders = [OpBuilder<(ins "ArrayRef<Value>":$operands,
         "std::function<hir::YieldOp(OpBuilder&, ArrayRef<Value>)>":$bodyCtor)>];

   let hasCustomAssemblyFormat = 1;
}

def BusTensorOp :HIR_Op<"bus_tensor",[]> {
   let summary = "Instantiate new unassigned bus tensor.";
   let description = [{ Instantiate a new !hir.bus_tensor 
      Example: 
         ```%t = hir.bus_tensor : : !hir.bus_tensor<2x4xi32>```
   }];

   let arguments = (ins);
   let results = (outs HIR_BusTensorType:$res);

   let assemblyFormat = [{custom<WithSSANames>(attr-dict) `:` type($res) }];
}

def BusTensorAssignOp: HIR_Op<"bus_tensor.assign",[]>{
   let summary = "Assign one bus to another.";
   let description = [{
      This op assigns one bus to another.
         Examples:
         ```mlir 
         hir.bus_tensor.assign %dest, %src: !hir.bus<i32>```
   }];

   let arguments = (ins HIR_BusTensorType: $dest, HIR_BusTensorType:$src);

   let assemblyFormat = [{ $dest `,` $src custom<WithSSANames>(attr-dict) `:`
      type($dest)
         custom<CopyType>(type($src),ref(type($dest))) }];
}

def BusTensorAssignElementOp :HIR_Op<"bus_tensor.assign_element",[]>{
   let summary = "Assign a bus to a bus_tensor element.";
   let description = [{
      This op assigns a single bus to a bus_tensor element.
         Examples:
         ```mlir hir.bus_tensor.assign_element %b to %x[%0] : !hir.bus<i32> to hir.bus_tensor<i32>```
   }];

   let arguments = (ins HIR_BusTensorType: $tensor, Variadic<Index>: $indices, HIR_BusType:$bus);
   let results = (outs );

   let assemblyFormat = [{ $tensor `[`$indices`]` `,` $bus   custom<WithSSANames>(attr-dict) `:`
      type($tensor) `,` type($bus) }];
}

def BusTensorGetElementOp :HIR_Op<"bus_tensor.get_element",[NoSideEffect]>{
   let summary = "Extract a bus from a bus tensor";
   let description = [{
      This op extracts a single bus from a tensor of buses.
         Examples:
         ```mlir %1 = hir.bus_tensor.get_element %x[%0] : tensor<4x!hir.bus<...>>```
   }];

   let arguments = (ins HIR_BusTensorType: $tensor, Variadic<Index>: $indices);
   let results = (outs HIR_BusType: $res);

   let assemblyFormat = [{ $tensor `[`$indices`]` custom<WithSSANames>(attr-dict) `:`
      type($tensor) `->` type($res) }];
}

def BusTensorInsertElementOp: HIR_Op<"bus_tensor.insert_element",[NoSideEffect]>{
   let summary = "Create a new bus_tensor by replacing an element from the input.";
   let description = [{
      This op creates a new bus_tensor from an old bus_tensor by replacing one
         element.
         Examples:
         ```mlir %new_tensor 
         = hir.bus_tensor.insert %element into %tensor[%c0,%c1] : type(%res)```
   }];

   let arguments = (ins HIR_BusType: $element, HIR_BusTensorType:$tensor, 
         Variadic<Index>: $indices);
   let results = (outs HIR_BusTensorType: $res);

   let hasCustomAssemblyFormat = 1;
   let hasVerifier = 1;
}

def BusTensorMapOp: HIR_Op<"bus_tensor.map",[SingleBlock,IsolatedFromAbove, NoSideEffect]>{
   let summary = "Create new bus tensors from input bus tensors.";
   let description = [{
      This op creates new bus tensors from multiple input bus tensors
         by calculating the output values from the input values every cycle.
         The body of the op can only contain ops from comb dialect and hir.yield
         terminator op. All input and output tensors must have same shape.
         Examples:
         ```mlir %out_bus_tensor 
         = hir.bus_tensor.map (%in1 <- %input1 , %in2 <- %input1):
            (!hir.bus_tensor<5xi1>,!hir.bus_tensor<5xi1>)-> (!hir.bus_tensor<5xi1>)
      {
         %r = comb.or %in1, %in2 : i1
            hir.yield %r:i1
      } ```
   }];

   let arguments = (ins Variadic<HIR_BusTensorType>: $operands);
   let results = (outs Variadic<HIR_BusTensorType>: $results);
   let regions = (region SizedRegion<1>:$body);

   let skipDefaultBuilders = 1;
   let builders = [OpBuilder<(ins "ArrayRef<Value>":$operands,
         "std::function<hir::YieldOp(OpBuilder&, ArrayRef<Value>)>":$bodyCtor)>];

   let hasCustomAssemblyFormat = 1;
   let hasVerifier = 1;
}

def BusBroadcastOp: HIR_Op<"bus.broadcast",[NoSideEffect]>{
   let summary = "broadcast bus to tensor of buses";
   let description = [{
      This op creates a new tensor of buses from a bus by broadcasting it.
         Examples:
         ```mlir %t 
         = hir.bus.broadcast %b : !hir.bus<i32> -> tensor<4x!hir.bus<i32>>```
   }];

   let arguments = (ins HIR_BusType: $bus);
   let results = (outs AnyType: $res);

   let assemblyFormat = [{ $bus custom<WithSSANames>(attr-dict) `:` type($bus) `->`type($res)}];
}

def InlineHWOp: HIR_Op<"inline_hw",[SingleBlock,IsolatedFromAbove,ParentOneOf<["FuncOp"]>]>{
   let summary = "Use hw, comb and sv dialect inside hir.";
   let arguments = (ins Variadic<HIR_BusType>: $inputs);
   let results = (outs Variadic<HIR_BusType>: $results);
   //let hasCustomAssemblyFormat = 1;
}
