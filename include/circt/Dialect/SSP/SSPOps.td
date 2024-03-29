//===- SSPOps.td - SSP operation definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class SSPOp<string mnemonic, list<Trait> traits = []> :
  Op<SSPDialect, mnemonic, traits>;

def InstanceOp : SSPOp<"instance",
    [NoRegionArguments, SingleBlock, NoTerminator,
     IsolatedFromAbove, OpAsmOpInterface]> {
  let summary = "Instance of a static scheduling problem.";
  let description = [{
    This operation represents an instance of a static scheduling problem,
    comprised of an operator library (`OperatorLibraryOp`, a container for
    `OperatorTypeOp`s) and the dependence graph (`DependenceGraphOp`, a
    container for `OperationOp`s). The instance and its components (operations,
    operator types and dependences) can carry properties, i.e. special MLIR
    attributes inheriting from the TableGen classes in `PropertyBase.td`. The
    `ssp` dialect provides attribute definitions (and short-form
    pretty-printing) for CIRCT's built-in scheduling problems.
    
    **Example**
    ```mlir
    ssp.instance "canis14_fig2" of "ModuloProblem" [II<3>] {
      library {
        operator_type @MemPort [latency<1>, limit<1>]
        operator_type @Add [latency<1>]
      }
      graph {
        %0 = operation<@MemPort> @load_A(@store_A [dist<1>]) [t<2>]
        %1 = operation<@MemPort> @load_B() [t<0>]
        %2 = operation<@Add> @add(%0, %1) [t<3>]
        operation<@MemPort> @store_A(%2) [t<4>]
      }
    }
    ```
  }];

  let arguments = (ins StrAttr:$instanceName, StrAttr:$problemName,
                       OptionalAttr<ArrayAttr>:$properties);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $instanceName `of` $problemName custom<Properties>($properties) $body attr-dict
  }];
  
  let hasVerifier = true;

  let extraClassDeclaration = [{
    // OpAsmOpInterface
    static ::llvm::StringRef getDefaultDialect() { return "ssp"; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }

    // Access to container ops
    ::circt::ssp::OperatorLibraryOp getOperatorLibrary();
    ::circt::ssp::DependenceGraphOp getDependenceGraph();
  }];

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "::mlir::StringAttr":$instanceName,
                   "::mlir::StringAttr":$problemName,
                   CArg<"::mlir::ArrayAttr", "::mlir::ArrayAttr()">:$properties), [{
      $_state.addAttribute($_builder.getStringAttr("instanceName"), instanceName);
      $_state.addAttribute($_builder.getStringAttr("problemName"), problemName);
      if (properties)
        $_state.addAttribute($_builder.getStringAttr("properties"), properties);
      ::mlir::Region* region = $_state.addRegion();
      region->push_back(new ::mlir::Block());
    }]>
  ];
}

class ContainerOp<string mnemonic, list<Trait> traits = []>
    : SSPOp<mnemonic, traits # [NoRegionArguments, SingleBlock, NoTerminator,
     SymbolTable, OpAsmOpInterface, HasParent<"InstanceOp">]> {
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";

  let extraClassDeclaration = [{
    // OpAsmOpInterface
    static ::llvm::StringRef getDefaultDialect() { return "ssp"; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins ), [{
      ::mlir::Region* region = $_state.addRegion();
      region->push_back(new ::mlir::Block());
    }]>
  ];
}

def OperatorLibraryOp : ContainerOp<"library"> {
  let summary = "Container for operator types.";
  let description = [{
    The operator library abstracts the characteristics of the target
    architecture/IR (onto which the source graph is scheduled), represented by
    the individual `OperatorTypeOp`s.
  }];
}

def DependenceGraphOp : ContainerOp<"graph",
    [RegionKindInterface, HasOnlyGraphRegion]> {
  let summary = "Container for (scheduling) operations.";
  let description = [{
    The dependence graph is spanned by `OperationOp`s (vertices) and a
    combination of MLIR value uses and symbol references (edges).
  }];
}

def OperatorTypeOp : SSPOp<"operator_type",
    [Symbol, HasParent<"OperatorLibraryOp">]> {
  let summary = "Element of the target architecture/IR.";
  let description = [{
    This operation represents an operator type, which can be augmented with a
    set of problem-specific properties, and is identified through a unique name.

    **Example**
    ```mlir
    operator_type @MemPort [latency<1>, limit<1>]
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<ArrayAttr>:$properties);
  let assemblyFormat = "$sym_name custom<Properties>($properties) attr-dict";
}

def OperationOp : SSPOp<"operation",
    [Symbol, HasParent<"DependenceGraphOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Vertex and incoming edges in the dependence graph.";
  let description = [{
    This MLIR operation represents an operation (in the terminology of CIRCT's
    scheduling infra) in a scheduling problem, or in other words, a vertex in
    the surrounding instance's dependence graph. In addition, it also encodes
    the operation's incoming dependences. In order to faithfully reproduce the
    internal modeling in the scheduling infrastructure, these dependence edges
    are either encoded as MLIR operands (def-use dependences) or symbol
    references (auxiliary dependences). To that end, `OperationOp`s can
    optionally be named, and accept/produce an arbitrary number of
    operands/results. The operation and the incoming dependences can carry
    properties.

    The `linkedOperatorType` property in the root `Problem` class is central to
    the problem models, because it links operations to their properties in the
    target IR. Therefore, the referenced operator type symbol is parsed/printed
    right after the operation keyword in the custom assembly syntax.

    **Examples**
    ```mlir
    // unnamed, only def-use dependences
    %2 = operation<@Add>(%0, %1)

    // unnamed, multiple results
    %5:2 = operation<@Div>(%3, %4) // multiple results
    
    // named, mix of def-use and auxiliary dependences
    operation<@MemPort> @store_A(%2, @store_B, @load_A)
    
    // dependence properties
    operation<@Barrier>(%2 [dist<1>], %5#1, @store_A [dist<3>])
    ```
  }];

  let arguments = (ins Variadic<NoneType>:$operands,
                       OptionalAttr<SymbolNameAttr>:$sym_name,
                       OptionalAttr<DependenceArrayAttr>:$dependences,
                       OptionalAttr<ArrayAttr>:$properties);
  let results = (outs Variadic<NoneType>:$results);

  let hasCustomAssemblyFormat = true;
  let hasVerifier = true;

  let extraClassDeclaration = [{
    // SymbolUserOpInterface
    static bool isOptionalSymbol() { return true; }
  }];

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "unsigned":$numResults,
                   "::mlir::ValueRange":$operands,
                   CArg<"::mlir::StringAttr", "::mlir::StringAttr()">:$sym_name,
                   CArg<"::mlir::ArrayAttr", "::mlir::ArrayAttr()">:$dependences,
                   CArg<"::mlir::ArrayAttr", "::mlir::ArrayAttr()">:$properties), [{
      $_state.addTypes(::llvm::SmallVector<::mlir::Type>(numResults, $_builder.getNoneType()));
      $_state.addOperands(operands);
      if (sym_name)
        $_state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), sym_name);
      if (dependences)
        $_state.addAttribute($_builder.getStringAttr("dependences"), dependences);
      if (properties)
        $_state.addAttribute($_builder.getStringAttr("properties"), properties);
    }]>
  ];
}
