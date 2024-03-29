//===- SSPAttributes.td - SSP attribute definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect attributes
//===----------------------------------------------------------------------===//

def DependenceAttr : AttrDef<SSPDialect, "Dependence"> {
  let summary = "Internal representation of dependence edges.";
  let description = [{
    An attribute to uniformly model def-use and auxiliary
    [dependences](https://circt.llvm.org/docs/Scheduling/#components) as well as
    to attach [properties](https://circt.llvm.org/docs/Scheduling/#properties)
    to them. This attribute is an implementation detail of the `ssp.OperationOp`
    and as such is supposed to be hidden by the custom parser/printer.
  }];

  let parameters = (ins "unsigned":$operandIdx,
                        OptionalParameter<"::mlir::FlatSymbolRefAttr">:$sourceRef,
                        OptionalParameter<"::mlir::ArrayAttr">:$properties);

  let mnemonic = "dependence";
  let assemblyFormat = [{
    `<` $operandIdx (`,` $sourceRef^)? (`:` $properties^)? `>`
  }];
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;

//===----------------------------------------------------------------------===//
// Property definitions for in-tree problems
//===----------------------------------------------------------------------===//

include "PropertyBase.td"

// Problem
def LinkedOperatorTypeProp : OperationProperty<SSPDialect,
  "LinkedOperatorType", "::mlir::FlatSymbolRefAttr", "::circt::scheduling::Problem"> {
  let mnemonic = "opr";
  // Need to override the default implementation because to (un)wrap the
  // StringAttr from/for the symbol reference
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op) {
      prob.set}] # propertyName # [{(op, getValue().getLeafReference());
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op, ::mlir::MLIRContext *ctx) {
      if (auto value = prob.get}] # propertyName # [{(op))
        return }] # cppClassName # [{::get(ctx, ::mlir::FlatSymbolRefAttr::get(ctx, *value));
      return {};
    }
  }];
}
def StartTimeProp : OperationProperty<SSPDialect,
  "StartTime", "unsigned", "::circt::scheduling::Problem"> {
  let mnemonic = "t";
}
def LatencyProp : OperatorTypeProperty<SSPDialect,
  "Latency", "unsigned", "::circt::scheduling::Problem"> {
  let mnemonic = "latency";
}

// CyclicProblem
def DistanceProp : DependenceProperty<SSPDialect,
  "Distance", "unsigned", "::circt::scheduling::CyclicProblem"> {
  let mnemonic = "dist";
}
def InitiationIntervalProp : InstanceProperty<SSPDialect,
  "InitiationInterval", "unsigned", "::circt::scheduling::CyclicProblem"> {
  let mnemonic = "II";
}

// SharedOperatorsProblem
def LimitProp : OperatorTypeProperty<SSPDialect,
  "Limit", "unsigned", "::circt::scheduling::SharedOperatorsProblem"> {
  let mnemonic = "limit";
}
