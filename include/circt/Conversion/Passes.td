//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_TD
#define CIRCT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// InstrumentCosim
//===----------------------------------------------------------------------===//

def InstrumentCosim : Pass<"instrument-cosim","mlir::ModuleOp"> {
  let summary = "Generate mlir files for cosimulation.";
  let constructor = "circt::createInstrumentCosimPass()";
  let dependentDialects = [
    "circt::hir::HIRDialect",
  ];
  let options = [
    Option<"dir", "dir", "std::string",
            "", "Output directory.">,
    Option<"entry", "entry", "std::string",
            "", "Entry function.">
   ];

}

//===----------------------------------------------------------------------===//
// HIRPragma
//===----------------------------------------------------------------------===//

def HIRPragma : Pass<"hir-pragma","mlir::ModuleOp"> {
  let summary = "Convert HLS pragma attrs from polygeist to hir specific attributes.";
  let constructor = "circt::createHIRPragmaPass()";
  let dependentDialects = [
    "circt::hir::HIRDialect",
  ];
  let options = [
    Option<"topLevelFuncName", "function", "std::string",
            "", "Top level function to convert to hir dialect.">
   ];

}

//===----------------------------------------------------------------------===//
// AutoAffineToHIR
//===----------------------------------------------------------------------===//

def AutoAffineToHIR : Pass<"auto-affine-to-hir","mlir::ModuleOp"> {
  let summary = "Convert Affine dialect to HIR dialect with autotuning.";
  let description = [{
    This pass converts affine dialect to hir while calculating initiation
    interval and schedule for improved op fusion.
  }];
  let constructor = "circt::createAutoAffineToHIRPass()";
  let dependentDialects = [
    "circt::hir::HIRDialect",
    "circt::hw::HWDialect",
    "circt::comb::CombDialect"
  ];
  let options = [
    Option<"dbg", "dbg", "bool","", "Enable debug print to stdout.">
   ];
}


//===----------------------------------------------------------------------===//
// AffineToHIR
//===----------------------------------------------------------------------===//

def AffineToHIR : Pass<"affine-to-hir","mlir::ModuleOp"> {
  let summary = "Convert Affine dialect to HIR dialect.";
  let description = [{
    This pass analyzes Affine loops and control flow, creates a Scheduling
    problem and solves it using ILP solver and lowers it to HIR.
  }];
  let constructor = "circt::createAffineToHIR()";
  let dependentDialects = [
    "circt::hir::HIRDialect",
    "circt::hw::HWDialect",
    "circt::comb::CombDialect"
  ];
  let options = [
    Option<"dbg", "dbg", "bool","", "Enable debug print to stdout.">
   ];
}

//===----------------------------------------------------------------------===//
// AffineToStaticLogic
//===----------------------------------------------------------------------===//

def AffineToStaticLogic : Pass<"convert-affine-to-staticlogic", "mlir::func::FuncOp"> {
  let summary = "Convert Affine dialect to StaticLogic pipelines";
  let description = [{
    This pass analyzes Affine loops and control flow, creates a Scheduling
    problem using the Calyx operator library, solves the problem, and lowers
    the loops to a StaticLogic pipeline.
  }];
  let constructor = "circt::createAffineToStaticLogic()";
  let dependentDialects = [
    "circt::staticlogic::StaticLogicDialect",
    "mlir::arith::ArithmeticDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExportVerilog and ExportSplitVerilog
//===----------------------------------------------------------------------===//

def TestPrepareForEmission : Pass<"test-prepare-for-emission",
                                  "hw::HWModuleOp"> {
  let summary = "Prepare IR for ExportVerilog";
  let description = [{
    This pass runs only PrepareForEmission logic for testing purpose.
    It is not necessary for users to run this pass explicitly since
    ExportVerilog internally runs PrepareForEmission.
  }];

  let constructor = "createTestPrepareForEmissionPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
}

def ExportVerilog : Pass<"export-verilog", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a (System)Verilog file";
  let description = [{
      This pass creates empty module bodies for external modules.  This is
      useful for linting to eliminate missing file errors.
  }];

  let constructor = "createExportVerilogPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
}

def ExportSplitVerilog : Pass<"export-split-verilog", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a (System)Verilog directory of files";
  let description = [{
    This pass generates (System)Verilog for the current design, mutating it
    where necessary to be valid Verilog.
  }];

  let constructor = "createExportSplitVerilogPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];

  let options = [
    Option<"directoryName", "dir-name", "std::string",
            "", "Directory to emit into">
   ];
}

//===----------------------------------------------------------------------===//
// ExportSystemC and ExportSplitSystemC
//===----------------------------------------------------------------------===//

def ExportSystemC : Pass<"export-systemc", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a SystemC C++ file";
  let description = [{
    This pass generates SystemC code for the current design. It outputs
    the entire design to a single output stream.
  }];

  let constructor = "createExportSystemCPass()";
  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect",
    "circt::systemc::SystemCDialect"
  ];
}

def ExportSplitSystemC : Pass<"export-split-systemc", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a directory of SystemC C++ files";
  let description = [{
    This pass generates SystemC code for the current design. It creates a new
    header file for every module in the design.
  }];

  let constructor = "createExportSplitSystemCPass()";
  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect",
    "circt::systemc::SystemCDialect"
  ];

  let options = [
    Option<"directoryName", "dir-name", "std::string",
            "", "Directory to emit into">
   ];
}

//===----------------------------------------------------------------------===//
// SCFToCalyx
//===----------------------------------------------------------------------===//

def SCFToCalyx : Pass<"lower-scf-to-calyx", "mlir::ModuleOp"> {
  let summary = "Lower SCF/Standard to Calyx";
  let description = [{
    This pass lowers SCF / standard to Calyx.
  }];
  let constructor = "circt::createSCFToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "::mlir::scf::SCFDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options = [
    Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
            "Identifier of top-level function to be the entry-point component"
            " of the Calyx program.">,
    Option<"ciderSourceLocationMetadata", "cider-source-location-metadata", "bool", "",
            "Whether to track source location for the Cider debugger.">
  ];
}

//===----------------------------------------------------------------------===//
// StaticLogicToCalyx
//===----------------------------------------------------------------------===//

def StaticLogicToCalyx : Pass<"lower-static-logic-to-calyx", "mlir::ModuleOp"> {
  let summary = "Lower StaticLogic to Calyx";
  let description = [{
    This pass lowers StaticLogic to Calyx.
  }];
  let constructor = "circt::createStaticLogicToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "::mlir::scf::SCFDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options = [
    Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
            "Identifier of top-level function to be the entry-point component"
            " of the Calyx program.">,
    Option<"ciderSourceLocationMetadata", "cider-source-location-metadata", "bool", "",
            "Whether to track source location for the Cider debugger.">
  ];
}

//===----------------------------------------------------------------------===//
// CalyxToHW
//===----------------------------------------------------------------------===//

def CalyxToHW : Pass<"lower-calyx-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Calyx to HW";
  let description = [{
    This pass lowers Calyx to HW.
  }];
  let constructor = "circt::createCalyxToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "seq::SeqDialect", "sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// CalyxToFSM
//===----------------------------------------------------------------------===//

def CalyxToFSM : Pass<"lower-calyx-to-fsm", "calyx::ComponentOp"> {
  let summary = "Lower Calyx to FSM";
  let description = [{
    This pass lowers a Calyx control schedule to an FSM representation.
    An `fsm.machine` operation is nested within the `control` region of the Calyx
    component. This machine is itself in an intermediate format wherein it has
    no I/O ports and solely contains output statements with `calyx.enable`s
    referencing `calyx.group` and transition logic guarded by the SSA values
    specified in the source control schedule.
    This intermediate state facilitates transformation of the FSM, given that
    top-level I/O has yet to be materialized (one input and output per activated
    group) as well as guard transition logic (every transition must be guarded
    on all groups active within the state having finished). As such, `calyx.enable`
    operations can easily be moved between states without worrying about updating
    transition guards while doing so.

    Eventually, the FSM must be materialized (materialize I/O ports, remove
    `calyx.enable` operations in favor of asserting output ports, guarding
    transitions by input `done` ports) and outlined to a separate module.
  }];
  let constructor = "circt::createCalyxToFSMPass()";
  let dependentDialects = ["fsm::FSMDialect", "comb::CombDialect"];
}

def MaterializeCalyxToFSM : Pass<"materialize-calyx-to-fsm", "calyx::ComponentOp"> {
  let summary = "Materializes an FSM embedded inside the control of this Calyx component.";
  let description = [{
    Materializes the FSM in the control of the component. This materializes the
    top-level I/O of the FSM to receive `group_done` signals as input and
    `group_go` signals as output, based on the `calyx.enable` operations
    used within the states of the FSM.
    Each transition of the FSM is predicated on the enabled groups within a
    state being done, or, for static groups, a separate sub-FSM is instantiated
    to await the group finishing.

    Given an FSM that enables N unique groups, the top-level FSM will have N+1
    in- and output ports, wherein:
    * Input # 0 to N-1 are `group_done` signals
    * Input N is the top-level `go` port
    * Output 0 to N-1 are `group_go` signals
    * Output N is the top-level `done` port
  }];
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect", "fsm::FSMDialect"];
  let constructor = "circt::createMaterializeCalyxToFSMPass()";
}


//===----------------------------------------------------------------------===//
// FSMToSV
//===----------------------------------------------------------------------===//

def ConvertFSMToSV : Pass<"convert-fsm-to-sv", "mlir::ModuleOp"> {
  let summary = "Convert FSM to HW";
  let constructor = "circt::createConvertFSMToSVPass()";
  let dependentDialects = ["circt::hw::HWDialect", "circt::comb::CombDialect",
                           "circt::seq::SeqDialect", "circt::sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// FIRRTLToHW
//===----------------------------------------------------------------------===//

def LowerFIRRTLToHW : Pass<"lower-firrtl-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower FIRRTL to HW";
  let description = [{
    Lower a module of FIRRTL dialect to the HW dialect family.
  }];
  let constructor = "circt::createLowerFIRRTLToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "seq::SeqDialect", "sv::SVDialect"];
  let options = [
    Option<"enableAnnotationWarning", "warn-on-unprocessed-annotations",
           "bool", "false",
    "Emit warnings on unprocessed annotations during lower-to-hw pass">,
    Option<"emitChiselAssertsAsSVA", "emit-chisel-asserts-as-sva",
           "bool", "false","Convert all Chisel asserts to SVA">
  ];
}

//===----------------------------------------------------------------------===//
// HandshakeToHW
//===----------------------------------------------------------------------===//

def HandshakeToHW : Pass<"lower-handshake-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to ESI/HW/Comb/Seq";
  let description = [{
    Lower Handshake to ESI/HW/Comb/Seq.
  }];
  let constructor = "circt::createHandshakeToHWPass()";
  let dependentDialects = ["hw::HWDialect", "esi::ESIDialect"];
}

//===----------------------------------------------------------------------===//
// HandshakeToFIRRTL
//===----------------------------------------------------------------------===//

def HandshakeToFIRRTL : Pass<"lower-handshake-to-firrtl", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to FIRRTL";
  let description = [{
    Lower Handshake to FIRRTL.
  }];
  let constructor = "circt::createHandshakeToFIRRTLPass()";
  let dependentDialects = ["firrtl::FIRRTLDialect"];
  let options = [
    Option<"enableFlattening", "flatten", "bool", "false",
    "Flattens the generated FIRRTL component by inlining all dataflow component"
    " instantiations into the top module.">,
  ];
}

//===----------------------------------------------------------------------===//
// HIRToHW
//===----------------------------------------------------------------------===//

def HIRToHW : Pass<"hir-to-hw", "mlir::ModuleOp"> {
  let summary = "Convert HIR to HW + Comb + SV dialect";
  let description = [{
    Convert an HIR module to HW, Comb and SV dialect.
  }];

  let constructor = "circt::createHIRToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// Moore to Comb/HW/LLHD
//===----------------------------------------------------------------------===//

def ConvertMooreToCore : Pass<"convert-moore-to-core", "mlir::ModuleOp"> {
  let summary = "Convert Moore to Core";
  let description = [{
    This pass translates Moore to the core dialects (Comb/HW/LLHD).
  }];
  let constructor = "circt::createConvertMooreToCorePass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "llhd::LLHDDialect"];
}

//===----------------------------------------------------------------------===//
// LLHDToLLVM
//===----------------------------------------------------------------------===//

def ConvertLLHDToLLVM : Pass<"convert-llhd-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert LLHD to LLVM";
  let description = [{
    This pass translates LLHD to LLVM."
  }];
  let constructor = "circt::createConvertLLHDToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// HWToLLVM
//===----------------------------------------------------------------------===//

def ConvertHWToLLVM : Pass<"convert-hw-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HW to LLVM";
  let description = [{
    This pass translates HW to LLVM."
  }];
  let constructor = "circt::createConvertHWToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// CombToLLVM
//===----------------------------------------------------------------------===//

def ConvertCombToLLVM : Pass<"convert-comb-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Comb to LLVM";
  let description = [{
    This pass translates Comb to LLVM."
  }];
  let constructor = "circt::createConvertCombToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// HWArithToHW
//===----------------------------------------------------------------------===//

def HWArithToHW : Pass<"lower-hwarith-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower HWArith to HW/Comb";
  let description = [{
    This pass lowers HWArith to HW/Comb.
  }];
  let constructor = "circt::createHWArithToHWPass()";
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect"];
}

//===----------------------------------------------------------------------===//
// HWToLLHD
//===----------------------------------------------------------------------===//

def ConvertHWToLLHD : Pass<"convert-hw-to-llhd", "mlir::ModuleOp"> {
  let summary = "Convert HW to LLHD";
  let description = [{
    This pass translates a HW design into an equivalent structural LLHD
    description.
  }];
  let constructor = "circt::createConvertHWToLLHDPass()";
  let dependentDialects = ["llhd::LLHDDialect"];
}

//===----------------------------------------------------------------------===//
// HWToSystemC
//===----------------------------------------------------------------------===//

def ConvertHWToSystemC : Pass<"convert-hw-to-systemc", "mlir::ModuleOp"> {
  let summary = "Convert HW to SystemC";
  let description = [{
    This pass translates a HW design into an equivalent SystemC design.
  }];
  let constructor = "circt::createConvertHWToSystemCPass()";
  let dependentDialects = ["systemc::SystemCDialect"];
}

//===----------------------------------------------------------------------===//
// StandardToHandshake
//===----------------------------------------------------------------------===//

def StandardToHandshake : Pass<"lower-std-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lower Standard MLIR into Handshake IR";
  let constructor = "circt::createStandardToHandshakePass()";
  let dependentDialects = ["handshake::HandshakeDialect"];
  let options =
    [Option<"sourceConstants", "source-constants", "bool", "false",
            "If true, will connect constants to source operations instead of "
            "to the control network. May reduce the size of the final circuit.">,
     Option<"disableTaskPipelining", "disable-task-pipelining", "bool", "false",
            "If true, will disable support for task pipelining. This relaxes the "
            "restrictions put on the structure of the input CDFG. Disabling "
            "task pipelining may severely reduce kernel II.">];
}

def HandshakeRemoveBlock : Pass<"handshake-remove-block-structure", "handshake::FuncOp"> {
  let summary = "Remove block structure in Handshake IR";
  let constructor = "circt::createHandshakeRemoveBlockPass()";
}

def InsertMergeBlocks : Pass<"insert-merge-blocks", "::mlir::ModuleOp"> {
  let summary = "Insert explicit merge blocks";
  let description = [{
    This pass inserts additional merge blocks for each block with more than
    two successors. A merge block is a block that only contains one operation,
    a terminator, and has two predecessors.
    The order successors are merged together mirrors the order different control
    paths are created. Thus, each block with two successors will have a corresponding
    merge block.

    This pass brings the CFG into a canonical form for further transformations.

    Treats loops and sub-CFGs with irregular control flow like single blocks.
  }];
  let constructor = "circt::createInsertMergeBlocksPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// StandardToStaticLogic
//===----------------------------------------------------------------------===//

def CreatePipeline : Pass<"create-pipeline", "mlir::func::FuncOp"> {
  let summary = "Create StaticLogic pipeline operations";
  let constructor = "circt::createCreatePipelinePass()";
  let dependentDialects = ["staticlogic::StaticLogicDialect"];
}

#endif // CIRCT_CONVERSION_PASSES_TD
