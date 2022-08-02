// RUN: circt-opt --help | FileCheck %s --check-prefix=HELP
// RUN: circt-opt --show-dialects | FileCheck %s --check-prefix=DIALECT

// HELP: OVERVIEW: CIRCT modular optimizer driver

// DIALECT: Available Dialects:
// DIALECT-NEXT: affine
// DIALECT-NEXT: arith
// DIALECT-NEXT: builtin
// DIALECT-NEXT: calyx
// DIALECT-NEXT: cf
// DIALECT-NEXT: chirrtl
// DIALECT-NEXT: comb
// DIALECT-NEXT: esi
// DIALECT-NEXT: firrtl
// DIALECT-NEXT: fsm
// DIALECT-NEXT: func
// DIALECT-NEXT: handshake
// DIALECT-NEXT: hir
// DIALECT-NEXT: hw
// DIALECT-NEXT: hwarith
// DIALECT-NEXT: llhd
// DIALECT-NEXT: llvm
// DIALECT-NEXT: memref
// DIALECT-NEXT: moore
// DIALECT-NEXT: msft
// DIALECT-NEXT: scf
// DIALECT-NEXT: seq
// DIALECT-NEXT: ssp
// DIALECT-NEXT: staticlogic
// DIALECT-NEXT: sv
// DIALECT-NEXT: systemc
