// RUN: circt-opt -cse %s | FileCheck %s

// CHECK-LABEL: @cse_macro_ref()
// CHECK: [[VAR:%.+]] = sv.macro.ref< "PRINTF_COND_"> : i1
// CHECK: [[AND:%.+]] = comb.and [[VAR]], [[VAR]] : i1
// CHECK: hw.output [[AND]] : i1
hw.module @cse_macro_ref() -> (out: i1) {
  %PRINTF_COND__0 = sv.macro.ref< "PRINTF_COND_"> : i1
  %PRINTF_COND__1 = sv.macro.ref< "PRINTF_COND_"> : i1
  %0 = comb.and %PRINTF_COND__0, %PRINTF_COND__1 : i1
  hw.output %0 : i1
}
