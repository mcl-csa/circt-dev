// RUN: circt-opt %s
#bram_r = {"rd_latency" = 1}
#bram_w = {"wr_latency" = 1}
hir.func @Array_Add at %t (%A:!hir.memref<128xi32> ports [#bram_r], 
%B : !hir.memref<128xi32> ports [#bram_r], 
%C:!hir.memref<128xi32> ports [#bram_w]){

hir.for %i:i8 = %c0_i8 to %c128_i8  step %c1_i8 iter_time(%ti = %t + 1){
  %x = hir.load %A[port 0][%i] at %ti :!hir.memref<128xi32> delay 1
  %y = comb.add  %x, %1  : i32
  %y1= hir.delay %y by 1 at %ti+1 : i32 
  hir.store %y1 to %C[port 1][%i] at %ti + 2 
    : !hir.memref<128xi32> delay 1
  hir.next_iter at %ti + 2
}

hir.for %i:i8 = %c0_i8 to %c128_i8  step %c1_i8 iter_time(%ti = %t + 1){
%x0 = hir.load %A[port 0][%i] at %t   :!hir.memref<128xi32> delay 1
%x1 = hir.load %A[port 0][%i+1] at %t+1  :!hir.memref<128xi32> delay 1
%y = comb.add %x0, %x1 
hir.store %y to %C[port 1][%i] at %t+2 
    : !hir.memref<128xi32> delay 1
  hir.next_iter at %ti + 1
}

  %c0_i8 = hw.constant 0: i8
  %c1_i8 = hw.constant 1:i8 
  %c128_i8 = hw.constant 128:i8 
  hir.for %i:i8 = %c0_i8 to %c128_i8  step %c1_i8 iter_time(%ti = %t + 1){
    %i_i7 = comb.extract %i from 0: (i8)->(i7)
    %i_delayed_i7 = hir.delay %i_i7 by 1 at %ti : i7 
    %a = hir.load %A[port 0][%i_i7] at %ti :!hir.memref<128xi32> delay 1
    %b = hir.load %B[port 0][%i_i7] at %ti : !hir.memref<128xi32> delay 1
    %c = comb.add  %a, %b  : i32
    hir.store %c to %C[port 0][%i_delayed_i7] at %ti + 1 
      : !hir.memref<128xi32> delay 1
    hir.next_iter at %ti + 1
  }
  hir.return
}{argNames=["A","B","C","t"]}
