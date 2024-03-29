// RUN: circt-opt %s
#bram_r = {"rd_latency"= 1}
#bram_w = {"wr_latency"= 1}
#reg_r  = {"rd_latency" = 0}
#reg_w  = {"wr_latency"= 1}
hir.func.extern @i32Multiplier at %t (%a:i32, %b:i32) ->(%result: i32 delay 4){argNames=["a","b","t"],resultNames=["result"]}

hir.func @gesummv_hir at %t(
%alpha:i32, 
%beta:i32, 
%tmp:!hir.memref<8xi32> ports [#bram_w] , 
%A:!hir.memref<8x8xi32>ports [#bram_r],
%B:!hir.memref<8x8xi32>ports [#bram_r],
%X:!hir.memref<8xi32>ports [#bram_r],
%Y:!hir.memref<8xi32>ports [#bram_w]
){


  %0 = arith.constant 0:index
  %c0_i32 = hw.constant 0:i32
  %c0_i5 = hw.constant 0:i5
  %c1_i5 = hw.constant 1:i5
  %c4_i5 = hw.constant 4:i5
  %c5_i5 = hw.constant 5:i5
  %c6_i5 = hw.constant 6:i5
  %c8_i5 = hw.constant 8:i5


  hir.for %i :i5 = %c0_i5  to %c8_i5 step %c1_i5  iter_time(%ti = %t){
    %tmpreg = hir.alloca reg  :!hir.memref<(bank 1)xi32> ports [#reg_r, #reg_w]
    %yreg = hir.alloca reg  :!hir.memref<(bank 1)xi32> ports [#reg_r, #reg_w]

    hir.store %c0_i32 to %tmpreg[port 1][%0] at %ti 
      : !hir.memref<(bank 1)xi32> delay 1
    hir.store %c0_i32 to %yreg[port 1][%0] at %ti 
      : !hir.memref<(bank 1)xi32> delay 1
    
    %i_i3 = comb.extract %i from 0 :(i5)->(i3)
    %tf=hir.for %j :i5 = %c0_i5  to %c8_i5  step %c1_i5  iter_time(%tj = %ti){
        %j_i3 = comb.extract %j from 0 :(i5)->(i3)
        %a_i_j = hir.load %A[port 0][%i_i3,%j_i3] at %tj
        : !hir.memref<8x8xi32> delay 1
        %b_i_j = hir.load %B[port 0][%i_i3,%j_i3] at %tj
        : !hir.memref<8x8xi32> delay 1
        %x_j = hir.load %X[port 0][%j_i3] at %tj
        : !hir.memref<8xi32> delay 1

        %t1 = hir.call "mul_0" @i32Multiplier(%a_i_j,%x_j) at %tj+1 
          : !hir.func<(i32,i32)->(i32 delay 4)>
        %tmp_in = hir.load %tmpreg[port 0][%0] at %tj + 5
          : !hir.memref<(bank 1)xi32> delay 0
        %tmp_next = comb.add %t1, %tmp_in :i32
        hir.store %tmp_next to %tmpreg[port 1][%0] at %tj+5 
          : !hir.memref<(bank 1)xi32> delay 1

        %t2 = hir.call "mul_1" @i32Multiplier(%b_i_j,%x_j) at %tj+1
          : !hir.func<(i32,i32)->(i32 delay 4)>
        %y = hir.load %yreg[port 0][%0] at %tj + 5
          :!hir.memref<(bank 1)xi32> delay 0
        %y_next = comb.add %t2, %y :i32
        hir.store %y_next to %yreg[port 1][%0] at %tj+5 
          : !hir.memref<(bank 1)xi32> delay 1
        hir.next_iter at %tj + 1
    }
    %tmp_in = hir.load %tmpreg[port 0][%0] at %tf + 5
      :!hir.memref<(bank 1)xi32> delay 0
    %i_i3_delayed = hir.delay %i_i3 by 13 at %ti : i3
    hir.store %tmp_in to %tmp[port 0][%i_i3_delayed] at %tf + 5 
      : !hir.memref<8xi32> delay 1
    %y = hir.load %yreg[port 0][%0] at %tf + 5
      :!hir.memref<(bank 1)xi32> delay 0
    %alpha_tmp = hir.call "mul_2" @i32Multiplier(%alpha,%tmp_in) at %tf+5
      : !hir.func<(i32,i32)->(i32 delay 4)>
    %beta_y = hir.call "mul_3" @i32Multiplier(%beta,%y) at %tf+5
      : !hir.func<(i32,i32)->(i32 delay 4)>
    %y_next = comb.add %alpha_tmp, %beta_y : i32 

    %i9 = hir.delay %i_i3 by 17 at %ti : i3
    hir.store %y_next to %Y[port 0][%i9] at %tf + 9
      : !hir.memref<8xi32> delay 1

    hir.next_iter at %tf + 5
  }

  hir.return
}{argNames =["alpha","beta","tmp","A","B","X","Y","t"]}
