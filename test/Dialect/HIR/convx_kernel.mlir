#bram_r = {"rd_latency"=1}
#reg_r = {"rd_latency"=0}
#bram_w = {"wr_latency"=1}
#reg_w = {"wr_latency"=1}

hir.func.extern @mul_f32 at %t(%a:f32, %b:f32) ->(%out:f32 delay 4)
hir.func.extern @add_f32 at %t(%a:f32, %b:f32) ->(%out:f32 delay 3)

func @convX(%arg0: memref<32x32xf32> {hir.memref.ports = [#bram_r,#bram_w]}, 
            %arg1: memref<32x32xf32> {hir.memref.ports = [#bram_r]}, 
            %arg2: memref<32xf32>{hir.memref.ports = [#bram_r]}) ->(f32 {hir.delay = 2} ) 
            attributes {llvm.linkage = #llvm.linkage<external>, hwAccel} {
  %cst = arith.constant 0.000000e+00: f32 
  affine.for %arg3 = 0 to 27 {
    affine.for %arg4 = 0 to 27 {
      affine.store %cst, %arg0[%arg3, %arg4] : memref<32x32xf32>
      affine.for %arg5 = 0 to 5 {
        %0 = affine.load %arg2[%arg5] {result_delays=[1]}: memref<32xf32>
        %1 = affine.load %arg1[%arg3, %arg4 + %arg5] {result_delays=[1]}: memref<32x32xf32>
        %2 = arith.mulf %0, %1 {result_delays=[4], hir_function=@mul_f32}: f32
        %3 = affine.load %arg0[%arg3, %arg4] {result_delays=[1]}: memref<32x32xf32>
        %4 = arith.addf %3, %2 {result_delays=[3], hir_function=@add_f32}: f32
        affine.store %4, %arg0[%arg3, %arg4] : memref<32x32xf32>
      }{II=5}
    }{II=10}
  }{II=10}
  return %cst :f32
}

