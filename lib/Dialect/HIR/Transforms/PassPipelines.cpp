#include "circt/Dialect/HIR/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void circt::hir::registerPassPipelines() {
  mlir::PassPipelineRegistration<>(
      "hir-opt", "Optimize HIR dialect.", [](mlir::OpPassManager &pm) {
        pm.addPass(circt::hir::createOptTimePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(circt::hir::createOptBitWidthPass());
      });

  mlir::PassPipelineRegistration<>(
      "hir-simplify",
      "Simplify HIR dialect to a bare minimum for lowering to verilog.",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(circt::hir::createLoopUnrollPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(hir::createSimplifyCtrlPass());
        pm.addPass(mlir::createSCCPPass());
        pm.addPass(circt::hir::createMemrefLoweringPass());
        pm.addPass(mlir::createSCCPPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });
}
