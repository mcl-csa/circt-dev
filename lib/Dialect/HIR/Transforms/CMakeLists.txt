add_circt_dialect_library(CIRCTHIRTransforms
  FuseHWInstPass.cpp
  OptBitWidthPass.cpp
  OptDelayPass.cpp
  OptTimePass.cpp
  LoopUnrollPass.cpp
  MemrefLoweringPass.cpp
  MemrefLoweringUtils.cpp
  PassPipelines.cpp
  SimplifyCtrl.cpp
  SimplifyCtrlUtils.cpp
  VerifySchedulePass.cpp
  PassRegistration.cpp
  OpFusionPass.cpp

  DEPENDS
  CIRCTHIR
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  CIRCTHIRAnalysis
  )
