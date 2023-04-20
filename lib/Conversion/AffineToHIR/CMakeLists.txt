add_circt_library(CIRCTAffineToHIR
  AffineToHIR.cpp
  AffineToHIRUtils.cpp
  AutoAffineToHIRPass.cpp
  HIRPragma.cpp
  #OpFusionAnalysis.cpp
  SchedulingAnalysis.cpp 
  SchedulingUtils.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTHIR
  CIRCTHIRAnalysis
)
