add_circt_library(CIRCTAffineToHIR
  AutoAffineToHIRPass.cpp
  AffineToHIR.cpp
  AffineToHIRUtils.cpp
  SchedulingAnalysis.cpp 
  SchedulingUtils.cpp
  HIRPragma.cpp
  

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTHIR
  CIRCTHIRAnalysis
)
