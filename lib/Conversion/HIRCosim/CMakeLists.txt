add_circt_library(CIRCTHIRCosim
  GenCosimFiles.cpp
  CPUModuleBuilder.cpp
  CosimInfo.cpp 

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTHIR
  CIRCTHIRAnalysis
)
