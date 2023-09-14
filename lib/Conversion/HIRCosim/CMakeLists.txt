add_circt_library(CIRCTHIRCosim
  InstrumentCosim.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTHIR
  CIRCTHIRAnalysis
)
