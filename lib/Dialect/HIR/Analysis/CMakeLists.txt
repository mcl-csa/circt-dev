add_circt_dialect_library(CIRCTHIRAnalysis
  TimingInfo.cpp

LINK_LIBS PUBLIC
MLIRTransformUtils
CIRCTHIR
)

