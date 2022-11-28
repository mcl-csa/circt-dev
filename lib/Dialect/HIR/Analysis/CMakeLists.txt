add_circt_dialect_library(CIRCTHIRAnalysis
  OpFusionAnalysis.cpp
  TimingInfo.cpp

LINK_LIBS PUBLIC
MLIRTransformUtils
CIRCTHIR
)

target_link_libraries(CIRCTHIRAnalysis PRIVATE -lglpk)
