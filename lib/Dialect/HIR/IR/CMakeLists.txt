add_circt_dialect_library(CIRCTHIR
  helper.cpp  
  HIROpSyntax.cpp  
  HIRDialect.cpp  
  HIROpInterfaces.cpp
  HIR.cpp  
  HIROpVerifier.cpp  
  HIROpCanonicalizer.cpp
  HIRTypeSyntax.cpp
  HIRTypeExtraClassDecls.cpp
  HIRTypeVerifiers.cpp
  HIRBuilders.cpp
  HIROpExtraClassDeclaration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Dialect/HIR

  DEPENDS
  MLIRHIRIncGen
  MLIRHIRInterfacesIncGen
  MLIRHIREnumsIncGen
  MLIRHIRAttrsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  )
