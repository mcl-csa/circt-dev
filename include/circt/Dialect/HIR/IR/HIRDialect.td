#ifndef HIR_DIALECT 
#define HIR_DIALECT
include "mlir/IR/OpBase.td"

// Dialect
def HIR_Dialect : Dialect {
  let name = "hir";
  let summary = "An MLIR dialect for Hardware description";
  let description = [{
    This dialect is designed for hardware description. A circuit described in HIR 
      dialect contains the schedule of execution and functional unit binding 
      information. HIR supports high level constructs such as loops and 
      conditionals.
  }];
  let cppNamespace = "::circt::hir";
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class HIR_Op<string mnemonic, list<Trait> traits = []>
: Op<HIR_Dialect, mnemonic, traits>;
#endif
