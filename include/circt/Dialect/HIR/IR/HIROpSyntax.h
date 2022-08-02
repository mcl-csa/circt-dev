#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace hir;
using namespace llvm;

// parser and printer for Time and offset.
ParseResult parseTimeAndOffset(mlir::OpAsmParser &,
                               OpAsmParser::UnresolvedOperand &, IntegerAttr &);

void printTimeAndOffset(OpAsmPrinter &, Operation *op, Value, IntegerAttr);

// parser and printer for array address types.

ParseResult parseOptionalArrayAccessTypes(mlir::OpAsmParser &parser,
                                          ArrayAttr &constAddrs,
                                          SmallVectorImpl<Type> &varAddrTypes);

void printOptionalArrayAccessTypes(OpAsmPrinter &printer, Operation *op,
                                   ArrayAttr constAddrs,
                                   TypeRange varAddrTypes);

// parser and printer for array address indices.
ParseResult parseOptionalArrayAccess(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &varAddrs,
    ArrayAttr &constAddrs);

void printOptionalArrayAccess(OpAsmPrinter &printer, Operation *op,
                              OperandRange varAddrs, ArrayAttr constAddrs);

ParseResult parseMemrefAndElementType(OpAsmParser &, Type &,
                                      SmallVectorImpl<Type> &, Type &);

void printMemrefAndElementType(OpAsmPrinter &, Operation *, Type, TypeRange,
                               Type);

ParseResult parseTypeAndDelayList(mlir::OpAsmParser &, SmallVectorImpl<Type> &,
                                  ArrayAttr &);

void printTypeAndDelayList(mlir::OpAsmPrinter &, TypeRange, ArrayAttr);

ParseResult parseBinOpOperandsAndResultType(mlir::OpAsmParser &parser,
                                            Type &resultTy, Type &op1Ty,
                                            Type &op2Ty);

void printBinOpOperandsAndResultType(mlir::OpAsmPrinter &, Operation *, Type,
                                     Type, Type);

ParseResult parseCopyType(mlir::OpAsmParser &, Type &, Type);
void printCopyType(mlir::OpAsmPrinter &, Operation *, Type, Type);
ParseResult parseCopyType(mlir::OpAsmParser &parser, Type &, TypeAttr);
void printCopyType(mlir::OpAsmPrinter &, Operation *, Type, TypeAttr);

ParseResult parseWithSSANames(mlir::OpAsmParser &, mlir::NamedAttrList &);
void printWithSSANames(mlir::OpAsmPrinter &, Operation *, mlir::DictionaryAttr);

ParseResult parseFunctionType(OpAsmParser &parser, TypeAttr &funcTyAttr,
                              SmallVectorImpl<Type> &operandTypes,
                              SmallVectorImpl<Type> &resultTypes);

void printFunctionType(OpAsmPrinter &printer, Operation *, TypeAttr funcTyAttr,
                       TypeRange, TypeRange);