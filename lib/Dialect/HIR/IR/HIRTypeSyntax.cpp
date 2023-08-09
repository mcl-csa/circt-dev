#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
using namespace circt;
using namespace hir;

//------------------------------------------------------------------------------
// Helper functions.
//------------------------------------------------------------------------------

static Optional<DictionaryAttr> parseDelayAttr(AsmParser &parser) {
  IntegerAttr delayAttr;
  auto *context = parser.getBuilder().getContext();
  if (succeeded(parser.parseOptionalKeyword("delay"))) {
    if (parser.parseAttribute(delayAttr, IntegerType::get(context, 64)))
      return llvm::None;
  } else {
    delayAttr = helper::getI64IntegerAttr(parser.getContext(), 0);
  }
  return helper::getDictionaryAttr("hir.delay", delayAttr);
}

static Optional<DictionaryAttr> parseMemrefPortsAttr(AsmParser &parser) {
  ArrayAttr memrefPortsAttr;
  if (parser.parseKeyword("ports") || parser.parseAttribute(memrefPortsAttr))
    return llvm::None;
  return helper::getDictionaryAttr("hir.memref.ports", memrefPortsAttr);
}

static Optional<DictionaryAttr> parseBusPortsAttr(AsmParser &parser) {

  if (parser.parseKeyword("ports") || parser.parseLSquare())
    return llvm::None;
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword("send")))
    keyword = "send";
  else if (succeeded(parser.parseOptionalKeyword("recv")))
    keyword = "recv";
  else {
    parser.emitError(parser.getCurrentLocation())
        << "Expected 'send' or 'recv' keyword.";
    return llvm::None;
  }

  if (parser.parseRSquare())
    return llvm::None;

  return helper::getDictionaryAttr(
      "hir.bus.ports", ArrayAttr::get(parser.getContext(),
                                      SmallVector<Attribute>({StringAttr::get(
                                          parser.getContext(), keyword)})));
}

static Optional<DictionaryAttr> parseArgAttr(AsmParser &parser, Type type) {
  if (helper::isBuiltinSizedType(type))
    return parseDelayAttr(parser);

  if (type.dyn_cast<hir::MemrefType>())
    return parseMemrefPortsAttr(parser);

  if (helper::isBusLikeType(type))
    return parseBusPortsAttr(parser);

  return helper::getDictionaryAttr(parser.getContext());
}

void printArgAttr(AsmPrinter &printer, DictionaryAttr attr, Type type) {
  if (helper::isBuiltinSizedType(type))
    printer << "delay " << helper::getHIRDelayAttr(attr);

  if (type.dyn_cast<hir::MemrefType>())
    printer << "ports " << helper::extractMemrefPortsFromDict(attr);

  if (helper::isBusLikeType(type)) {
    printer << "ports [" << helper::extractBusPortFromDict(attr) << "]";
  }
}

//---------------------------------------------------------------------------//
//-------------------------ASSEMBLY-FORMAT Custom Parser/Printer-------------//
//---------------------------------------------------------------------------//

ParseResult
parseTypedArgList(AsmParser &parser, FailureOr<SmallVector<Type>> &argTypes,
                  FailureOr<SmallVector<DictionaryAttr>> &argAttrs) {
  SmallVector<Type> argTypesStorage;
  SmallVector<DictionaryAttr> argAttrsStorage;
  if (parser.parseLParen())
    return failure();

  if (succeeded(parser.parseOptionalRParen())) {
    argTypes = argTypesStorage;
    argAttrs = argAttrsStorage;
    return success();
  }

  do {
    Type type;
    if (parser.parseType(type))
      return failure();
    argTypesStorage.push_back(type);
    auto inputAttr = parseArgAttr(parser, type);
    if (!inputAttr.hasValue())
      return failure();
    argAttrsStorage.push_back(*inputAttr);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen())
    return failure();
  argTypes = argTypesStorage;
  argAttrs = argAttrsStorage;
  return success();
}

void printTypedArgList(AsmPrinter &printer, ArrayRef<Type> argTypes,
                       ArrayRef<DictionaryAttr> argAttrs) {

  printer << "(";
  for (size_t i = 0; i < argTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << argTypes[i] << " ";
    printArgAttr(printer, argAttrs[i], argTypes[i]);
  }
  printer << ")";
}

ParseResult
parseBankedDimensionList(AsmParser &parser,
                         mlir::FailureOr<SmallVector<int64_t>> &shape,
                         mlir::FailureOr<SmallVector<hir::DimKind>> &dimKinds) {
  SmallVector<int64_t> shapeStorage;
  SmallVector<DimKind> dimKindsStorage;
  while (true) {
    int dimSize;
    // try to parse an ADDR dimension.
    mlir::OptionalParseResult result = parser.parseOptionalInteger(dimSize);
    if (result.hasValue()) {
      if (failed(result.getValue()))
        return failure();
      if (parser.parseXInDimensionList())
        return failure();
      shapeStorage.push_back(dimSize);
      dimKindsStorage.push_back(ADDR);
      continue;
    }

    // try to parse a BANK dimension.
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseKeyword("bank") || parser.parseInteger(dimSize) ||
          parser.parseRParen())
        return failure();
      if (parser.parseXInDimensionList())
        return failure();
      shapeStorage.push_back(dimSize);
      dimKindsStorage.push_back(BANK);
      continue;
    }

    // else the dimension list is over.
    break;
  }
  shape = shapeStorage;
  dimKinds = dimKindsStorage;
  return success();
}

void printBankedDimensionList(AsmPrinter &printer, ArrayRef<int64_t> shape,
                              ArrayRef<hir::DimKind> dimKinds) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == DimKind::BANK)
      printer << "(bank ";
    printer << shape[i];
    if (dimKinds[i] == DimKind::BANK)
      printer << ")";
    printer << "x";
  }
}

ParseResult parseDimensionList(AsmParser &parser,
                               FailureOr<SmallVector<int64_t>> &shape) {
  SmallVector<int64_t> shapeVec;
  if (parser.parseDimensionList(shapeVec))
    return failure();
  shape = shapeVec;
  return success();
}

void printDimensionList(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  for (auto d : shape)
    printer << d << "x";
}