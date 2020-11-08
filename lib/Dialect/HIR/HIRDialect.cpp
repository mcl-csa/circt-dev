//=========- HIR.cpp - Registration, Parser & Printer----------------------===//
//
// This file implements parsers and printers for Types and registers the types
// and operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/HIRDialect.h"
#include "circt/Dialect/HIR/HIR.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace hir;

HIRDialect::HIRDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<TimeType, StreamType, ConstType, MemrefType, WireType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/HIR.cpp.inc"
      >();
}

namespace Helpers {
static ParseResult parsePortKind(DialectAsmParser &parser,
                                 Details::PortKind &port) {
  if (!parser.parseOptionalKeyword("rw"))
    port = Details::rw;
  else if (!parser.parseOptionalKeyword("r"))
    port = Details::r;
  else if (!parser.parseOptionalKeyword("w"))
    port = Details::w;
  else
    return failure();
  return success();
}

static OptionalParseResult
parseOptionalMemrefPacking(DialectAsmParser &parser,
                           SmallVectorImpl<unsigned> &packing) {
  if (parser.parseOptionalKeyword("packing"))
    return OptionalParseResult(None);
  if (parser.parseEqual() || parser.parseLSquare())
    return failure();
  if (!parser.parseOptionalRSquare())
    return success();

  unsigned dim;
  if (parser.parseInteger(dim))
    return failure();
  packing.push_back(dim);
  while (!parser.parseOptionalComma()) {
    if (parser.parseInteger(dim))
      return failure();
    packing.push_back(dim);
  }
  if (parser.parseRSquare())
    return failure();
  return success();
}

static ParseResult parseShapedType(DialectAsmParser &parser,
                                   SmallVectorImpl<unsigned> &shape,
                                   Type &elementType) {
  unsigned dim;

  if (parser.parseInteger(dim))
    return failure();
  shape.push_back(dim);
  if (parser.parseOptionalStar())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected '*' followed by type"),
           failure();
  while (true) {
    auto optionalParseRes = parser.parseOptionalInteger(dim);
    if (!optionalParseRes.hasValue())
      break;
    if (optionalParseRes.getValue())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected integer literal or type"),
             failure();
    shape.push_back(dim);
    if (parser.parseOptionalStar())
      return parser.emitError(parser.getCurrentLocation(),
                              "expected '*' followed by type"),
             failure();
  }
  if (parser.parseType(elementType))
    return failure();
}
}; // namespace Helpers.

// Types.

// Memref Type.
static Type parseMemrefType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<unsigned, 4> shape;
  Type elementType;
  hir::Details::PortKind default_port = hir::Details::rw;
  if (parser.parseLess())
    return Type();

  if (Helpers::parseShapedType(parser, shape, elementType))
    return Type();

  SmallVector<unsigned, 4> default_packing;
  // default packing is [0,1,2,3] for shape = [d3,d2,d1,d0] i.e. d0 is fastest
  // changing dim in linear index and no distributed dimensions.
  for (int i = 0; i < shape.size(); i++)
    default_packing.push_back(i);

  if (!parser.parseOptionalGreater())
    return MemrefType::get(context, shape, elementType, default_packing,
                           default_port);

  if (parser.parseComma())
    return Type();

  SmallVector<unsigned, 4> packing;
  OptionalParseResult hasPacking =
      Helpers::parseOptionalMemrefPacking(parser, packing);

  if (hasPacking.hasValue()) {
    if (hasPacking.getValue())
      return Type();
    if (!parser.parseOptionalGreater())
      return MemrefType::get(context, shape, elementType, packing,
                             default_port);
    if (parser.parseComma())
      return Type();
  }

  hir::Details::PortKind port;
  if (Helpers::parsePortKind(parser, port) || parser.parseGreater())
    return Type();

  return MemrefType::get(context, shape, elementType,
                         hasPacking.hasValue() ? packing : default_packing,
                         port);
}

static void printMemrefType(MemrefType memrefTy, DialectAsmPrinter &printer) {
  printer << memrefTy.getKeyword();
  printer << "<";
  auto shape = memrefTy.getShape();
  auto elementType = memrefTy.getElementType();
  auto packing = memrefTy.getPacking();
  auto port = memrefTy.getPort();
  for (auto dim : shape)
    printer << dim << "*";

  printer << elementType << ", packing = [";

  for (int i = 0; i < packing.size(); i++)
    printer << packing[i] << ((i == (packing.size() - 1)) ? "" : ", ");
  printer << "]";

  if (port == Details::r) {
    printer << ", r";
  } else if (port == Details::w) {
    printer << ", w";
  } else if (port == Details::rw) {
    printer << "";
  } else {
    printer << ", unknown";
  }
  printer << ">";
}

// Stream Type.
static Type parseStreamType(DialectAsmParser &parser, MLIRContext *context) {
  Type elementType;
  hir::Details::PortKind default_port = hir::Details::rw;
  hir::Details::PortKind port;
  if (parser.parseLess())
    return Type();

  if (parser.parseType(elementType))
    return Type();

  if (parser.parseComma())
    return Type();

  if (Helpers::parsePortKind(parser, port))
    return Type();
  if (parser.parseGreater())
    return Type();

  return StreamType::get(context, elementType, port);
}

static void printStreamType(StreamType streamType, DialectAsmPrinter &printer) {
  auto elementType = streamType.getElementType();
  auto port = streamType.getPort();
  printer << streamType.getKeyword();
  printer << "<";
  printer << elementType << ", ";

  if (port == Details::r) {
    printer << ", r";
  } else if (port == Details::w) {
    printer << ", w";
  } else if (port == Details::rw) {
    printer << "";
  } else {
    printer << ", unknown";
  }
  printer << ">";
}

// WireType.
static Type parseWireType(DialectAsmParser &parser, MLIRContext *context) {
  SmallVector<unsigned, 4> shape;
  Type elementType;
  hir::Details::PortKind default_port = hir::Details::rw;
  if (parser.parseLess())
    return Type();

  if (Helpers::parseShapedType(parser, shape, elementType))
    return Type();

  SmallVector<unsigned, 4> default_packing;
  // default packing is [0,1,2,3] for shape = [d3,d2,d1,d0] i.e. d0 is fastest
  // changing dim in linear index and no distributed dimensions.
  for (int i = 0; i < shape.size(); i++)
    default_packing.push_back(i);

  if (!parser.parseOptionalGreater())
    return WireType::get(context, shape, elementType, default_port);

  if (parser.parseComma())
    return Type();

  hir::Details::PortKind port;
  if (Helpers::parsePortKind(parser, port) || parser.parseGreater())
    return Type();

  return WireType::get(context, shape, elementType, port);
}

static void printWireType(WireType wireTy, DialectAsmPrinter &printer) {
  printer << wireTy.getKeyword();
  printer << "<";
  auto shape = wireTy.getShape();
  auto elementType = wireTy.getElementType();
  auto port = wireTy.getPort();
  for (auto dim : shape)
    printer << dim << "*";

  printer << elementType;

  if (port == Details::r) {
    printer << ", r";
  } else if (port == Details::w) {
    printer << ", w";
  } else if (port == Details::rw) {
    printer << "";
  } else {
    printer << ", unknown";
  }
  printer << ">";
}

// parseType and printType.
Type HIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();

  if (typeKeyword == TimeType::getKeyword())
    return TimeType::get(getContext());

  if (typeKeyword == MemrefType::getKeyword())
    return parseMemrefType(parser, getContext());

  if (typeKeyword == WireType::getKeyword())
    return parseWireType(parser, getContext());

  if (typeKeyword == ConstType::getKeyword())
    return ConstType::get(getContext());

  if (typeKeyword == StreamType::getKeyword())
    return parseStreamType(parser, getContext());

  return parser.emitError(parser.getNameLoc(), "unknown hir type"), Type();
}

void HIRDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (TimeType timeTy = type.dyn_cast<TimeType>()) {
    printer << timeTy.getKeyword();
    return;
  }
  if (MemrefType memrefTy = type.dyn_cast<MemrefType>()) {
    printMemrefType(memrefTy, printer);
    return;
  }
  if (WireType wireTy = type.dyn_cast<WireType>()) {
    printWireType(wireTy, printer);
    return;
  }
  if (ConstType constTy = type.dyn_cast<ConstType>()) {
    printer << constTy.getKeyword();
    return;
  }
}
