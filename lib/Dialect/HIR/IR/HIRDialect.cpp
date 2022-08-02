//=========- HIR.cpp - Registration, Parser & Printer----------------------===//
//
// This file implements parsers and printers for Types and registers the types
// and operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace hir;

//-----------------------------------------------------------------------------
// HIR Dialect interfaces
//-----------------------------------------------------------------------------
namespace {
/// This class defines the interface for handling inlining with HIR operations.
struct HIRInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// This hook checks to see if the given callable operation is legal to inline
  /// into the given call. For Toy this hook can simply return true, as the Toy
  /// Call operation is always inlinable.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// This hook checks to see if the given operation is legal to inline into the
  /// given region. For Toy this hook can simply return true, as all Toy
  /// operations are inlinable.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *src, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "hir.yield" needs to be handled here.
    auto returnOp = cast<hir::ReturnOp>(op);
    assert(returnOp);

    // Replace the values directly with the return operands.
    assert(returnOp.operands().size() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.operands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // end anonymous namespace

//-----------------------------------------------------------------------------
// HIR Dialect
//-----------------------------------------------------------------------------
void HIRDialect::initialize() {
  addTypes<TimeType, BusType, BusTensorType, FuncType, MemrefType>();
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/HIR/IR/HIR.cpp.inc"
      >();
  addInterfaces<HIRInlinerInterface>();
  addAttributes<MemKindEnumAttr>();
}

Operation *HIRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (value.getType().isa<IntegerType>())
    return builder.create<hw::ConstantOp>(loc, type,
                                          value.dyn_cast<IntegerAttr>());
  // For index type.
  return builder.create<circt::hw::ConstantOp>(loc, type,
                                               value.dyn_cast<IntegerAttr>());
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HIR/IR/HIRAttrs.cpp.inc"
#include "circt/Dialect/HIR/IR/HIRDialect.cpp.inc"
#include "circt/Dialect/HIR/IR/HIREnums.cpp.inc"
