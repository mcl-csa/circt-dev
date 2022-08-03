#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace hir {
enum PortKind { rd = 0, wr = 1, rw = 2 };
enum BusDirection { SAME = 0, FLIPPED = 1 };
enum DimKind { ADDR = 0, BANK = 1 };
} // namespace hir
} // namespace circt

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/HIR/IR/HIRTypes.h.inc"

mlir::ParseResult parseBankedDimensionList(
    mlir::AsmParser &parser, mlir::FailureOr<mlir::SmallVector<int64_t>> &shape,
    mlir::FailureOr<mlir::SmallVector<circt::hir::DimKind>> &dimKinds);

void printBankedDimensionList(mlir::AsmPrinter &printer,
                              mlir::ArrayRef<int64_t> shape,
                              mlir::ArrayRef<circt::hir::DimKind> dimKinds);

mlir::ParseResult parseTypedArgList(
    mlir::AsmParser &parser,
    mlir::FailureOr<mlir::SmallVector<mlir::Type>> &argTypes,
    mlir::FailureOr<mlir::SmallVector<mlir::DictionaryAttr>> &argAttrs);

void printTypedArgList(mlir::AsmPrinter &printer,
                       mlir::ArrayRef<mlir::Type> argTypes,
                       mlir::ArrayRef<mlir::DictionaryAttr> argAttrs);

mlir::ParseResult
parseDimensionList(mlir::AsmParser &parser,
                   mlir::FailureOr<mlir::SmallVector<int64_t>> &shape);
void printDimensionList(mlir::AsmPrinter &printer,
                        mlir::ArrayRef<int64_t> shape);

namespace circt {
namespace hir {
namespace Details {
/// Storage class for BusType.
// struct BusTypeStorage : public TypeStorage {
//  BusTypeStorage(ArrayRef<StringAttr> memberNames, ArrayRef<Type>
//  memberTypes,
//                 ArrayRef<BusDirection> memberDirections)
//      : memberNames(memberNames), memberTypes(memberTypes),
//        memberDirections(memberDirections) {}
//
//  using KeyTy =
//      std::tuple<ArrayRef<StringAttr>, ArrayRef<Type>,
//      ArrayRef<BusDirection>>;
//
//  /// Define the comparison function for the key type.
//  bool operator==(const KeyTy &key) const {
//    return key == KeyTy(memberNames, memberTypes, memberDirections);
//  }
//
//  /// Define a hash function for the key type.
//  static llvm::hash_code hashKey(const KeyTy &key) {
//    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
//                              std::get<2>(key));
//  }
//
//  /// Define a construction function for the key type.
//  static KeyTy getKey(ArrayRef<StringAttr> memberNames,
//                      ArrayRef<Type> memberTypes,
//                      ArrayRef<BusDirection> memberDirections) {
//    return KeyTy(memberNames, memberTypes, memberDirections);
//  }
//
//  /// Define a construction method for creating a new instance of this
//  /// storage.
//  static BusTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
//                                   const KeyTy &key) {
//    auto memberNames = allocator.copyInto(std::get<0>(key));
//    auto memberTypes = allocator.copyInto(std::get<1>(key));
//    auto memberDirections = allocator.copyInto(std::get<2>(key));
//    return new (allocator.allocate<BusTypeStorage>())
//        BusTypeStorage(memberNames, memberTypes, memberDirections);
//  }
//
//  ArrayRef<StringAttr> memberNames;
//  ArrayRef<Type> memberTypes;
//  ArrayRef<BusDirection> memberDirections;
//};

/// Storage class for BusTensorType.
// struct BusTensorTypeStorage : public TypeStorage {
//  BusTensorTypeStorage(ArrayRef<int64_t> shape, Type elementTy)
//      : shape(shape), elementTy(elementTy) {}
//
//  using KeyTy = std::tuple<ArrayRef<int64_t>, Type>;
//
//  bool operator==(const KeyTy &key) const {
//    return key == KeyTy(shape, elementTy);
//  }
//
//  static llvm::hash_code hashKey(const KeyTy &key) {
//    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
//  }
//
//  /// Define a construction function for the key type.
//  static KeyTy getKey(ArrayRef<int64_t> shape, Type elementTy) {
//    return KeyTy(shape, elementTy);
//  }

/// Define a construction method for creating a new instance of this
/// storage.
// static BusTensorTypeStorage *construct(mlir::TypeStorageAllocator
// &allocator,
//                                       const KeyTy &key) {
//  auto shape = allocator.copyInto(std::get<0>(key));
//  auto elementTy = std::get<1>(key);
//  return new (allocator.allocate<BusTensorTypeStorage>())
//      BusTensorTypeStorage(shape, elementTy);
//}
//
// ArrayRef<int64_t> shape;
// Type elementTy;
//};
} // namespace Details.

/// This class defines a bus type.
// class BusType : public Type::TypeBase<BusType, Type,
// Details::BusTypeStorage> { public:
//  using Base::Base;
//
//  static StringRef getKeyword() { return "bus"; }
//  static BusType get(MLIRContext *context, Type type) {
//
//    return Base::get(context, StringAttr::get(context, "bus"), type,
//                     hir::BusDirection::SAME);
//  }
//
//  Type getElementType() { return getImpl()->memberTypes[0]; }
//  Type parse(mlir::AsmParser &);
//  void print(mlir::AsmPrinter &);
//};

/// This class defines a bus_tensor type.
// class BusTensorType : public Type::TypeBase<BusTensorType, Type,
//                                            Details::BusTensorTypeStorage> {
// public:
//  using Base::Base;
//
//  static StringRef getKeyword() { return "bus_tensor"; }
//  static BusTensorType get(MLIRContext *context, ArrayRef<int64_t> shape,
//                           Type type) {
//
//    return Base::get(context, shape, type);
//  }
//
//  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
//                              ArrayRef<int64_t> shape, Type elementTy);
//  Type getElementType() { return getImpl()->elementTy; }
//  ArrayRef<int64_t> getShape() { return getImpl()->shape; }
//  size_t getNumElements() {
//    size_t numElements = 1;
//    for (auto dim : getShape())
//      numElements *= dim;
//    return numElements;
//  }
//  Type parse(mlir::AsmParser &);
//  void print(mlir::AsmPrinter &);
//};

struct Time {
  Time() {}
  Time(Value timeVar, int64_t offset) : timeVar(timeVar), offset(offset) {
    assert(timeVar.getType().isa<hir::TimeType>());
    assert(offset >= 0);
  }

  Value getTimeVar() const { return timeVar; }
  int64_t getOffset() const { return offset; }
  Time addOffset(int64_t extraOffset) const {
    auto newOffset = offset + extraOffset;
    assert(newOffset >= 0);
    return Time(timeVar, newOffset);
  }
  bool operator==(Time const &rhs) {
    assert(timeVar);
    return (timeVar == rhs.getTimeVar()) && (offset == rhs.getOffset());
  }
  bool operator!=(Time const &rhs) {
    assert(timeVar);
    return !(*this == rhs);
  }
  llvm::Optional<bool> operator<(Time const &rhs) {
    assert(timeVar);
    return (timeVar == rhs.getTimeVar()) && (this->offset < rhs.getOffset());
  }

private:
  Value timeVar;
  int64_t offset;
};
} // namespace hir.
} // namespace circt

#include "circt/Dialect/HIR/IR/HIROpInterfaces.h.inc"
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HIR/IR/HIRAttrs.h.inc"
#define GET_OP_CLASSES
#include "circt/Dialect/HIR/IR/HIR.h.inc"
#endif // HIR_HIR_H.
