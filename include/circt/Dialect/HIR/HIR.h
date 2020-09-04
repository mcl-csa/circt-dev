#ifndef HIR_HIR_H
#define HIR_HIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace llvm;

namespace mlir {
namespace hir {
/// Defines the kind corresponding to the type. So MemrefType has Kind
/// MemrefKind
enum Kinds {
  TimeKind = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  ConstKind,
  MemrefKind,
  WireKind
};

namespace Details {
/// PortKind tells what type of port this is. r => read port, w => write port
/// and rw => read-write port.
enum PortKind { r = 0, w = 1, rw = 2 };

/// Storage class for ConstType.
struct ConstTypeStorage : public TypeStorage {
  ConstTypeStorage(Type elementType) : elementType(elementType) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = Type;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  /// Define a construction method for creating a new instance of this storage.
  static ConstTypeStorage *construct(TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ConstTypeStorage>()) ConstTypeStorage(key);
  }

  Type elementType;
};

/// Storage class for MemrefType.
struct MemrefTypeStorage : public TypeStorage {
  MemrefTypeStorage(ArrayRef<unsigned> shape, Type elementType,
                    ArrayRef<unsigned> packing, PortKind port = rw)
      : shape(shape), elementType(elementType), packing(packing), port(port) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy =
      std::tuple<ArrayRef<unsigned>, Type, ArrayRef<unsigned>, PortKind>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementType, packing, port);
  }

  /// Define a hash function for the key type.
  static hash_code hashKey(const KeyTy &key) {
    return hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key),
                        std::get<3>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<unsigned> shape, Type elementType,
                      ArrayRef<unsigned> packing, PortKind port = rw) {
    return KeyTy(shape, elementType, packing, port);
  }

  /// Define a construction method for creating a new instance of this storage.
  static MemrefTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    ArrayRef<unsigned> shape = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    ArrayRef<unsigned> packing = allocator.copyInto(std::get<2>(key));
    Details::PortKind port = std::get<3>(key);
    return new (allocator.allocate<MemrefTypeStorage>())
        MemrefTypeStorage(shape, elementType, packing, port);
  }

  ArrayRef<unsigned> shape;
  ArrayRef<unsigned> packing;
  Type elementType;
  PortKind port;
};

struct WireTypeStorage : public TypeStorage {
  /// Storage class for WireType.
  WireTypeStorage(ArrayRef<unsigned> shape, Type elementType,
                  ArrayRef<unsigned> packing, PortKind port = rw)
      : shape(shape), elementType(elementType), port(port) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::tuple<ArrayRef<unsigned>, Type, PortKind>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(shape, elementType, port);
  }

  /// Define a hash function for the key type.
  static hash_code hashKey(const KeyTy &key) {
    return hash_combine(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  }

  /// Define a construction function for the key type.
  static KeyTy getKey(ArrayRef<unsigned> shape, Type elementType,
                      ArrayRef<unsigned> packing, PortKind port = rw) {
    return KeyTy(shape, elementType, port);
  }

  /// Define a construction method for creating a new instance of this storage.
  static WireTypeStorage *construct(TypeStorageAllocator &allocator,
                                    const KeyTy &key) {
    ArrayRef<unsigned> shape = allocator.copyInto(std::get<0>(key));
    Type elementType = std::get<1>(key);
    Details::PortKind port = std::get<2>(key);
    return new (allocator.allocate<WireTypeStorage>())
        WireTypeStorage(shape, elementType, port);
  }

  ArrayRef<unsigned> shape;
  Type elementType;
  PortKind port;
};
} // namespace Details.

/// This class defines hir.time type in the dialect.
class TimeType : public Type::TypeBase<TimeType, Type, DefaultTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TimeKind; }
  static StringRef getKeyword() { return "time"; }
  static TimeType get(MLIRContext *context) {
    return Base::get(context, TimeKind);
  }
};

/// This class defines hir.const type in the dialect.
class ConstType
    : public Type::TypeBase<ConstType, Type, Details::ConstTypeStorage> {
public:
  using Base::Base;
  static bool kindof(unsigned kind) { return kind == ConstKind; }
  static StringRef getKeyword() { return "const"; }
  static ConstType get(MLIRContext *context, Type elementType) {
    return Base::get(context, ConstKind, elementType);
  }
  Type getElementType() { return getImpl()->elementType; }
};

/// This class defines hir.memref type in the dialect.
class MemrefType
    : public Type::TypeBase<MemrefType, Type, Details::MemrefTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == MemrefKind; }
  static StringRef getKeyword() { return "memref"; }
  static MemrefType get(MLIRContext *context, ArrayRef<unsigned> shape,
                        Type elementType, ArrayRef<unsigned> packing,
                        Details::PortKind port) {
    return Base::get(context, MemrefKind, shape, elementType, packing, port);
  }
  ArrayRef<unsigned> getShape() { return getImpl()->shape; }
  Type getElementType() { return getImpl()->elementType; }
  ArrayRef<unsigned> getPacking() { return getImpl()->packing; }
  Details::PortKind getPort() { return getImpl()->port; }
};

/// This class defines hir.wire type in the dialect.
class WireType
    : public Type::TypeBase<WireType, Type, Details::WireTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == WireKind; }
  static StringRef getKeyword() { return "wire"; }
  static WireType get(MLIRContext *context, ArrayRef<unsigned> shape,
                      Type elementType, Details::PortKind port) {
    return Base::get(context, WireKind, shape, elementType, port);
  }
  ArrayRef<unsigned> getShape() { return getImpl()->shape; }
  Type getElementType() { return getImpl()->elementType; }
  Details::PortKind getPort() { return getImpl()->port; }
};

#define GET_OP_CLASSES
#include "circt/Dialect/HIR/HIR.h.inc"

} // namespace hir.
} // namespace mlir.

#endif // HIR_HIR_H.
