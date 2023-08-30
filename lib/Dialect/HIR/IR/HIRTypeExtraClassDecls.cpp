#include "circt/Dialect/HIR/IR/HIR.h"
using namespace circt;
using namespace hir;

int64_t MemrefType::getNumElementsPerBank() {
  int count = 1;
  auto dimKinds = getDimKinds();
  auto shape = getShape();
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == ADDR)
      count *= shape[i];
  }
  return count;
}

int64_t MemrefType::getNumBanks() {
  int64_t count = 1;
  auto dimKinds = getDimKinds();
  auto shape = getShape();
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == BANK) {
      count *= shape[i];
    }
  }
  return count;
}

SmallVector<int64_t> MemrefType::filterShape(DimKind dimKind) {
  auto shape = getShape();
  auto dimKinds = getDimKinds();
  SmallVector<int64_t> bankShape;
  for (size_t i = 0; i < getShape().size(); i++) {
    if (dimKinds[i] == dimKind)
      bankShape.push_back(shape[i]);
  }
  return bankShape;
}

FunctionType FuncType::getFunctionType() {
  SmallVector<Type> functionArgTypes;
  for (auto ty : getInputTypes())
    functionArgTypes.push_back(ty);
  functionArgTypes.push_back(hir::TimeType::get(getContext()));

  return FunctionType::get(getContext(), functionArgTypes, getResultTypes());
}

unsigned int FuncType::getNumInputs() { return getInputTypes().size(); }
unsigned int FuncType::getNumResults() { return getResultTypes().size(); }

Type FuncType::getInputType(unsigned int i) { return getInputTypes()[i]; }
DictionaryAttr FuncType::getInputAttr(unsigned int i) {
  return getInputAttrs()[i];
}
Type FuncType::getResultType(unsigned int i) { return getResultTypes()[i]; }
DictionaryAttr FuncType::getResultAttr(unsigned int i) {
  return getResultAttrs()[i];
}

size_t BusTensorType::getNumElements() {
  size_t numElements = 1;
  for (auto dim : getShape())
    numElements *= dim;
  return numElements;
}