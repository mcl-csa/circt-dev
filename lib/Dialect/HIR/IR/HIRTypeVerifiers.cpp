#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
using namespace circt;
using namespace hir;
//------------------------------------------------------------------------------
// Type members.
//------------------------------------------------------------------------------
LogicalResult
verifyDelayAttribute(mlir::function_ref<InFlightDiagnostic()> emitError,
                     DictionaryAttr attrDict) {
  if (!attrDict)
    return emitError() << "Could not find hir.delay attr";
  auto delayNameAndAttr = attrDict.getNamed("hir.delay");
  if (!delayNameAndAttr)
    return emitError() << "Could not find hir.delay attr";
  if (!delayNameAndAttr->getValue().dyn_cast<IntegerAttr>())
    return emitError() << "hir.delay attr must be IntegerAttr";
  return success();
}

LogicalResult
verifyMemrefPortsAttribute(mlir::function_ref<InFlightDiagnostic()> emitError,
                           DictionaryAttr attrDict) {
  auto memrefPortsNameAndAttr = attrDict.getNamed("hir.memref.ports");
  if (!memrefPortsNameAndAttr)
    return failure();
  if (!memrefPortsNameAndAttr->getValue().dyn_cast<ArrayAttr>())
    return failure();
  return success();
}

LogicalResult
verifyBusPortsAttribute(mlir::function_ref<InFlightDiagnostic()> emitError,
                        DictionaryAttr attrDict) {
  auto memrefPortsNameAndAttr =
      attrDict.getNamed(helper::getHIRBusPortAttrName());
  if (!memrefPortsNameAndAttr)
    return failure();
  if (!memrefPortsNameAndAttr->getValue().dyn_cast<ArrayAttr>())
    return failure();
  return success();
}

LogicalResult MemrefType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 ArrayRef<DimKind> dimKinds) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (dimKinds[i] == ADDR) {
      if ((pow(2, helper::clog2(shape[i]))) != shape[i]) {
        return emitError()
               << "hir.memref dimension sizes must be a power of two, dim " << i
               << " has size " << shape[i];
      }
      if (shape[i] <= 0) {
        return emitError() << "hir.memref dimension size must be >0. dim " << i
                           << " has size " << shape[i];
      }
    }
  }
  return success();
}
LogicalResult FuncType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<Type> inputTypes,
                               ArrayRef<DictionaryAttr> inputAttrs,
                               ArrayRef<Type> resultTypes,
                               ArrayRef<DictionaryAttr> resultAttrs) {
  if (inputAttrs.size() != inputTypes.size())
    return emitError() << "Number of input attributes is not same as number of "
                          "input types.";

  if (resultAttrs.size() != resultTypes.size())
    return emitError()
           << "Number of result attributes is not same as number of "
              "result types.";

  // Verify inputs.
  for (size_t i = 0; i < inputTypes.size(); i++) {
    if (helper::isBuiltinSizedType(inputTypes[i])) {
      if (failed(verifyDelayAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected hir.delay IntegerAttr for input arg"
                           << std::to_string(i) << ".";
    } else if (inputTypes[i].dyn_cast<hir::MemrefType>()) {
      if (failed(verifyMemrefPortsAttribute(emitError, inputAttrs[i])))
        return emitError()
               << "Expected hir.memref.ports ArrayAttr for input arg"
               << std::to_string(i) << ".";
    } else if (helper::isBusLikeType(inputTypes[i])) {
      if (failed(verifyBusPortsAttribute(emitError, inputAttrs[i])))
        return emitError() << "Expected" << helper::getHIRBusPortAttrName()
                           << "ArrayAttr for input arg" << std::to_string(i)
                           << ".";
    } else if (inputTypes[i].dyn_cast<hir::TimeType>()) {
      continue;
    } else {
      return emitError() << "Expected MLIR-builtin-type or hir::MemrefType or "
                            "hir::BusType or hir::BusTensorType or "
                            "hir::TimeType in inputTypes, got :\n\t"
                         << inputTypes[i];
    }
  }

  // Verify results.
  for (size_t i = 0; i < resultTypes.size(); i++) {
    if (helper::isBuiltinSizedType(resultTypes[i])) {
      if (failed(verifyDelayAttribute(emitError, resultAttrs[i])))
        return emitError() << "Expected hir.delay attribute to be an "
                              "IntegerAttr for result "
                           << std::to_string(i) << ".";
    } else if (resultTypes[i].dyn_cast<hir::TimeType>()) {
      return success();
    } else {
      return emitError() << "Expected MLIR-builtin-type or hir::TimeType in "
                            "resultTypes, got :\n\t"
                         << resultTypes[i];
    }
  }
  return success();
}

LogicalResult
BusType::verify(mlir::function_ref<InFlightDiagnostic()> emitError,
                Type elementTy) {
  if (!helper::isBuiltinSizedType(elementTy))
    emitError() << "Bus inner type can only be an integer/float or a "
                   "tuple/tensor of these types.";
  return success();
}

LogicalResult
BusTensorType::verify(mlir::function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<int64_t> shape, Type elementTy) {
  if (!helper::isBuiltinSizedType(elementTy))
    emitError() << "Bus inner type can only be an integer/float or a "
                   "tuple/tensor of these types.";
  for (auto dim : shape)
    if (dim <= 0)
      emitError() << "Dimension size must be greater than zero.";
  return success();
}

/// required for functionlike trait
LogicalResult hir::FuncOp::verifyBody() { return success(); }
