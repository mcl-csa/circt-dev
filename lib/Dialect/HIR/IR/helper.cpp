#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include <stack>
#include <string>

using namespace circt;
using namespace hir;
using namespace llvm;

namespace helper {

std::string typeToString(Type t) {
  std::string typeStr;
  llvm::raw_string_ostream typeOstream(typeStr);
  t.print(typeOstream);
  return typeStr;
}

std::optional<int64_t> getBitWidth(Type type) {
  if (type.dyn_cast<hir::TimeType>())
    return 1;
  if (type.isa<IntegerType>() || type.isa<mlir::FloatType>())
    return type.getIntOrFloatBitWidth();

  if (auto tensorTy = type.dyn_cast<mlir::TensorType>())
    return tensorTy.getNumElements() *
           getBitWidth(tensorTy.getElementType()).getValue();

  if (auto busTy = type.dyn_cast<hir::BusType>())
    return getBitWidth(busTy.getElementType());

  if (auto busTensorTy = type.dyn_cast<hir::BusTensorType>())
    return busTensorTy.getNumElements() *
           getBitWidth(busTensorTy.getElementType()).getValue();

  return llvm::None;
}

unsigned clog2(int value) { return (int)(ceil(log2(((double)value)))); }

IntegerAttr getI64IntegerAttr(MLIRContext *context, int value) {
  return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, value));
}

DictionaryAttr getDictionaryAttr(MLIRContext *context) {
  return DictionaryAttr::get(context, SmallVector<NamedAttribute>({}));
}

DictionaryAttr getDictionaryAttr(StringRef name, Attribute attr) {
  mlir::Builder builder(attr.getContext());
  assert(name != "");
  return DictionaryAttr::get(builder.getContext(),
                             builder.getNamedAttr(name, attr));
}

DictionaryAttr getDictionaryAttr(mlir::RewriterBase &rewriter, StringRef name,
                                 Attribute attr) {
  return DictionaryAttr::get(rewriter.getContext(),
                             rewriter.getNamedAttr(name, attr));
}

bool isBuiltinSizedType(Type ty) {
  if ((ty.isa<IntegerType>() && ty.dyn_cast<IntegerType>().isSignless()) ||
      ty.isa<mlir::FloatType>())
    return true;
  if (ty.isa<TupleType>()) {
    bool tupleMembersArePrimitive = true;
    for (auto memberTy : ty.dyn_cast<TupleType>().getTypes())
      tupleMembersArePrimitive &= isBuiltinSizedType(memberTy);
    if (tupleMembersArePrimitive)
      return true;
  }
  if (ty.isa<mlir::TensorType>() &&
      isBuiltinSizedType(ty.dyn_cast<mlir::TensorType>().getElementType()))
    return true;
  return false;
}

bool isBusLikeType(mlir::Type ty) {
  return (ty.isa<hir::BusType>() || ty.isa<hir::BusTensorType>());
}

TimeType getTimeType(MLIRContext *context) { return TimeType::get(context); }

mlir::ParseResult parseMemrefPortsArray(mlir::DialectAsmParser &parser,
                                        mlir::ArrayAttr &ports) {
  SmallVector<StringRef> portsArray;
  if (parser.parseLParen())
    return failure();

  do {
    StringRef keyword;
    if (succeeded(parser.parseKeyword("send")))
      keyword = "send";
    else if (succeeded(parser.parseKeyword("recv")))
      keyword = "recv";
    else
      return parser.emitError(parser.getCurrentLocation())
             << "Expected 'send' or 'recv' keyword";
    portsArray.push_back(keyword);
  } while (succeeded(parser.parseOptionalComma()));

  ports = parser.getBuilder().getStrArrayAttr(portsArray);
  return success();
}

ParseResult parseIntegerAttr(IntegerAttr &value, StringRef attrName,
                             OpAsmParser &parser, OperationState &result) {

  return parser.parseAttribute(
      value, IntegerType::get(parser.getBuilder().getContext(), 64), attrName,
      result.attributes);
}

std::optional<int64_t> getConstantIntValue(Value var) {
  auto arithConstantOp =
      dyn_cast_or_null<mlir::arith::ConstantOp>(var.getDefiningOp());
  if (arithConstantOp) {
    auto integerAttr = arithConstantOp.getValue().dyn_cast<IntegerAttr>();
    return integerAttr.getInt();
  }
  auto hwConstantOp =
      dyn_cast_or_null<circt::hw::ConstantOp>(var.getDefiningOp());
  if (hwConstantOp) {
    return hwConstantOp.getValue().getSExtValue();
  }
  return llvm::None;
}

mlir::LogicalResult isConstantIntValue(mlir::Value var) {

  if (dyn_cast<mlir::arith::ConstantOp>(var.getDefiningOp()))
    return success();
  return failure();
}

std::optional<int64_t> calcLinearIndex(mlir::ArrayRef<mlir::Value> indices,
                                       mlir::ArrayRef<int64_t> dims) {
  int64_t linearIdx = 0;
  int64_t stride = 1;
  // This can happen if there are no BANK(ADDR) indices.
  if (indices.size() == 0)
    return 0;

  for (int i = indices.size() - 1; i >= 0; i--) {
    auto idxConst = getConstantIntValue(indices[i]);
    if (!idxConst)
      return llvm::None;
    linearIdx += idxConst.getValue() * stride;
    assert(linearIdx <= 1000000);
    stride *= dims[i];
  }
  if (0 > linearIdx) {
    assert(linearIdx);
  }
  assert(linearIdx <= 1000000);
  return linearIdx;
}

std::optional<int64_t> getHIRDelayAttr(mlir::DictionaryAttr dict) {
  auto attr = dict.getNamed("hir.delay");
  if (attr.hasValue())
    if (auto delayAttr = attr.getValue().getValue())
      if (auto intAttr = delayAttr.dyn_cast<IntegerAttr>())
        return intAttr.getInt();
  return llvm::None;
}
mlir::arith::ConstantOp emitConstantOp(mlir::OpBuilder &builder,
                                       int64_t value) {
  return builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                                 builder.getIndexAttr(value));
}

std::optional<ArrayAttr> extractMemrefPortsFromDict(mlir::DictionaryAttr dict) {
  if (!dict.getNamed("hir.memref.ports").hasValue())
    return llvm::None;
  return dict.getNamed("hir.memref.ports")
      .getValue()
      .getValue()
      .dyn_cast<ArrayAttr>();
}

ArrayAttr getPortAttrForReg(Builder &builder) {
  auto rdPort = builder.getDictionaryAttr(
      builder.getNamedAttr("rd_latency", builder.getI64IntegerAttr(0)));
  auto wrPort = builder.getDictionaryAttr(
      builder.getNamedAttr("wr_latency", builder.getI64IntegerAttr(1)));
  return builder.getArrayAttr({rdPort, wrPort});
}

std::optional<int64_t> getMemrefPortRdLatency(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto rdLatencyAttr = portDict.getNamed("rd_latency");
  if (rdLatencyAttr)
    return rdLatencyAttr.getValue().getValue().dyn_cast<IntegerAttr>().getInt();
  return llvm::None;
}

std::optional<int64_t> getMemrefPortWrLatency(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto wrLatencyAttr = portDict.getNamed("wr_latency");
  if (wrLatencyAttr)
    return wrLatencyAttr.getValue().getValue().dyn_cast<IntegerAttr>().getInt();
  return llvm::None;
}

bool isMemrefWrPort(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto wrLatencyAttr = portDict.getAs<IntegerAttr>("wr_latency");
  if (wrLatencyAttr)
    return true;
  return false;
}

bool isMemrefRdPort(Attribute port) {
  auto portDict = port.dyn_cast<DictionaryAttr>();
  auto rdLatencyAttr = portDict.getAs<IntegerAttr>("rd_latency");
  if (rdLatencyAttr)
    return true;
  return false;
}

StringRef extractBusPortFromDict(mlir::DictionaryAttr dict) {
  auto ports = dict.getNamed("hir.bus.ports")
                   .getValue()
                   .getValue()
                   .dyn_cast<ArrayAttr>();
  // Bus port should be either send or recv.
  assert(ports.size() == 1);
  return ports[0].dyn_cast<StringAttr>().getValue();
}

llvm::StringRef getInlineAttrName() { return "inline"; }

void eraseOps(SmallVectorImpl<Operation *> &opsToErase) {
  // Erase the ops in reverse order so that if there are any dependent ops, they
  // get erased first.
  for (auto op = opsToErase.rbegin(); op != opsToErase.rend(); op++)
    (*op)->erase();
  opsToErase.clear();
}

Value lookupOrOriginal(BlockAndValueMapping &mapper, Value originalValue) {
  if (mapper.contains(originalValue))
    return mapper.lookup(originalValue);
  return originalValue;
}
void setNames(Operation *operation, ArrayRef<StringRef> names) {
  OpBuilder builder(operation);
  operation->setAttr("names", builder.getStrArrayAttr(names));
}
SmallVector<Type> getTypes(ArrayRef<Value> values) {
  SmallVector<Type> types;
  for (auto value : values)
    types.push_back(value.getType());
  return types;
}

std::optional<StringRef> getOptionalName(Operation *operation,
                                         int64_t resultNum) {
  auto namesAttr = operation->getAttr("names").dyn_cast_or_null<ArrayAttr>();
  if (!namesAttr)
    return llvm::None;
  auto nameAttr = namesAttr[resultNum].dyn_cast_or_null<StringAttr>();
  if (!nameAttr)
    return llvm::None;
  auto name = nameAttr.getValue();
  if (name.size() == 0)
    return llvm::None;
  return name;
}

std::optional<mlir::StringRef> getOptionalName(mlir::Value v) {
  Operation *operation = v.getDefiningOp();
  if (operation) {
    for (size_t i = 0; i < operation->getNumResults(); i++) {
      if (operation->getResult(i) == v) {
        return getOptionalName(operation, i);
      }
    }
    return llvm::None;
  }
  auto *bb = v.getParentBlock();
  auto argNames = bb->getParentOp()->getAttrOfType<mlir::ArrayAttr>("argNames");
  if (argNames) {
    assert(argNames.size() == bb->getNumArguments());
    for (size_t i = 0; i < bb->getNumArguments(); i++) {
      if (v == bb->getArgument(i)) {
        return argNames[i].dyn_cast<mlir::StringAttr>().getValue();
      }
    }
  }
  return llvm::None;
}

std::optional<Type> getElementType(circt::Type ty) {
  if (auto tensorTy = ty.dyn_cast<mlir::TensorType>())
    return tensorTy.getElementType();
  if (auto busTy = ty.dyn_cast<hir::BusType>())
    return busTy.getElementType();
  if (auto busTensorTy = ty.dyn_cast<hir::BusTensorType>())
    return busTensorTy.getElementType();
  if (auto arrayTy = ty.dyn_cast<hw::ArrayType>())
    return arrayTy.getElementType();
  return llvm::None;
}

circt::Operation *
declareExternalFuncForCall(hir::CallOp callOp, StringRef verilogName,
                           SmallVector<std::string> inputNames,
                           SmallVector<std::string> resultNames) {
  if (auto *alreadyDeclaredOp = callOp.getCalleeDecl())
    return alreadyDeclaredOp;
  OpBuilder builder(callOp);
  auto moduleOp = callOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToStart(moduleOp.getBody());

  // declOp.getFuncBody().front();
  assert(inputNames.size() == callOp.getFuncType().getInputTypes().size());
  inputNames.push_back("t");
  SmallVector<mlir::StringRef> inputNamesRef;
  for (size_t i = 0; i < inputNames.size(); i++) {
    inputNamesRef.push_back(inputNames[i]);
  }
  assert(resultNames.size() == callOp.getFuncType().getResultTypes().size());
  SmallVector<mlir::StringRef> resultNamesRef;
  for (size_t i = 0; i < resultNames.size(); i++) {
    resultNamesRef.push_back(resultNames[i]);
  }

  auto declOp = builder.create<hir::FuncExternOp>(
      builder.getUnknownLoc(), callOp.callee(), callOp.getFuncType(),
      builder.getStrArrayAttr(inputNamesRef),
      builder.getStrArrayAttr(resultNamesRef));

  OpBuilder declOpBuilder(declOp);
  FuncExternOp::ensureTerminator(declOp.getFuncBody(), declOpBuilder,
                                 builder.getUnknownLoc());

  if (auto params = callOp->getAttr("params"))
    declOp->setAttr("params", params);
  declOp->setAttr("verilogName", builder.getStringAttr(verilogName));
  return declOp;
}

Value materializeIntegerConstant(OpBuilder &builder, int value, int64_t width) {
  return builder.create<hw::ConstantOp>(
      builder.getUnknownLoc(),
      builder.getIntegerAttr(IntegerType::get(builder.getContext(), width),
                             value));
}

static Optional<Type> convertBusType(hir::BusType busTy) {
  return convertToHWType(busTy.getElementType());
}

static Optional<Type> convertBusTensorType(hir::BusTensorType busTensorTy) {
  auto elementHWTy = convertToHWType(busTensorTy.getElementType());
  if (!elementHWTy) {
    Builder builder(busTensorTy.getContext());
    mlir::emitError(builder.getUnknownLoc())
        << "Could not convert bus tensor element type "
        << busTensorTy.getElementType();
  }

  // BusTensor element must always be a value type and hence always convertible
  // to a valid hw type.
  if (busTensorTy.getNumElements() == 1)
    return elementHWTy;
  return hw::ArrayType::get(*elementHWTy, busTensorTy.getNumElements());
}

static Optional<Type> convertTensorType(mlir::TensorType tensorTy) {
  auto elementHWTy = convertToHWType(tensorTy.getElementType());
  // Tensor element must always be a value type and hence always convertible
  // to a valid hw type.
  if (tensorTy.getNumElements() == 1)
    return elementHWTy;
  return hw::ArrayType::get(*elementHWTy, tensorTy.getNumElements());
}

static Optional<Type> convertTupleType(mlir::TupleType tupleTy) {
  int64_t width = 0;
  for (auto elementTy : tupleTy.getTypes()) {
    // We can't handle tensors/arrays inside tuple.
    auto elementHWTy = convertToHWType(elementTy);
    if (!(elementHWTy && elementHWTy.getValue().isa<IntegerType>()))
      return llvm::None;
    width += (*elementHWTy).dyn_cast<IntegerType>().getWidth();
  }
  return IntegerType::get(tupleTy.getContext(), width);
}

std::optional<Type> convertToHWType(Type type) {
  if (type.isa<TimeType>())
    return IntegerType::get(type.getContext(), 1);
  if (type.isa<IntegerType>())
    return type;
  if (type.isa<mlir::FloatType>())
    return IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth());
  if (auto ty = type.dyn_cast<mlir::TensorType>())
    return convertTensorType(ty);
  if (auto ty = type.dyn_cast<mlir::TupleType>())
    return convertTupleType(ty);
  if (auto ty = type.dyn_cast<hir::BusType>())
    return convertBusType(ty);
  if (auto ty = type.dyn_cast<hir::BusTensorType>())
    return convertBusTensorType(ty);
  return llvm::None;
}

Value insertBusSelectLogic(OpBuilder &builder, Value selectBus, Value trueBus,
                           Value falseBus) {
  auto uLoc = builder.getUnknownLoc();
  if (trueBus.getType() != falseBus.getType()) {
    trueBus.getDefiningOp()->emitError("type mismatch. Below is trueBus")
        << ", falseBus type = " << falseBus.getType();
    assert(false);
  }

  return builder
      .create<hir::BusMapOp>(
          builder.getUnknownLoc(),
          ArrayRef<Value>({selectBus, trueBus, falseBus}),
          [&uLoc](OpBuilder &builder, ArrayRef<Value> operands) {
            Value result = builder.create<comb::MuxOp>(
                uLoc, operands[0], operands[1], operands[2]);

            return builder.create<hir::YieldOp>(uLoc, result);
          })
      .getResult(0);
}

Value insertMultiBusSelectLogic(OpBuilder &builder, Value selectBusT,
                                Value trueBusT, Value falseBusT) {
  auto uLoc = builder.getUnknownLoc();
  return builder
      .create<hir::BusTensorMapOp>(
          builder.getUnknownLoc(),
          ArrayRef<Value>({selectBusT, trueBusT, falseBusT}),
          [&uLoc](OpBuilder &builder, ArrayRef<Value> operands) {
            Value result = builder.create<comb::MuxOp>(
                uLoc, operands[0], operands[1], operands[2]);

            return builder.create<hir::YieldOp>(uLoc, result);
          })
      .getResult(0);
}

Value emitRegisterAlloca(OpBuilder &builder, Type elementTy) {
  auto rdAttr = builder.getDictionaryAttr(
      builder.getNamedAttr("rd_latency", builder.getI64IntegerAttr(0)));
  auto wrAttr = builder.getDictionaryAttr(
      builder.getNamedAttr("wr_latency", builder.getI64IntegerAttr(1)));
  auto ports = builder.getArrayAttr({rdAttr, wrAttr});

  auto regMemKind =
      hir::MemKindEnumAttr::get(builder.getContext(), MemKindEnum::reg);

  return builder.create<hir::AllocaOp>(
      builder.getUnknownLoc(),
      hir::MemrefType::get(builder.getContext(), 1, elementTy,
                           hir::DimKind::BANK),
      regMemKind, ports);
}

LogicalResult validatePositiveConstant(ArrayRef<Value> indices) {
  for (auto idx : indices) {
    auto idxValue = getConstantIntValue(idx);
    if (!idxValue)
      return idx.getDefiningOp()->emitError(
          "Expected this to be a arith.constant");

    if (*idxValue < 0)
      return idx.getDefiningOp()->emitError(
          "Expected this to be a +ve arith.constant");
  }
  return success();
}

Value emitIntegerBusOp(OpBuilder &builder, int64_t width) {
  assert(width > 0);
  return builder.create<hir::BusOp>(
      builder.getUnknownLoc(),
      hir::BusType::get(builder.getContext(), builder.getIntegerType(width)));
}
std::optional<int64_t> getOptionalTimeOffset(mlir::Operation *operation) {
  auto offsetAttr = operation->getAttrOfType<IntegerAttr>("offset");
  if (!offsetAttr)
    return llvm::None;
  return offsetAttr.getInt();
}

} // namespace helper
