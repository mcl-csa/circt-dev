#ifndef HIR_HELPER_H
#define HIR_HELPER_H

#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace helper {
// Templates.

// Declarations.
std::optional<int64_t> getBitWidth(mlir::Type);
unsigned clog2(int);

bool isBuiltinSizedType(mlir::Type);
bool isBusLikeType(mlir::Type);
std::optional<int64_t> getConstantIntValue(mlir::Value var);
mlir::LogicalResult isConstantIntValue(mlir::Value var);
mlir::IntegerAttr getI64IntegerAttr(mlir::MLIRContext *context, int value);

circt::hir::TimeType getTimeType(mlir::MLIRContext *context);
mlir::ParseResult parseMemrefPortsArray(mlir::DialectAsmParser &,
                                        mlir::ArrayAttr &);
mlir::ParseResult parseIntegerAttr(mlir::IntegerAttr &value,
                                   mlir::StringRef attrName,
                                   mlir::OpAsmParser &parser,
                                   mlir::OperationState &result);

mlir::DictionaryAttr getDictionaryAttr(mlir::StringRef name,
                                       mlir::Attribute attr);
mlir::DictionaryAttr getDictionaryAttr(mlir::MLIRContext *);

mlir::DictionaryAttr getDictionaryAttr(mlir::RewriterBase &builder,
                                       mlir::StringRef name,
                                       mlir::Attribute attr);

std::optional<int64_t> calcLinearIndex(mlir::ArrayRef<mlir::Value> indices,
                                       mlir::ArrayRef<int64_t> dims);

std::optional<int64_t> getHIRDelayAttr(mlir::DictionaryAttr dict);
mlir::arith::ConstantOp emitConstantOp(mlir::OpBuilder &builder, int64_t value);
std::optional<mlir::ArrayAttr>
extractMemrefPortsFromDict(mlir::DictionaryAttr dict);
mlir::ArrayAttr getPortAttrForReg(mlir::Builder &builder);
std::optional<int64_t> getMemrefPortRdLatency(mlir::Attribute port);
std::optional<int64_t> getMemrefPortWrLatency(mlir::Attribute port);
bool isMemrefWrPort(mlir::Attribute port);
bool isMemrefRdPort(mlir::Attribute port);
llvm::StringRef extractBusPortFromDict(mlir::DictionaryAttr dict);
llvm::StringRef getInlineAttrName();
void eraseOps(mlir::SmallVectorImpl<mlir::Operation *> &opsToErase);
mlir::Value lookupOrOriginal(mlir::BlockAndValueMapping &mapper,
                             mlir::Value originalValue);
void setNames(mlir::Operation *, mlir::ArrayRef<mlir::StringRef>);

mlir::SmallVector<mlir::Type> getTypes(mlir::ArrayRef<mlir::Value>);
std::optional<mlir::StringRef> getOptionalName(mlir::Operation *operation,
                                               int64_t resultNum);
std::optional<mlir::StringRef> getOptionalName(mlir::Value v);
std::optional<circt::Type> getElementType(circt::Type);
circt::Operation *
declareExternalFuncForCall(circt::hir::CallOp callOp,
                           llvm::StringRef verilogName,
                           circt::SmallVector<std::string> inputNames,
                           circt::SmallVector<std::string> resultNames = {});
mlir::Value materializeIntegerConstant(mlir::OpBuilder &builder, int value,
                                       int64_t width);
std::optional<mlir::Type> convertToHWType(mlir::Type type);

mlir::Value insertBusSelectLogic(mlir::OpBuilder &builder,
                                 mlir::Value selectBus, mlir::Value trueBus,
                                 mlir::Value falseBus);
mlir::Value insertMultiBusSelectLogic(mlir::OpBuilder &builder,
                                      mlir::Value selectBusT,
                                      mlir::Value trueBusT,
                                      mlir::Value falseBusT);
mlir::Value emitRegisterAlloca(mlir::OpBuilder &builder, mlir::Type elementTy);

mlir::LogicalResult
validatePositiveConstant(mlir::ArrayRef<mlir::Value> indices);
mlir::Value emitIntegerBusOp(mlir::OpBuilder &builder, int64_t width);
std::optional<int64_t> getOptionalTimeOffset(mlir::Operation *);
} // namespace helper
#endif
