#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <mlir/IR/BuiltinTypes.h>
using namespace circt;
using namespace hir;
//-----------------------------------------------------------------------------
// Helper functions.
//-----------------------------------------------------------------------------
namespace {
LogicalResult verifyTimeAndOffset(Value time, std::optional<uint64_t> offset) {
  if (time && !offset)
    return failure();
  if (offset && offset.value() < 0)
    return failure();
  return success();
}

LogicalResult checkRegionCaptures(Region &region) {
  SmallVector<InFlightDiagnostic> errorMsgs;
  mlir::visitUsedValuesDefinedAbove(region, [&errorMsgs](OpOperand *operand) {
    Type ty = operand->get().getType();
    if (ty.isa<hir::MemrefType>() || ty.isa<hir::BusType>())
      return;
    if (ty.isa<hir::TimeType>())
      errorMsgs.push_back(
          operand->getOwner()->emitError()
          << "hir::TimeType can not be captured by this region. Only the "
             "region's "
             "own time-var (and any other time-var derived from it) is "
             "available inside the region.");
    return;
  });
  if (!errorMsgs.empty())
    return failure();
  return success();
}
} // namespace
//-----------------------------------------------------------------------------

namespace circt {
namespace hir {
//-----------------------------------------------------------------------------
// HIR Op verifiers.
//-----------------------------------------------------------------------------
LogicalResult FuncOp::verify() {
  auto funcTy = this->getFuncTy().dyn_cast<hir::FuncType>();
  if (!funcTy)
    return this->emitError("OpVerifier failed. hir::FuncOp::funcTy must be of "
                           "type hir::FuncType.");
  for (Type arg : funcTy.getInputTypes()) {
    if (arg.isa<IndexType>())
      return this->emitError(
          "hir.func op does not support index type in argument location.");
  }

  auto argAttrs =
      this->getOperation()->getAttrOfType<mlir::ArrayAttr>("arg_attrs");
  auto inputAttrs = this->getFuncType().getInputAttrs();
  if (!inputAttrs.empty()) {
    if (!argAttrs)
      return this->emitError("Could not find arg_attrs.");

    if (inputAttrs.size() + 1 != argAttrs.size())
      return this->emitError("Wrong number of arg_attrs.");
  }
  if (!this->getFuncType().getResultAttrs().empty()) {
    if (!this->getOperation()->getAttrOfType<mlir::ArrayAttr>("res_attrs"))
      return this->emitError("Could not find res_attrs.");

    if (this->getFuncType().getResultAttrs().size() !=
        this->getOperation()
            ->getAttrOfType<mlir::ArrayAttr>("res_attrs")
            .size())
      return this->emitError("Wrong number of res_attrs.");
  }
  // Check if the port assignment of memref use is correct.
  for (uint64_t i = 0; i < this->getFuncBody().getNumArguments(); i++) {
    Value arg = this->getFuncBody().getArguments()[i];
    if (arg.getType().isa<hir::MemrefType>()) {
      for (auto &use : arg.getUses()) {
        if (auto loadOp = dyn_cast<hir::LoadOp>(use.getOwner())) {
          auto port = loadOp.getPort();
          if (!port)
            continue;
          auto ports = helper::extractMemrefPortsFromDict(
              this->getFuncType().getInputAttrs()[i]);

          if (ports.value().size() <= port.value())
            return use.getOwner()->emitError()
                   << "specified port does not exist.";
          auto loadOpPort = ports.value()[port.value()];
          if (!helper::isMemrefRdPort(loadOpPort))
            return use.getOwner()->emitError()
                   << "specified port is not a read port.";
        } else if (auto storeOp = dyn_cast<hir::StoreOp>(use.getOwner())) {
          auto port = storeOp.getPort();
          if (!port)
            continue;
          auto ports = helper::extractMemrefPortsFromDict(
              this->getFuncType().getInputAttrs()[i]);

          if (ports.value().size() <= port.value())
            return use.getOwner()->emitError()
                   << "specified port does not exist.";

          auto storeOpPort = ports.value()[port.value()];
          if (!helper::isMemrefWrPort(storeOpPort))
            return use.getOwner()->emitError()
                   << "specified port is not a write port.";
        }
      }
    }
  }
  auto inputTypes = funcTy.getInputTypes();
  auto resultTypes = funcTy.getResultTypes();
  if (!inputTypes.empty()) {
    auto argNames = this->getOperation()->getAttrOfType<ArrayAttr>("argNames");
    if (!argNames)
      return this->emitError("argNames attribute required.");
    // argNames also contains the start time.
    if (argNames.size() != inputTypes.size() + 1)
      return this->emitError("Mismatch in number of argument names.")
             << argNames.size() << " names provided but required "
             << inputTypes.size() + 1;
  }
  if (!resultTypes.empty()) {
    auto resultNames =
        this->getOperation()->getAttrOfType<ArrayAttr>("resultNames");
    if (!resultNames)
      return this->emitError("resultNames attribute is required.");
    if (resultNames.size() != resultTypes.size())
      return this->emitError("Mismatch in number of result names.");
  }
  return success();
}

LogicalResult checkIndices(Location loc, OperandRange indices,
                           MemrefType memTy) {
  if (memTy.getShape().size() != indices.size())
    return mlir::emitError(loc)
           << "Wrong number of dimensions while indexing memory.";
  for (size_t i = 0; i < indices.size(); i++) {
    auto constIdx = helper::getConstantIntValue(indices[i]);
    if (!constIdx)
      continue;
    if (!(memTy.getShape()[i] > *constIdx))
      return emitError(loc) << "Constant index is out-of-bounds!"
                            << "{idx:" << *constIdx
                            << ", dim-size:" << memTy.getShape()[i] << "}.";
  }
  return success();
}

LogicalResult AllocaOp::verify() {
  auto res = this->getRes();
  auto ports = this->getPorts();
  for (auto &use : res.getUses()) {
    if (auto loadOp = dyn_cast<hir::LoadOp>(use.getOwner())) {
      auto port = loadOp.getPort();
      if (!port)
        continue;
      auto loadOpPort = ports[port.value()];
      if (!helper::isMemrefRdPort(loadOpPort))
        return use.getOwner()->emitError()
               << "specified port is not a read port.";
    } else if (auto storeOp = dyn_cast<hir::StoreOp>(use.getOwner())) {
      auto port = storeOp.getPort();
      if (failed(checkIndices(
              storeOp->getLoc(), storeOp.getIndices(),
              storeOp.getMem().getType().dyn_cast<hir::MemrefType>())))
        return failure();
      if (!port)
        continue;
      if (ports.size() <= port.value())
        return use.getOwner()
                   ->emitError("Invalid port number.")
                   .attachNote(this->getLoc())
               << "Memref defined here.";
      auto storeOpPort = ports[port.value()];
      if (!helper::isMemrefWrPort(storeOpPort))
        return use.getOwner()->emitError()
               << "specified port is not a write port.";
    }
  }
  if (this->getMemKind() == MemKindEnum::reg) {
    if (this->getRes()
            .getType()
            .dyn_cast<hir::MemrefType>()
            .getNumElementsPerBank() != 1)
      return this->emitError("'reg' must have all dims banked.");
    if (this->getPorts().size() != 2)
      return this->emitError("'reg' must two ports, read and write.");
    if (helper::isMemrefWrPort(this->getPorts()[0]))
      return this->emitError("'reg' port 0 must be read-only.");
    if (helper::isMemrefRdPort(this->getPorts()[1]))
      return this->emitError("'reg' port 1 must be write-only.");
    if (helper::getMemrefPortRdLatency(this->getPorts()[0]).value() != 0)
      return this->emitError("'reg' read latency must be 0.");
    if (helper::getMemrefPortWrLatency(this->getPorts()[1]).value() != 1)
      return this->emitError("'reg' write latency must be 1.");
  } else {
    if (this->getRes()
            .getType()
            .dyn_cast<hir::MemrefType>()
            .getNumElementsPerBank() == 1)
      this->emitWarning(
          "Your probably want to use 'reg' since there is only one "
          "element per bank.");
  }
  return success();
}

LogicalResult BusTensorMapOp::verify() {
  if (this->getNumOperands() == 0)
    return this->emitError() << "Op must have atleast one operand.";
  if (this->getNumResults() == 0)
    return this->emitError() << "Op must have atleast one result.";
  auto busTensorTy =
      this->getResult(0).getType().dyn_cast_or_null<hir::BusTensorType>();
  if (!busTensorTy)
    return this->emitError(
        "Expected input and result types to be hir.bus_tensor type.");

  for (auto operand : this->getOperands()) {
    auto operandTy = operand.getType().dyn_cast_or_null<hir::BusTensorType>();
    if (!operandTy)
      return this->emitError(
          "Expected input and result types to be hir.bus_tensor type.");
    if (operandTy.getShape() != busTensorTy.getShape())
      return this->emitError(
          "Expected all input and result tensors to have same shape.");
  }

  for (auto result : this->getResults()) {
    auto resultTy = result.getType().dyn_cast_or_null<hir::BusTensorType>();
    if (!resultTy)
      return this->emitError(
          "Expected input and result types to be hir.bus_tensor type.");
    if (resultTy.getShape() != busTensorTy.getShape())
      return this->emitError(
          "Expected input and result tensors to have same shape.");
  }
  return success();
}

LogicalResult YieldOp::verify() {
  auto *operation = this->getOperation()->getParentRegion()->getParentOp();
  auto resultTypes = operation->getResultTypes();
  auto operands = this->getOperands();
  if (resultTypes.size() != operands.size())
    return this->emitError()
           << "Expected " << resultTypes.size() << " operands.";
  for (uint64_t i = 0; i < resultTypes.size(); i++) {
    if (!helper::isBuiltinSizedType(operands[i].getType()))
      return this->emitError() << "verifyYieldOp: Unsupported input type.";
  }
  return success();
}

LogicalResult FuncExternOp::verify() {
  auto funcTy = this->getFuncTy().dyn_cast<hir::FuncType>();
  if (!funcTy)
    return this->emitError("OpVerifier failed. hir::FuncOp::funcTy must be of "
                           "type hir::FuncType.");
  for (Type arg : funcTy.getInputTypes()) {
    if (arg.isa<IndexType>())
      return this->emitError(
          "hir.func op does not support index type in argument location.");
  }

  auto inputTypes = funcTy.getInputTypes();
  auto resultTypes = funcTy.getResultTypes();
  if (!inputTypes.empty()) {
    auto argNames = this->getOperation()->getAttrOfType<ArrayAttr>("argNames");
    // argNames also contains the start time.
    if ((!argNames) || (argNames.size() - 1 != inputTypes.size()))
      return this->emitError("Mismatch in number of argument names.");
  }
  if (!resultTypes.empty()) {
    auto resultNames =
        this->getOperation()->getAttrOfType<ArrayAttr>("resultNames");
    if ((!resultNames) || (resultNames.size() != resultTypes.size()))
      return this->emitError("Mismatch in number of result names.");
  }
  return success();
}

std::optional<std::string> typeMismatch(Location loc, hir::FuncType ta,
                                        hir::FuncType tb) {

  auto inputTypesA = ta.getInputTypes();
  auto inputTypesB = tb.getInputTypes();
  if (inputTypesA.size() != inputTypesB.size())
    return std::string("Mismatch with function declaration. Mismatched number "
                       "of input types.");
  for (size_t i = 0; i < inputTypesA.size(); i++) {
    if (inputTypesA[i] != inputTypesB[i])
      return std::string("Type Mismatch in input arg ") + std::to_string(i);
  }

  auto resultTypesA = ta.getResultTypes();
  auto resultTypesB = tb.getResultTypes();
  if (resultTypesA.size() != resultTypesB.size())
    return std::string("Mismatch with function declaration. Mismatched number "
                       "of result types.");
  for (size_t i = 0; i < resultTypesA.size(); i++) {
    if (resultTypesA[i] != resultTypesB[i])
      return std::string("Type Mismatch in result arg ") + std::to_string(i);
  }

  // if (ta != tb)
  //  return std::string("Mismatch in FuncType.");
  return std::nullopt;
}

bool isEqual(mlir::DictionaryAttr d1, mlir::DictionaryAttr d2) {

  if (!d1 && !d2)
    return true;
  if (!d1 || !d2)
    return false;

  for (auto param : d1) {
    auto value = d2.get(param.getName());
    if (!value || (value != param.getValue()))
      return false;
  }

  for (auto param : d2) {
    auto value = d1.get(param.getName());
    if (!value || (value != param.getValue()))
      return false;
  }

  return true;
}

LogicalResult CallOp::verify() {
  auto inputTypes = this->getFuncType().getInputTypes();
  auto operands = this->getOperands();
  for (uint64_t i = 0; i < operands.size(); i++) {
    if (operands[i].getType() != inputTypes[i]) {
      this->emitError() << "input arg " << i
                        << " expects different type than prior uses: '"
                        << inputTypes[i] << "' vs '" << operands[i].getType();
      return failure();
    }
  }
  auto *calleeDeclOperation = this->getCalleeDecl();
  if (!calleeDeclOperation)
    return this->emitError() << "Could not find function declaration.";

  if (auto funcOp = dyn_cast_or_null<hir::FuncOp>(calleeDeclOperation)) {
    auto error =
        typeMismatch(this->getLoc(), funcOp.getFuncType(), this->getFuncType());
    if (error)
      return this->emitError("Mismatch with function definition.")
                 .attachNote(funcOp.getLoc())
             << error.value() << "\n\ntypes are \n"
             << funcOp.getFuncType() << "\n and \n"
             << this->getFuncType();
  } else if (hir::FuncExternOp funcExternOp =
                 dyn_cast_or_null<hir::FuncExternOp>(calleeDeclOperation)) {
    auto error = typeMismatch(this->getLoc(), funcExternOp.getFuncType(),
                              this->getFuncType());
    if (error)
      return this->emitError("Mismatch with function declaration.")
                 .attachNote(funcExternOp.getLoc())
             << error.value();
  }
  auto callOpParams =
      this->getOperation()->getAttrOfType<mlir::DictionaryAttr>("params");
  auto declOpParams =
      calleeDeclOperation->getAttrOfType<mlir::DictionaryAttr>("params");
  if (!isEqual(callOpParams, declOpParams))
    return this->emitError(
        "Mismatch in params attribute between function declaration and use.");
  return success();
}

LogicalResult CastOp::verify() {
  if (this->getInput().getType().isa<mlir::IndexType>() &&
      this->getRes().getType().isa<mlir::IntegerType>())
    return success();
  auto inputHWType = helper::convertToHWType(this->getInput().getType());
  auto resultHWType = helper::convertToHWType(this->getRes().getType());
  if (!inputHWType)
    return this->emitError()
           << "Input type should be convertible to a valid hardware type.";
  if (!resultHWType)
    return this->emitError()
           << "Result type should be convertible to a valid hardware type.";
  if (*inputHWType != *resultHWType)
    return this->emitError() << "Incompatible Input and Result types.";

  return success();
}

LogicalResult DelayOp::verify() {
  Type inputTy = this->getInput().getType();
  if (helper::isBuiltinSizedType(inputTy) || helper::isBusLikeType(inputTy))
    return success();
  return this->emitError("hir.delay op only supports signless-integer, float "
                         "and tuple/tensor of these types.");
  if (this->getDelay() < 0) {
    return this->emitError("Delay can not be negative.");
  }
}

LogicalResult ForOp::verify() {
  if (failed(verifyTimeAndOffset(this->getTstart(), this->getOffset())))
    return this->emitError("Invalid offset.");
  auto ivTy = this->getInductionVar().getType();
  if (this->getOperation()->getAttr("unroll"))
    if (!ivTy.isa<IndexType>())
      return this->emitError("Expected induction-var to be IndexType for loop "
                             "with 'unroll' attribute.");
  if (!ivTy.isIntOrIndex())
    return this->emitError(
        "Expected induction var to be IntegerType or IndexType.");
  if (this->getLb().getType() != ivTy)
    return this->emitError("Expected lower bound to be of type ")
           << ivTy << ".";
  if (this->getUb().getType() != ivTy)
    return this->emitError("Expected upper bound to be of type ")
           << ivTy << ".";
  if (this->getStep().getType() != ivTy)
    return this->emitError("Expected step size to be of type ") << ivTy << ".";
  if (this->getInductionVar().getType() != ivTy)
    return this->emitError("Expected induction var to be of type ")
           << ivTy << ".";
  if (!this->getIterTimeVar().getType().isa<hir::TimeType>())
    return this->emitError("Expected time var to be of !hir.time type.");
  auto nextIterOp =
      dyn_cast<hir::NextIterOp>(this->getLoopBody().front().getTerminator());
  if (nextIterOp.getIterArgs().size() != this->getIterArgs().size())
    return nextIterOp.emitError(
        "Mismatch in number of iter args with the enclosing ForOp.");
  if (failed(checkRegionCaptures(this->getLoopBody())))
    return failure();

  return success();
}

LogicalResult LoadOp::verify() {
  if (failed(verifyTimeAndOffset(this->getTstart(), this->getOffset())))
    return this->emitError("Invalid offset.");
  return success();
}
LogicalResult StoreOp::verify() {
  if (failed(verifyTimeAndOffset(this->getTstart(), this->getOffset())))
    return this->emitError("Invalid offset.");
  return success();
}

LogicalResult IfOp::verify() {
  if (failed(verifyTimeAndOffset(this->getTstart(), this->getOffset())))
    return this->emitError("Invalid offset.");
  if (failed(checkRegionCaptures(this->getIfRegion())))
    return failure();
  if (failed(checkRegionCaptures(this->getElseRegion())))
    return failure();
  return success();
}
LogicalResult NextIterOp::verify() {
  if (this->getCondition() &&
      isa<hir::ForOp>(this->getOperation()->getParentOp()))
    return this->emitError(
        "condition is not supported in hir.next_iter when it "
        "is inside a hir.for op.");
  return success();
}

LogicalResult ProbeOp::verify() {
  auto ty = this->getInput().getType();
  if (!(helper::isBuiltinSizedType(ty) ||
        ty.isa<mlir::IndexType, hir::TimeType>() || helper::isBusLikeType(ty)))
    return this->emitError() << "Unsupported type for hir.probe.";
  if (this->getVerilogName().empty() ||
      this->getVerilogName().startswith("%") ||
      isdigit(this->getVerilogName().data()[0]))
    return this->emitError() << "Invalid name.";
  return success();
}

LogicalResult BusTensorInsertElementOp::verify() {
  auto busTensorTy = this->getTensor().getType().dyn_cast<hir::BusTensorType>();
  if (!busTensorTy)
    return this->emitError()
           << "Expected BusTensorType, got " << this->getTensor().getType();
  auto busTy = this->getElement().getType().dyn_cast<hir::BusType>();
  if (!busTy)
    return this->emitError()
           << "Expected BusType, got " << this->getElement().getType();
  if (busTy.getElementType() != busTensorTy.getElementType())
    return this->emitError() << "Incompatible input types";
  return success();
}

LogicalResult IsFirstIterOp::verify() {
  auto *parentOperation = (*this)->getParentOp();
  if (auto whileOp = dyn_cast<WhileOp>(parentOperation)) {
    if (this->getStartTime() != Time(whileOp.getIterTimeVar(), 0))
      return this->emitError("This op must be scheduled at time ")
             << whileOp.getIterTimeVar();
  } else {
    auto forOp = dyn_cast<ForOp>(parentOperation);
    assert(forOp);
    if (this->getStartTime() != Time(forOp.getIterTimeVar(), 0))
      return this->emitError("This op must be scheduled at time ")
             << forOp.getIterTimeVar();
  }

  return success();
}
//-----------------------------------------------------------------------------
} // namespace hir
} // namespace circt
