//===- HIRPragma.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HIRPragma.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <list>
#include <stack>

using namespace circt;
namespace {
struct HIRPragma : public HIRPragmaBase<HIRPragma> {
  void runOnOperation() override;

private:
  LogicalResult visitOp(mlir::func::FuncOp);
  LogicalResult visitOp(mlir::memref::AllocaOp);
  LogicalResult visitOp(mlir::affine::AffineForOp);
  LogicalResult visitOp(mlir::affine::AffineLoadOp);
  LogicalResult visitOp(mlir::affine::AffineStoreOp);
  LogicalResult visitOp(mlir::func::CallOp);
  LogicalResult visitOp(mlir::arith::NegFOp);
  LogicalResult visitOp(mlir::LLVM::UndefOp);
  LogicalResult visitArithFOp(Operation *operation);
  std::optional<int> selectRdPort(Value mem);
  std::optional<int> selectWrPort(Value mem);
  void safelyEraseOps();

private:
  SmallVector<Operation *> toErase;
  llvm::DenseMap<Value, int> mapMemref2PrevUsedRdPort;
  llvm::DenseMap<Value, int> mapMemref2PrevUsedWrPort;
};

} // namespace

ArrayAttr getMemrefPortAttr(Value mem) {
  assert(mem.getType().isa<mlir::MemRefType>());
  ArrayAttr portsAttr;
  if (auto *operation = mem.getDefiningOp()) {
    auto allocaOp = dyn_cast<mlir::memref::AllocaOp>(operation);
    assert(allocaOp);
    portsAttr = allocaOp->getAttrOfType<ArrayAttr>("hir.memref.ports");
    if (!portsAttr)
      allocaOp->emitError("Could not find hir.memref.ports attr.");
  } else {
    auto funcOp =
        dyn_cast<mlir::func::FuncOp>(mem.getParentRegion()->getParentOp());
    assert(funcOp);
    size_t argNum;
    for (argNum = 0; argNum < mem.getParentRegion()->getNumArguments();
         argNum++)
      if (mem.getParentRegion()->getArgument(argNum) == mem)
        break;

    auto memrefAttrs = funcOp->getAttrOfType<ArrayAttr>("arg_attrs")[argNum]
                           .dyn_cast<DictionaryAttr>();
    if (memrefAttrs)
      portsAttr = memrefAttrs.getAs<ArrayAttr>("hir.memref.ports");

    if (!portsAttr)
      funcOp->emitError("Could not find hir.memref.ports attr for arg.");
  }

  return portsAttr;
}

std::optional<int> HIRPragma::selectRdPort(Value mem) {
  auto portsAttr = getMemrefPortAttr(mem);
  assert(0 < portsAttr.size() && portsAttr.size() <= 2);
  if (portsAttr.size() == 1)
    return 0;
  auto prevPortNum = mapMemref2PrevUsedRdPort[mem];
  auto nextPortNum = (prevPortNum + 1) % 2;
  auto pOld = portsAttr[prevPortNum].dyn_cast<DictionaryAttr>();
  auto pNext = portsAttr[nextPortNum].dyn_cast<DictionaryAttr>();

  if (helper::isMemrefRdPort(pNext)) {
    mapMemref2PrevUsedRdPort[mem] = nextPortNum;
    return nextPortNum;
  }
  if (helper::isMemrefRdPort(pOld))
    return prevPortNum;
  return std::nullopt;
}

std::optional<int> HIRPragma::selectWrPort(Value mem) {
  auto portsAttr = getMemrefPortAttr(mem);
  assert(0 < portsAttr.size() && portsAttr.size() <= 2);
  if (portsAttr.size() == 1)
    return 0;
  auto prevPortNum = mapMemref2PrevUsedWrPort[mem];
  auto nextPortNum = (prevPortNum + 1) % 2;
  auto pOld = portsAttr[prevPortNum].dyn_cast<DictionaryAttr>();
  auto pNext = portsAttr[nextPortNum].dyn_cast<DictionaryAttr>();

  if (helper::isMemrefWrPort(pNext)) {
    mapMemref2PrevUsedWrPort[mem] = nextPortNum;
    return nextPortNum;
  }
  if (helper::isMemrefWrPort(pOld))
    return prevPortNum;
  return std::nullopt;
}

void HIRPragma::safelyEraseOps() {
  llvm::DenseSet<Operation *> erasedOps;
  for (auto *operation : toErase) {
    if (erasedOps.contains(operation))
      continue;
    assert(operation != getOperation());
    erasedOps.insert(operation);
    operation->erase();
  }
  toErase.clear();
}

void HIRPragma::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(&moduleOp.getBodyRegion());
  moduleOp->setAttrs(builder.getDictionaryAttr(
      builder.getNamedAttr("hir.hls", builder.getUnitAttr())));

  std::list<mlir::func::FuncOp> hwAccelOps;
  // Hoist all declarations and put the hw functions in a set.
  // FIXME: Currently we assume that top level function only calls declarations.
  // All functions with body are inlined.
  // We need to process all the child functions in post-order starting from top
  // level func.
  moduleOp->walk([this, &hwAccelOps, &builder](Operation *operation) {
    if (auto op = dyn_cast<mlir::func::FuncOp>(operation)) {
      if (op.isDeclaration()) {
        // push the declarations to the front so that they are processed
        // before their use. hir-to-hw lowering pass does not work if the
        // declarations come later.
        op->setAttr("hwAccel", builder.getUnitAttr());
        hwAccelOps.push_front(op);
      } else if (op.getName() == this->topLevelFuncName) {
        op->setAttr("hwAccel", builder.getUnitAttr());
        hwAccelOps.push_back(op);
      } else {
        toErase.push_back(op);
      }
    }
  });

  if (hwAccelOps.size() == 0)
    return;
  builder.setInsertionPointToStart(hwAccelOps.front()->getBlock());
  // Hoist all decls. We need decls before use in hir-to-hw pass.
  for (auto &funcOp : hwAccelOps) {
    if (funcOp.isDeclaration()) {
      Operation *old = funcOp;
      funcOp =
          dyn_cast<mlir::func::FuncOp>(builder.cloneWithoutRegions(funcOp));
      old->erase();
    }
  }
  for (auto funcOp : hwAccelOps) {
    if (failed(visitOp(funcOp))) {
      signalPassFailure();
      break;
    }
    auto walkresult =
        funcOp->walk<mlir::WalkOrder::PostOrder>([this](Operation *operation) {
          if (auto op = dyn_cast<mlir::memref::AllocaOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (auto op = dyn_cast<mlir::affine::AffineForOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (auto op =
                         dyn_cast<mlir::affine::AffineLoadOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (auto op =
                         dyn_cast<mlir::affine::AffineStoreOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (auto op = dyn_cast<mlir::func::CallOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (auto op = dyn_cast<mlir::arith::NegFOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (auto op = dyn_cast<mlir::LLVM::UndefOp>(operation)) {
            if (failed(visitOp(op)))
              return WalkResult::interrupt();
          } else if (isa<mlir::arith::ConstantOp>(operation)) {
            WalkResult::advance();
          } else if (isa<mlir::arith::ArithDialect>(operation->getDialect())) {
            if (failed(visitArithFOp(operation)))
              return WalkResult::interrupt();
          } else if (isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::WhileOp,
                         mlir::memref::LoadOp, mlir::memref::StoreOp>(
                         operation)) {
            return operation->emitError(
                       "We do not support this op for hir conversion."),
                   WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (walkresult.wasInterrupted()) {
      signalPassFailure();
      break;
    }
  }
  safelyEraseOps();
}

DictionaryAttr getHIRValueAttrs(DictionaryAttr hlsAttrs) {
  if (!hlsAttrs)
    return DictionaryAttr();
  Builder builder(hlsAttrs.getContext());
  SmallVector<NamedAttribute> attrs;

  int latency = 0;
  if (auto latencyAttr = hlsAttrs.getAs<IntegerAttr>("hls.INTERFACE_LATENCY"))
    latency = latencyAttr.getInt();
  attrs.push_back(
      builder.getNamedAttr("hir.delay", builder.getI64IntegerAttr(latency)));
  return builder.getDictionaryAttr(attrs);
}

std::optional<DictionaryAttr> getHIRMemrefAttrs(Operation *operation,
                                                DictionaryAttr hlsAttrs,
                                                MemRefType ty) {
  if (!hlsAttrs)
    return DictionaryAttr();
  Builder builder(hlsAttrs.getContext());
  SmallVector<NamedAttribute> attrs;
  for (auto kv : hlsAttrs)
    if (kv.getName().str().substr(0, 3) != "hls")
      attrs.push_back(kv);

  ArrayAttr portsAttr;
  bool isReg = false;
  if (ty.getNumElements() == 1)
    isReg = true;
  else if (auto dim = hlsAttrs.getAs<IntegerAttr>("hls.ARRAY_PARTITION_DIM"))
    if (dim == 0)
      isReg = true;

  std::string type = "";
  std::optional<NamedAttribute> rdPort;
  std::optional<NamedAttribute> wrPort;
  if (isReg) {
    rdPort = builder.getNamedAttr("rd_latency", builder.getI64IntegerAttr(0));
    wrPort = builder.getNamedAttr("wr_latency", builder.getI64IntegerAttr(1));
    type = "ram_2p";
  } else if (auto ty = hlsAttrs.getAs<StringAttr>("hls.BIND_STORAGE_TYPE")) {
    auto rdLatency = hlsAttrs.getAs<IntegerAttr>("hls.BIND_STORAGE_RD_LATENCY");
    auto wrLatency = hlsAttrs.getAs<IntegerAttr>("hls.BIND_STORAGE_WR_LATENCY");
    if (rdLatency)
      rdPort = builder.getNamedAttr("rd_latency", rdLatency);
    if (wrLatency)
      wrPort = builder.getNamedAttr("wr_latency", wrLatency);
    type = ty.str();
  } else if (auto ty =
                 hlsAttrs.getAs<StringAttr>("hls.INTERFACE_STORAGE_TYPE")) {
    auto rdLatency = hlsAttrs.getAs<IntegerAttr>("hls.INTERFACE_RD_LATENCY");
    auto wrLatency = hlsAttrs.getAs<IntegerAttr>("hls.INTERFACE_WR_LATENCY");
    if (rdLatency)
      rdPort = builder.getNamedAttr("rd_latency", rdLatency);
    if (wrLatency)
      wrPort = builder.getNamedAttr("wr_latency", wrLatency);
    type = ty.str();
  } else {
    operation->emitError("Could not determine memory type.");
    return std::nullopt;
  }
  if (type == "ram_1p") {
    if (!rdPort && !wrPort) {
      operation->emitError("Could not determine rd and/or wr port latency. "
                           "Original attrs : ")
          << hlsAttrs;
      return std::nullopt;
    }
    if (rdPort && wrPort)
      portsAttr =
          builder.getArrayAttr(builder.getDictionaryAttr({*rdPort, *wrPort}));
    else if (rdPort)
      portsAttr = builder.getArrayAttr(builder.getDictionaryAttr({*rdPort}));
    else
      portsAttr = builder.getArrayAttr(builder.getDictionaryAttr({*wrPort}));

  } else if (type == "ram_2p") {
    if (!rdPort || !wrPort) {
      operation->emitError(
          "Need both rd and wr port latency for simple dual port ram.");
      return std::nullopt;
    }
    portsAttr = builder.getArrayAttr({builder.getDictionaryAttr(*rdPort),
                                      builder.getDictionaryAttr(*wrPort)});
  } else if (type == "ram_t2p") {
    if (!rdPort || !wrPort) {
      operation->emitError(
          "Need both rd and wr port latency for true dual port ram.");
      return std::nullopt;
    }
    portsAttr =
        builder.getArrayAttr({builder.getDictionaryAttr({*rdPort, *wrPort}),
                              builder.getDictionaryAttr({*rdPort, *wrPort})});
  } else {
    operation->emitError("Unknown type of memory : ") << type;
    return std::nullopt;
  }

  attrs.push_back(builder.getNamedAttr("hir.memref.ports", portsAttr));

  if (isa<mlir::memref::AllocaOp>(operation)) {
    StringAttr memKind;
    if (isReg)
      memKind = builder.getStringAttr("reg");
    else if (auto impl = hlsAttrs.getAs<StringAttr>("hls.BIND_STORAGE_IMPL"))
      memKind = builder.getStringAttr(impl.strref().lower());
    if (!memKind)
      return operation->emitError("Could not determine memory impl."),
             std::nullopt;
    attrs.push_back(builder.getNamedAttr("mem_kind", memKind));
  }

  return builder.getDictionaryAttr(attrs);
}

std::optional<ArrayAttr> newArgOrResultAttrs(Operation *op,
                                             ArrayAttr originalAttrs,
                                             ArrayRef<Type> types) {
  if (!originalAttrs)
    return ArrayAttr();

  Builder builder(originalAttrs.getContext());
  SmallVector<Attribute> newArgAttrs;
  for (size_t i = 0; i < originalAttrs.size(); i++) {
    auto ty = types[i];
    auto argAttr = originalAttrs[i].dyn_cast<DictionaryAttr>();
    std::optional<DictionaryAttr> newAttr;
    if (auto memTy = ty.dyn_cast<mlir::MemRefType>()) {
      newAttr = getHIRMemrefAttrs(op, argAttr, memTy);
    } else if (ty.isIntOrFloat()) {
      newAttr = getHIRValueAttrs(argAttr);
    } else {
      assert(false && "unreachable");
    }
    if (!newAttr)
      return op->emitError("Could not get memref attr for arg ") << i,
             std::nullopt;
    newArgAttrs.push_back(*newAttr);
  }
  return builder.getArrayAttr(newArgAttrs);
}

LogicalResult HIRPragma::visitOp(mlir::func::FuncOp op) {
  if (!op->hasAttr("hwAccel"))
    return success();
  Builder builder(op);
  auto newArgAttr = newArgOrResultAttrs(
      op, op->getAttrOfType<ArrayAttr>("arg_attrs"), op.getArgumentTypes());
  if (!newArgAttr.has_value())
    return failure();
  if (*newArgAttr)
    op->setAttr("arg_attrs", *newArgAttr);
  else if (op.isDeclaration())
    return op->emitError("Could not find arg_attrs.");
  auto newResultAttr = newArgOrResultAttrs(
      op, op->getAttrOfType<ArrayAttr>("res_attrs"), op.getArgumentTypes());
  if (!newResultAttr)
    return failure();
  if (*newResultAttr)
    op->setAttr("res_attrs", *newResultAttr);
  else if (op.isDeclaration() && op.getNumResults() > 0 && !*newResultAttr)
    return op->emitError("Could not find res_attrs.");
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::memref::AllocaOp op) {
  auto newAttr =
      getHIRMemrefAttrs(op, op->getAttrDictionary(),
                        op.getMemref().getType().dyn_cast<MemRefType>());

  if (!newAttr)
    return failure();
  op->setAttrs(*newAttr);

  // Remove unnecessary load ops.
  // FIXME: This is too simple. It can not eliminate a load if another load to a
  // different address is in between. Use a map instead of one 'loadOp' variable
  // to handle that.
  // std::optional<mlir::affine::AffineLoadOp> loadOp;
  // for (auto *user : op.getMemref().getUsers()) {
  //  if (auto u = dyn_cast<mlir::affine::AffineLoadOp>(user)) {
  //    if (loadOp && loadOp->getIndices() == u.getIndices()) {
  //      u->replaceAllUsesWith(*loadOp);
  //      toErase.push_back(u);
  //    } else {
  //      loadOp = u;
  //    }
  //  } else if (auto u = dyn_cast<mlir::affine::AffineStoreOp>(user)) {
  //    loadOp = std::nullopt;
  //  } else
  //    return user->emitError("Only affine.load and affine.store are supported
  //    "
  //                           "by -hir-pragma pass.");
  //}
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::affine::AffineForOp op) {
  auto iiAttr = op->getAttrOfType<IntegerAttr>("hls.PIPELINE_II");
  auto unrollAttr = op->getAttrOfType<IntegerAttr>("hls.UNROLL_FACTOR");
  if (iiAttr) {
    op->setAttr("II", iiAttr);
    op->removeAttr("hls.PIPELINE_II");
  } else {
    op->emitRemark("No initiation interval specified.");
  }

  if (unrollAttr) {
    op->setAttr("UNROLL", unrollAttr);
    op->removeAttr("hls.UNROLL_FACTOR");
  }
  OpBuilder builder(op);
  auto lb = op.getLowerBound().getMap().getSingleConstantResult();
  auto ub = op.getUpperBound().getMap().getSingleConstantResult();
  auto step = op.getStep();
  if (ub < lb + step)
    return op.emitError("The for loop must have atleast one iteration.");
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::affine::AffineLoadOp op) {
  Builder builder(op);
  auto memref = op.getMemref();
  ArrayAttr portsAttr = getMemrefPortAttr(memref);
  auto rdPortNum = selectRdPort(memref);

  if (!rdPortNum)
    return op->emitError("Could not find a read port of the memref.");

  IntegerAttr resultDelay = builder.getI64IntegerAttr(
      helper::getMemrefPortRdLatency(portsAttr[*rdPortNum]).value());

  op->setAttr("result_delays", builder.getArrayAttr(resultDelay));
  op->setAttr("hir.memref_port", builder.getI64IntegerAttr(*rdPortNum));
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::affine::AffineStoreOp op) {
  Builder builder(op);
  auto memref = op.getMemref();
  auto wrPortNum = selectWrPort(memref);

  if (!wrPortNum)
    return op->emitError("Could not find a read port of the memref.");

  op->setAttr("hir.memref_port", builder.getI64IntegerAttr(*wrPortNum));
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::func::CallOp op) {
  if (op->getNumResults() == 0)
    return success();
  auto *calleeOp = getOperation().lookupSymbol(op.getCalleeAttr());
  if (!calleeOp)
    return op->emitError("Could not find callee for this call op.");

  SmallVector<Attribute> resultDelays;
  if (calleeOp->hasAttrOfType<ArrayAttr>("res_attrs"))
    for (auto resAttr : calleeOp->getAttrOfType<ArrayAttr>("res_attrs")) {
      auto delayAttr =
          resAttr.dyn_cast<DictionaryAttr>().getAs<IntegerAttr>("hir.delay");
      if (!delayAttr) {
        calleeOp->emitWarning("Could not find hir.delay attr.");
      }
      resultDelays.push_back(delayAttr);
    }
  else
    return calleeOp->emitError("could not find res_attr in the callee.");

  Builder builder(op);
  op->setAttr("result_delays", builder.getArrayAttr(resultDelays));
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::arith::NegFOp op) {
  OpBuilder builder(op);
  builder.setInsertionPoint(op->getParentOfType<mlir::func::FuncOp>());
  auto funcOp = builder.create<mlir::func::FuncOp>(
      builder.getUnknownLoc(), "neg_f32",
      FunctionType::get(builder.getContext(), {op.getResult().getType()},
                        op.getResult().getType()));
  funcOp.setPrivate();
  funcOp->setAttr("hwAccel", builder.getUnitAttr());

  auto zeroDelayAttr = builder.getDictionaryAttr(
      builder.getNamedAttr("hir.delay", builder.getI64IntegerAttr(0)));
  funcOp->setAttr("arg_attrs", builder.getArrayAttr(zeroDelayAttr));
  funcOp->setAttr("res_attrs", builder.getArrayAttr(zeroDelayAttr));
  funcOp->setAttr("argNames", builder.getStrArrayAttr({"a"}));
  funcOp->setAttr("resultNames", builder.getStrArrayAttr({"out"}));

  builder.setInsertionPoint(op);
  auto newOp = builder.create<mlir::func::CallOp>(op->getLoc(), funcOp,
                                                  op->getOperands());
  newOp->setAttr("result_delays", builder.getI64ArrayAttr({0}));
  op->replaceAllUsesWith(newOp);
  toErase.push_back(op);
  return success();
}

LogicalResult HIRPragma::visitOp(mlir::LLVM::UndefOp op) {
  for (auto *user : op.getResult().getUsers()) {
    if (!isa<mlir::affine::AffineStoreOp>(user))
      return user->emitError(
          "Only affine.store op uses for llvm.mlir.undef is allowed.");
    toErase.push_back(user);
  }
  toErase.push_back(op);
  return success();
}

LogicalResult HIRPragma::visitArithFOp(Operation *operation) {
  assert(operation->getNumResults() == 1);

  std::string opName;
  std::string typeStr =
      "f" +
      std::to_string(operation->getResult(0).getType().getIntOrFloatBitWidth());

  if (isa<mlir::arith::AddFOp>(operation))
    opName = "add_" + typeStr;
  else if (isa<mlir::arith::SubFOp>(operation))
    opName = "sub_" + typeStr;
  else if (isa<mlir::arith::MulFOp>(operation))
    opName = "mul_" + typeStr;
  else if (isa<mlir::arith::DivFOp>(operation))
    opName = "div_" + typeStr;
  else
    return operation->emitError("Unknown arith operation.");

  auto funcDecl =
      dyn_cast_or_null<mlir::func::FuncOp>(getOperation().lookupSymbol(opName));
  if (!funcDecl) {
    return operation->emitError("Could not find the decl of function ")
           << opName << " to lower following arith op.";
  }
  OpBuilder builder(operation);
  auto arithCallOp = builder.create<mlir::func::CallOp>(
      operation->getLoc(), funcDecl, operation->getOperands());
  operation->replaceAllUsesWith(arithCallOp);
  if (failed(visitOp(arithCallOp)))
    return failure();
  toErase.push_back(operation);
  return success();
}
//-----------------------------------------------------------------------------
std::unique_ptr<mlir::Pass> circt::createHIRPragmaPass() {
  return std::make_unique<HIRPragma>();
}
