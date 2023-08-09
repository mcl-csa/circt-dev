#include "circt/Dialect/HIR/IR/HIROpSyntax.h"
#include "circt/Dialect/HIR/IR/helper.h"
#define min(x, y) x > y ? x : y
//------------------------------------------------------------------------------
//---------------------------- Helper functions --------------------------------
//------------------------------------------------------------------------------

SmallVector<Type> getTypes(ArrayRef<OpAsmParser::Argument> args) {
  SmallVector<Type> types;
  for (auto arg : args)
    types.push_back(arg.type);
  return types;
}

static ParseResult parseDelayAttr(OpAsmParser &parser,
                                  SmallVectorImpl<DictionaryAttr> &attrsList) {
  NamedAttrList argAttrs;
  IntegerAttr delayAttr;
  auto *context = parser.getBuilder().getContext();
  if (succeeded(parser.parseOptionalKeyword("delay"))) {
    if (parser.parseAttribute(delayAttr, IntegerType::get(context, 64),
                              "hir.delay", argAttrs))
      return failure();
    attrsList.push_back(DictionaryAttr::get(context, argAttrs));
  } else {
    attrsList.push_back(helper::getDictionaryAttr(
        "hir.delay", helper::getI64IntegerAttr(context, 0)));
  }
  return success();
}

static ParseResult
parseMemrefPortsAttr(OpAsmParser &parser,
                     SmallVectorImpl<DictionaryAttr> &attrsList) {

  Attribute memrefPortsAttr;
  if (parser.parseKeyword("ports") || parser.parseAttribute(memrefPortsAttr))
    return failure();

  attrsList.push_back(
      helper::getDictionaryAttr("hir.memref.ports", memrefPortsAttr));

  return success();
}

static ParseResult
parseBusPortsAttr(OpAsmParser &parser,
                  SmallVectorImpl<DictionaryAttr> &attrsList) {

  llvm::SmallString<5> busPort;
  auto *context = parser.getBuilder().getContext();

  if (parser.parseKeyword("ports") || parser.parseLSquare())
    return failure();
  if (succeeded(parser.parseOptionalKeyword("send")))
    busPort = "send";
  else {
    if (failed(parser.parseOptionalKeyword("recv")))
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected 'send' or 'recv' port.");
    busPort = "recv";
  }
  if (parser.parseRSquare())
    return failure();

  attrsList.push_back(helper::getDictionaryAttr(
      "hir.bus.ports",
      ArrayAttr::get(context,
                     SmallVector<Attribute>({StringAttr::get(
                         parser.getBuilder().getContext(), busPort)}))));
  return success();
}

static ParseResult
parseOptionalIterArgs(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                      ArrayAttr &delays) {

  if (parser.parseOptionalKeyword("iter_args"))
    return success();

  SmallVector<Attribute> delayAttrs;
  if (parser.parseLParen())
    return failure();
  do {
    OpAsmParser::Argument arg;
    OpAsmParser::UnresolvedOperand operand;
    int64_t delay = 0;
    if (parser.parseArgument(arg) || parser.parseEqual() ||
        parser.parseOperand(operand) || parser.parseColonType(arg.type))
      return failure();
    if (succeeded(parser.parseOptionalKeyword("delay")))
      if (parser.parseInteger(delay))
        return failure();
    entryArgs.push_back(arg);
    operands.push_back(operand);
    delayAttrs.push_back(parser.getBuilder().getI64IntegerAttr(delay));
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRParen())
    return failure();

  delays = ArrayAttr::get(parser.getContext(), delayAttrs);
  return success();
}

static void printOptionalIterArgs(OpAsmPrinter &printer, OperandRange iterArgs,
                                  ArrayRef<BlockArgument> regionArgs,
                                  ArrayAttr iterArgDelays) {
  if (iterArgs.size() > 0) {
    printer << " iter_args(";
    for (size_t i = 0; i < iterArgs.size(); i++) {
      if (i > 0)
        printer << ",";
      printer << regionArgs[i] << "=" << iterArgs[i] << ": "
              << iterArgs[i].getType();
      auto delay = iterArgDelays[i].dyn_cast<IntegerAttr>().getInt();
      if (delay > 0)
        printer << "delay " << delay;
    }
    printer << ")";
  }
}

//-----------------------------------------------------------------------------
//------------------- Parser/Printer used in assembly-format ------------------
//-----------------------------------------------------------------------------

ParseResult parseTimeAndOffset(mlir::OpAsmParser &parser,
                               mlir::OpAsmParser::UnresolvedOperand &tstart,
                               IntegerAttr &delay) {

  // parse tstart.
  if (parser.parseOperand(tstart))
    return failure();

  // early exit if no offsets.
  if (parser.parseOptionalPlus()) {
    delay = parser.getBuilder().getI64IntegerAttr(0);
    return success();
  }

  // Parse offset
  int64_t offset;
  if (parser.parseInteger(offset))
    return failure();
  delay = parser.getBuilder().getI64IntegerAttr(offset);

  return success();
}

void printTimeAndOffset(OpAsmPrinter &printer, Operation *op, Value tstart,
                        IntegerAttr offset) {
  printer << tstart;

  if (offset.getInt() == 0)
    return;
  printer << " + " << offset.getInt();
}

ParseResult parseOptionalArrayAccess(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &varAddrs,
    ArrayAttr &constAddrs) {
  // If constAddr[i] is -ve then its an operand and the operand is at
  // varAddrs[-constAddr[i]-1], i.e. constAddr[i]>=0 is const and <0 is operand.
  llvm::SmallVector<Attribute, 4> tempConstAddrs;
  // early exit.
  if (parser.parseOptionalLSquare())
    return success();
  do {
    int val;
    auto *context = parser.getBuilder().getContext();
    OpAsmParser::UnresolvedOperand var;
    mlir::OptionalParseResult result = parser.parseOptionalInteger(val);
    if (result.hasValue() && !result.getValue()) {
      tempConstAddrs.push_back(helper::getI64IntegerAttr(context, val));
    } else if (!parser.parseOperand(var)) {
      varAddrs.push_back(var);
      tempConstAddrs.push_back(helper::getI64IntegerAttr(
          parser.getBuilder().getContext(), -varAddrs.size()));
    } else
      return failure();
  } while (!parser.parseOptionalComma());
  constAddrs = ArrayAttr::get(parser.getBuilder().getContext(), tempConstAddrs);
  if (parser.parseRSquare())
    return failure();
  return success();
}

void printOptionalArrayAccess(OpAsmPrinter &printer, Operation *op,
                              OperandRange varAddrs, ArrayAttr constAddrs) {
  if (constAddrs.size() == 0)
    return;
  printer << "[";
  for (size_t i = 0; i < constAddrs.size(); i++) {
    int idx = constAddrs[i].dyn_cast<IntegerAttr>().getInt();
    if (i > 0)
      printer << ", ";
    if (idx >= 0)
      printer << idx;
    else
      printer << varAddrs[-idx - 1];
  }
  printer << "]";
}

ParseResult parseOptionalArrayAccessTypes(mlir::OpAsmParser &parser,
                                          ArrayAttr &constAddrs,
                                          SmallVectorImpl<Type> &varAddrTypes) {
  // Early exit if no address types.
  if (parser.parseOptionalLSquare())
    return success();
  // Parse the types list.
  int i = 0;
  do {
    int idx = constAddrs[i].dyn_cast<IntegerAttr>().getInt();
    i++;
    Type t;
    if (!parser.parseOptionalKeyword("const"))
      t = IndexType::get(parser.getBuilder().getContext());
    else if (parser.parseType(t))
      return failure();

    if (idx < 0)
      varAddrTypes.push_back(t);
    else if (!t.isa<IndexType>())
      return parser.emitError(parser.getCurrentLocation(),
                              "Expected index type");
  } while (!parser.parseOptionalComma());

  // Finish parsing.
  if (parser.parseRSquare())
    return failure();

  return success();
}

void printOptionalArrayAccessTypes(OpAsmPrinter &printer, Operation *op,
                                   ArrayAttr constAddrs,
                                   TypeRange varAddrTypes) {
  if (constAddrs.size() == 0)
    return;
  printer << "[";
  for (size_t i = 0; i < constAddrs.size(); i++) {
    if (i > 0)
      printer << ", ";
    int idx = constAddrs[i].dyn_cast<IntegerAttr>().getInt();
    if (idx >= 0)
      printer << "const";
    else if (varAddrTypes[-idx - 1].isa<IndexType>())
      printer << "const";
    else
      printer << varAddrTypes[-idx - 1];
  }
  printer << "]";
}

ParseResult parseMemrefAndElementType(OpAsmParser &parser, Type &memrefTy,
                                      SmallVectorImpl<Type> &idxTypes,
                                      Type &elementTy) {
  auto memTyLoc = parser.getCurrentLocation();
  if (parser.parseType(memrefTy))
    return failure();
  auto memTy = memrefTy.dyn_cast<hir::MemrefType>();
  if (!memTy)
    return parser.emitError(memTyLoc, "Expected hir.memref type!");
  auto builder = parser.getBuilder();
  auto *context = builder.getContext();
  auto shape = memTy.getShape();
  auto dimKinds = memTy.getDimKinds();

  for (int i = 0; i < (int)dimKinds.size(); i++) {
    if (dimKinds[i] == hir::BANK)
      idxTypes.push_back(IndexType::get(context));
    else {
      idxTypes.push_back(
          IntegerType::get(context, min(1, helper::clog2(shape[i]))));
    }
  }

  elementTy = memTy.getElementType();
  return success();
}

void printMemrefAndElementType(OpAsmPrinter &printer, Operation *,
                               Type memrefTy, TypeRange, Type) {
  printer << memrefTy;
}

ParseResult parseTypeAndDelayList(mlir::OpAsmParser &parser,
                                  SmallVectorImpl<Type> &typeList,
                                  ArrayAttr &delayList) {
  SmallVector<Attribute> delayAttrArray;
  do {
    Type ty;
    int delay;
    if (parser.parseType(ty))
      return failure();
    typeList.push_back(ty);
    if (succeeded(parser.parseOptionalKeyword("delay"))) {
      if (parser.parseInteger(delay))
        return failure();
      delayAttrArray.push_back(parser.getBuilder().getI64IntegerAttr(delay));
    } else if (helper::isBuiltinSizedType(ty)) {
      delayAttrArray.push_back(parser.getBuilder().getI64IntegerAttr(0));
    } else {
      delayAttrArray.push_back(Attribute());
    }
  } while (succeeded(parser.parseOptionalComma()));
  delayList = parser.getBuilder().getArrayAttr(delayAttrArray);
  return success();
}

void printTypeAndDelayList(mlir::OpAsmPrinter &printer, TypeRange typeList,
                           ArrayAttr delayList) {
  for (uint64_t i = 0; i < typeList.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << typeList[i];
    if (!delayList[i])
      continue;
    int delay = delayList[i].dyn_cast<IntegerAttr>().getInt();
    if (delay != 0)
      printer << " delay " << delay;
  }
}

ParseResult parseBinOpOperandsAndResultType(mlir::OpAsmParser &parser,
                                            Type &resultTy, Type &op1Ty,
                                            Type &op2Ty) {
  if (parser.parseType(resultTy))
    return failure();
  op1Ty = resultTy;
  op2Ty = resultTy;
  return success();
}

void printBinOpOperandsAndResultType(mlir::OpAsmPrinter &printer, Operation *,
                                     Type resultTy, Type, Type) {
  printer << resultTy;
}

ParseResult parseCopyType(mlir::OpAsmParser &parser, Type &destTy, Type srcTy) {
  destTy = srcTy;
  return success();
}
void printCopyType(mlir::OpAsmPrinter &, Operation *, Type, Type) {}

ParseResult parseCopyType(mlir::OpAsmParser &parser, Type &destTy,
                          TypeAttr srcTyAttr) {
  destTy = srcTyAttr.getValue();
  return success();
}
void printCopyType(mlir::OpAsmPrinter &, Operation *, Type, TypeAttr) {}

ParseResult parseWithSSANames(mlir::OpAsmParser &parser,
                              mlir::NamedAttrList &attrDict) {

  if (parser.parseOptionalAttrDict(attrDict))
    return failure();

  // If the attribute dictionary contains no 'names' attribute, infer it from
  // the SSA name (if specified).
  bool hadNames = llvm::any_of(
      attrDict, [](NamedAttribute attr) { return attr.getName() == "names"; });

  // If there was no name specified, check to see if there was a useful name
  // specified in the asm file.
  if (hadNames || parser.getNumResults() == 0)
    return success();

  SmallVector<StringRef, 4> names;
  auto *context = parser.getBuilder().getContext();

  for (size_t i = 0, e = parser.getNumResults(); i != e; ++i) {
    auto resultName = parser.getResultName(i);
    StringRef nameStr;
    if (!resultName.first.empty() && !isdigit(resultName.first[0]))
      nameStr = resultName.first;

    names.push_back(nameStr);
  }

  auto namesAttr = parser.getBuilder().getStrArrayAttr(names);
  attrDict.push_back(
      NamedAttribute(StringAttr::get(context, "names"), namesAttr));
  return success();
}

void printWithSSANames(mlir::OpAsmPrinter &printer, Operation *op,
                       mlir::DictionaryAttr attrDict) {

  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  auto names = op->getAttrOfType<ArrayAttr>("names");
  bool namesDisagree;
  if (names)
    namesDisagree = names.size() != op->getNumResults();
  else
    namesDisagree = true;

  SmallString<32> resultNameStr;
  for (size_t i = 0, e = op->getNumResults(); i != e && !namesDisagree; ++i) {
    resultNameStr.clear();
    llvm::raw_svector_ostream tmpStream(resultNameStr);
    printer.printOperand(op->getResult(i), tmpStream);

    auto expectedName = names[i].dyn_cast<StringAttr>();
    if (!expectedName ||
        tmpStream.str().drop_front() != expectedName.getValue()) {
      if (!expectedName.getValue().empty())
        namesDisagree = true;
    }
  }
  SmallVector<StringRef, 10> elidedAttrs = {
      "offset",         "delay",        "ports",
      "port",           "result_attrs", "callee",
      "funcTy",         "portNums",     "operand_segment_sizes",
      "index",          "mem_type",     "argNames",
      "instance_name",  "value",        "mem_kind",
      "iter_arg_delays"};
  if (!namesDisagree)
    elidedAttrs.push_back("names");
  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

ParseResult parseFunctionType(OpAsmParser &parser, TypeAttr &funcTyAttr,
                              SmallVectorImpl<Type> &operandTypes,
                              SmallVectorImpl<Type> &resultTypes) {
  auto errLoc = parser.getCurrentLocation();
  hir::FuncType funcTy;
  if (parser.parseType<hir::FuncType>(funcTy))
    return parser.emitError(errLoc, "Expected !hir.func<...> type.");

  for (auto operandTy : funcTy.getInputTypes())
    operandTypes.push_back(operandTy);
  for (auto resultTy : funcTy.getFunctionType().getResults())
    resultTypes.push_back(resultTy);
  funcTyAttr = TypeAttr::get(funcTy);
  return success();
}

void printFunctionType(OpAsmPrinter &printer, Operation *, TypeAttr funcTyAttr,
                       TypeRange, TypeRange) {
  printer << funcTyAttr;
}

//------------------------------------------------------------------------------
//-------------------------- Op parsers and printers ---------------------------
//------------------------------------------------------------------------------

/// IfOp
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand cond;
  OpAsmParser::UnresolvedOperand tstart;
  OpAsmParser::Argument timevar;
  IntegerAttr offsetAttr;
  SmallVector<Type> resultTypes;
  ArrayAttr resultAttrs;

  // parse the boolean condition
  if (parser.parseOperand(cond))
    return failure();

  // parse time.
  if (parser.parseKeyword("at") || parser.parseKeyword("time") ||
      parser.parseLParen() || parser.parseArgument(timevar) ||
      parser.parseEqual())
    return failure();
  timevar.type = TimeType::get(parser.getContext());

  if (failed(parseTimeAndOffset(parser, tstart, offsetAttr)))
    return failure();

  if (parser.parseRParen())
    return failure();

  if (succeeded(parser.parseOptionalArrow()))
    if (parser.parseLParen() ||
        parseTypeAndDelayList(parser, resultTypes, resultAttrs) ||
        parser.parseRParen())
      return failure();
  auto *context = parser.getBuilder().getContext();
  if (parser.resolveOperand(cond, IntegerType::get(context, 1),
                            result.operands))
    return failure();

  if (parser.resolveOperand(tstart, TimeType::get(context), result.operands))
    return failure();

  if (offsetAttr)
    result.addAttribute("offset", offsetAttr);
  if (resultTypes.size() > 0)
    result.addAttribute("result_attrs", resultAttrs);
  // Add outputs.
  if (resultTypes.size() > 0)
    result.addTypes(resultTypes);

  Region *ifBody = result.addRegion();
  Region *elseBody = result.addRegion();

  if (parser.parseRegion(*ifBody, timevar))
    return failure();
  if (parser.parseKeyword("else"))
    return failure();
  if (parser.parseRegion(*elseBody, timevar))
    return failure();

  if (failed(parseWithSSANames(parser, result.attributes)))
    return failure();

  // IfOp::ensureTerminator(*ifBody, builder, result.location);
  return success();
}

void IfOp::print(OpAsmPrinter &printer) {

  printer << " " << this->condition();

  printer << " at time(" << this->if_region().getArgument(0) << " = ";
  printTimeAndOffset(printer, this->getOperation(), this->tstart(),
                     this->offsetAttr());
  printer << ")";

  if (this->results().size() > 0) {
    printer << " -> (";
    printTypeAndDelayList(printer, this->getResultTypes(),
                          this->result_attrs().getValue());
    printer << ")";
  }

  printer.printRegion(this->if_region(),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printer << "else";
  printer.printRegion(this->else_region(), false, true);
  printWithSSANames(printer, this->getOperation(),
                    this->getOperation()->getAttrDictionary());
}

// WhileOp
// Example:
// hir.while(%b) at iter_time(%tw = %t + 1){}
ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::Argument iterTimeVar;
  OpAsmParser::UnresolvedOperand conditionVar;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgOperands;
  SmallVector<OpAsmParser::Argument> iterArgs;
  ArrayAttr iterArgDelays;
  OpAsmParser::UnresolvedOperand tstart;
  IntegerAttr offset;

  if (parser.parseOperand(conditionVar))
    return failure();

  if (parseOptionalIterArgs(parser, iterArgs, iterArgOperands, iterArgDelays))
    return failure();
  if (parser.parseKeyword("iter_time") || parser.parseLParen() ||
      parser.parseArgument(iterTimeVar) || parser.parseEqual())
    return failure();
  iterTimeVar.type = TimeType::get(parser.getContext());
  if (parseTimeAndOffset(parser, tstart, offset) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(conditionVar, parser.getBuilder().getI1Type(),
                            result.operands))
    return failure();

  if (!iterArgOperands.empty())
    if (parser.resolveOperands(iterArgOperands, getTypes(iterArgs),
                               parser.getNameLoc(), result.operands))
      return failure();
  if (parser.resolveOperand(
          tstart, hir::TimeType::get(parser.getBuilder().getContext()),
          result.operands))
    return failure();

  if (offset)
    result.addAttribute("offset", offset);

  if (!iterArgOperands.empty())
    result.addAttribute("iter_arg_delays", iterArgDelays);

  iterArgs.push_back(iterTimeVar);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, iterArgs))
    return failure();

  // Parse the attr-dict
  if (parseWithSSANames(parser, result.attributes))
    return failure();
  result.addTypes(getTypes(iterArgs));
  return success();
}

void WhileOp::print(OpAsmPrinter &printer) {
  printer << " " << this->condition();
  printOptionalIterArgs(
      printer, this->iter_args(), this->body().front().getArguments(),
      this->getOperation()->getAttrOfType<ArrayAttr>("iter_arg_delays"));
  printer << " iter_time(" << this->getIterTimeVar() << " = ";
  printTimeAndOffset(printer, this->getOperation(), this->tstart(),
                     this->offsetAttr());
  printer << ")";
  printer.printRegion(this->getOperation()->getRegion(0),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printWithSSANames(printer, this->getOperation(),
                    this->getOperation()->getAttrDictionary());
}

// ForOp.
// Example:
// hir.for %i = %l to %u step %s iter_time(%ti = %t + 1){...}
ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *context = parser.getBuilder().getContext();

  OpAsmParser::Argument iterTimeVar;
  OpAsmParser::Argument inductionVar;
  OpAsmParser::UnresolvedOperand lb;
  OpAsmParser::UnresolvedOperand ub;
  OpAsmParser::UnresolvedOperand step;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgOperands;
  SmallVector<OpAsmParser::Argument> iterArgs;
  ArrayAttr iterArgDelays;

  OpAsmParser::UnresolvedOperand tstart;
  IntegerAttr offset;

  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVar, true) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  if (parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") ||
      parser.parseOperand(step))
    return failure();

  // Parse iter_args.
  if (parseOptionalIterArgs(parser, iterArgs, iterArgOperands, iterArgDelays))
    return failure();

  // Parse iter-time.
  if (parser.parseKeyword("iter_time") || parser.parseLParen())
    return failure();

  if (parser.parseArgument(iterTimeVar) || parser.parseEqual())
    return failure();
  iterTimeVar.type = TimeType::get(parser.getContext());

  if (parseTimeAndOffset(parser, tstart, offset) || parser.parseRParen())
    return failure();

  // resolve the loop bounds.
  if (parser.resolveOperand(lb, inductionVar.type, result.operands) ||
      parser.resolveOperand(ub, inductionVar.type, result.operands) ||
      parser.resolveOperand(step, inductionVar.type, result.operands))
    return failure();

  SmallVector<Type> iterArgTypes = getTypes(iterArgs);
  if (!iterArgOperands.empty() &&
      failed(parser.resolveOperands(iterArgOperands, iterArgTypes,
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  // resolve optional tstart and offset.
  if (parser.resolveOperand(tstart, iterTimeVar.type, result.operands))
    return failure();
  if (offset)
    result.addAttribute("offset", offset);
  if (iterArgOperands.size() > 0)
    result.addAttribute("iter_arg_delays", iterArgDelays);

  SmallVector<Type> resultTypes(iterArgTypes);
  iterArgs.push_back(inductionVar);
  iterArgs.push_back(iterTimeVar);
  iterArgTypes.push_back(inductionVar.type);
  iterArgTypes.push_back(TimeType::get(context));
  resultTypes.push_back(TimeType::get(context));

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, iterArgs))
    return failure();

  // Parse the attr-dict
  if (parseWithSSANames(parser, result.attributes))
    return failure();

  // ForOp result is the time at which last iteration calls next_iter.
  result.addTypes(resultTypes);

  // ForOp::ensureTerminator(*body, builder, result.location);
  return success();
}
void ForOp::print(OpAsmPrinter &printer) {
  printer << " " << this->getInductionVar() << " : "
          << this->getInductionVar().getType() << " = " << this->lb() << " to "
          << this->ub() << " step " << this->step();
  printOptionalIterArgs(
      printer, this->iter_args(), this->body().front().getArguments(),
      this->getOperation()->getAttrOfType<ArrayAttr>("iter_arg_delays"));

  printer << " iter_time( " << this->getIterTimeVar() << " = ";
  printTimeAndOffset(printer, this->getOperation(), this->tstart(),
                     this->offsetAttr());
  printer << ")";

  printer.printRegion(this->getOperation()->getRegion(0),
                      /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);

  printWithSSANames(printer, this->getOperation(),
                    this->getOperation()->getAttrDictionary());
}

Region &ForOp::getLoopBody() { return body(); }

/// FuncOp
/// Example:
/// hir.def @foo at %t (%x :i32 delay 1, %y: f32) ->(%out: i1 delay 4){}
ParseResult parseArgList(OpAsmParser &parser,
                         SmallVectorImpl<OpAsmParser::Argument> &args,
                         SmallVectorImpl<DictionaryAttr> &argAttrs) {

  auto *context = parser.getBuilder().getContext();
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    while (1) {
      // Parse operand and type
      auto operandLoc = parser.getCurrentLocation();
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg, true))
        return failure();
      args.push_back(arg);
      // Parse argAttr
      if (helper::isBuiltinSizedType(arg.type)) {
        if (parseDelayAttr(parser, argAttrs))
          return failure();
      } else if (arg.type.isa<hir::TimeType>()) {
        argAttrs.push_back(
            DictionaryAttr::get(context, SmallVector<NamedAttribute>({})));
      } else if (arg.type.isa<hir::MemrefType>()) {
        if (parseMemrefPortsAttr(parser, argAttrs))
          return failure();
      } else if (helper::isBusLikeType(arg.type)) {
        if (parseBusPortsAttr(parser, argAttrs))
          return failure();
      } else
        return parser.emitError(operandLoc, "Unsupported type.");

      if (failed(parser.parseOptionalComma()))
        break;
    }
    if (parser.parseRParen())
      return failure();
  }
  return success();
}

ParseResult
parseFuncSignature(OpAsmParser &parser, hir::FuncType &funcTy,
                   SmallVectorImpl<OpAsmParser::Argument> &args,
                   SmallVectorImpl<OpAsmParser::Argument> &results) {
  SmallVector<DictionaryAttr> argAttrs;
  SmallVector<DictionaryAttr> resultAttrs;
  // parse args
  if (parseArgList(parser, args, argAttrs))
    return failure();

  // If result types present then parse them.
  if (succeeded(parser.parseOptionalArrow()))
    if (parseArgList(parser, results, resultAttrs))
      return failure();

  funcTy = hir::FuncType::get(parser.getBuilder().getContext(), getTypes(args),
                              argAttrs, getTypes(results), resultAttrs);
  return success();
}

ParseResult parseFuncDecl(OpAsmParser &parser, OperationState &result,
                          SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                          hir::FuncType &funcTy) {

  SmallVector<OpAsmParser::Argument, 4> resultArgs;
  OpAsmParser::Argument tstart;
  auto &builder = parser.getBuilder();
  // Parse the name as a symbol.
  StringAttr functionName;
  if (parser.parseSymbolName(functionName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();
  // Parse tstart.
  if (parser.parseKeyword("at") || parser.parseArgument(tstart))
    return failure();
  tstart.type = TimeType::get(parser.getContext());
  //  Parse the function signature.
  if (parseFuncSignature(parser, funcTy, entryArgs, resultArgs))
    return failure();
  entryArgs.push_back(tstart);

  result.addAttribute("funcTy", TypeAttr::get(funcTy));

  // Add the attributes for FunctionLike interface.
  auto functionTy = funcTy.getFunctionType();
  result.addAttribute(mlir::function_interface_impl::getTypeAttrName(),
                      TypeAttr::get(functionTy));

  SmallVector<DictionaryAttr> argAttrs;
  for (DictionaryAttr attr : funcTy.getInputAttrs()) {
    argAttrs.push_back(attr);
  };
  argAttrs.push_back(builder.getDictionaryAttr(SmallVector<NamedAttribute>()));
  mlir::function_interface_impl::addArgAndResultAttrs(builder, result, argAttrs,
                                                      funcTy.getResultAttrs());

  return success();
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  hir::FuncType funcTy;

  auto &builder = parser.getBuilder();

  if (parseFuncDecl(parser, result, entryArgs, funcTy))
    return failure();

  // Parse the function body.
  auto *body = result.addRegion();
  SmallVector<Type> entryArgTypes;
  auto functionTy = funcTy.getFunctionType();
  for (auto ty : functionTy.getInputs()) {
    entryArgTypes.push_back(ty);
  }

  if (parser.parseRegion(*body, entryArgs))
    return failure();

  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();

  FuncOp::ensureTerminator(*body, builder, result.location);
  return success();
}

ParseResult FuncExternOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  hir::FuncType funcTy;
  if (parseFuncDecl(parser, result, entryArgs, funcTy))
    return failure();
  // Parse the function body.
  auto *body = result.addRegion();
  SmallVector<Type> entryArgTypes;
  auto functionTy = funcTy.getFunctionType();
  for (auto ty : functionTy.getInputs()) {
    entryArgTypes.push_back(ty);
  }
  auto &builder = parser.getBuilder();
  body->push_back(new Block);
  body->front().addArguments(
      entryArgTypes,
      SmallVector<Location>(entryArgTypes.size(), builder.getUnknownLoc()));
  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();
  FuncExternOp::ensureTerminator(*body, builder, result.location);
  return success();
}

static void printArgList(OpAsmPrinter &printer, ArrayRef<BlockArgument> args,
                         ArrayRef<Type> argTypes,
                         ArrayRef<DictionaryAttr> argAttrs) {
  assert(args.size() == argTypes.size() + 1);
  printer << "(";
  for (unsigned i = 0; i < argTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << args[i] << " : " << argTypes[i];

    if (helper::isBuiltinSizedType(argTypes[i])) {
      auto delay = helper::getHIRDelayAttr(argAttrs[i]);
      if (delay)
        printer << " delay " << delay;
    } else if (argTypes[i].isa<hir::MemrefType>()) {
      printer << " ports " << helper::extractMemrefPortsFromDict(argAttrs[i]);
    } else if (helper::isBusLikeType(argTypes[i])) {
      printer << " ports [" << helper::extractBusPortFromDict(argAttrs[i])
              << "]";
    }
  }
  printer << ")";
}

static void printArgList(OpAsmPrinter &printer, ArrayAttr argNames,
                         ArrayRef<Type> argTypes,
                         ArrayRef<DictionaryAttr> argAttrs) {
  printer << "(";
  for (unsigned i = 0; i < argTypes.size(); i++) {
    if (i > 0)
      printer << ", ";
    printer << "%" << argNames[i].dyn_cast<StringAttr>().getValue() << " : "
            << argTypes[i];

    if (helper::isBuiltinSizedType(argTypes[i])) {
      auto delay = helper::getHIRDelayAttr(argAttrs[i]);
      if (delay)
        printer << " delay " << delay;
    } else if (argTypes[i].isa<hir::MemrefType>()) {
      printer << " ports " << helper::extractMemrefPortsFromDict(argAttrs[i]);
    } else if (helper::isBusLikeType(argTypes[i])) {
      printer << " ports [" << helper::extractBusPortFromDict(argAttrs[i])
              << "]";
    }
  }
  printer << ")";
}

void printFuncSignature(OpAsmPrinter &printer, hir::FuncType funcTy,
                        ArrayRef<BlockArgument> args, ArrayAttr resultNames) {
  auto inputTypes = funcTy.getInputTypes();
  auto resultTypes = funcTy.getResultTypes();
  ArrayRef<DictionaryAttr> inputAttrs = funcTy.getInputAttrs();
  ArrayRef<DictionaryAttr> resultAttrs = funcTy.getResultAttrs();

  printArgList(printer, args, inputTypes, inputAttrs);

  if (resultTypes.size() == 0)
    return;

  printer << " -> ";

  printArgList(printer, resultNames, resultTypes, resultAttrs);
}

void FuncExternOp::print(OpAsmPrinter &printer) {
  // Print function name, signature, and control.
  auto args = this->getRegion().front().getArguments();
  printer << " ";
  printer.printSymbolName(this->sym_name());
  printer << " at " << args.back() << " ";
  printFuncSignature(printer, this->getFuncType(), args,
                     this->resultNames().value_or(ArrayAttr()));
  printer.printOptionalAttrDict(
      this->getOperation()->getAttrs(),
      {"funcTy", "function_type", "arg_attrs", "res_attrs", "sym_name"});
}

void FuncOp::print(OpAsmPrinter &printer) {
  // Print function name, signature, and control.
  printer << " ";
  printer.printSymbolName(this->sym_name());
  Region &body = this->getOperation()->getRegion(0);
  printer << " at "
          << body.front().getArgument(body.front().getNumArguments() - 1)
          << " ";

  printFuncSignature(printer, this->getFuncType(),
                     this->getRegion().front().getArguments(),
                     this->resultNames().value_or(ArrayAttr()));

  printer.printRegion(body, /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict(
      this->getOperation()->getAttrs(),
      {"funcTy", "function_type", "arg_attrs", "res_attrs", "sym_name"});
}

/// TensorInsertOp parser and printer.
/// Syntax: hir.tensor.insert %element into %tensor[%c0, %c1]
/// custom<WithSSANames>(attr-dict): type($res)
ParseResult BusTensorInsertElementOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  OpAsmParser::UnresolvedOperand element;
  OpAsmParser::UnresolvedOperand inputTensor;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  hir::BusTensorType resTy;
  auto builder = parser.getBuilder();
  if (parser.parseOperand(element) || parser.parseKeyword("into") ||
      parser.parseOperand(inputTensor))
    return failure();

  if (parser.parseOperandList(indices, -1,
                              mlir::OpAsmParser::Delimiter::Square))
    return failure();

  if (parseWithSSANames(parser, result.attributes))
    return failure();
  if (parser.parseColonType(resTy))
    return failure();

  if (parser.resolveOperand(
          element,
          hir::BusType::get(parser.getContext(), resTy.getElementType()),
          result.operands))
    return failure();
  if (parser.resolveOperand(inputTensor, resTy, result.operands))
    return failure();
  if (parser.resolveOperands(indices, builder.getIndexType(), result.operands))
    return failure();

  result.addTypes(resTy);
  return success();
}

void BusTensorInsertElementOp::print(OpAsmPrinter &printer) {
  printer << " " << this->element() << " into " << this->tensor();
  printer << "[";
  printer.printOperands(this->indices());
  printer << "]";
  printWithSSANames(printer, this->getOperation(),
                    this->getOperation()->getAttrDictionary());
  printer << " : ";
  printer << this->res().getType();
}

/// BusMapOp parser and printer
ParseResult BusMapOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::Argument regionArg;
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseArgument(regionArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    operands.push_back(operand);
    regionArgs.push_back(regionArg);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseColon())
    return failure();

  mlir::FunctionType funcTy;
  if (parser.parseType(funcTy))
    return failure();

  for (size_t i = 0; i < funcTy.getNumInputs(); i++) {
    auto ty = funcTy.getInputs()[i].dyn_cast<hir::BusType>();
    if (!ty)
      return parser.emitError(parser.getCurrentLocation(),
                              "All input types must be hir.bus type.");
    regionArgs[i].type = ty.getElementType();
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  if (failed(parser.resolveOperands(operands, funcTy.getInputs(),
                                    parser.getNameLoc(), result.operands)))
    return failure();

  result.addTypes(funcTy.getResults());
  return success();
}

void BusMapOp::print(OpAsmPrinter &printer) {
  printer << " (";
  for (size_t i = 0; i < this->getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << this->body().front().getArgument(i) << " = "
            << this->operands()[i];
  }
  printer << ") : (";
  for (size_t i = 0; i < this->getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << this->operands()[i].getType();
  }
  printer << ") -> (";
  for (size_t i = 0; i < this->getNumResults(); i++) {
    auto result = this->getResult(i);
    if (i > 0)
      printer << ",";
    printer << result.getType();
  }
  printer << ")";
  printer.printRegion(this->body(), false, true);
}

/// BusTensorMapOp parser and printer
ParseResult BusTensorMapOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseLParen())
    return failure();

  do {
    OpAsmParser::Argument regionArg;
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseArgument(regionArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    operands.push_back(operand);
    regionArgs.push_back(regionArg);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseColon())
    return failure();

  mlir::FunctionType funcTy;
  if (parser.parseType(funcTy))
    return failure();

  for (size_t i = 0; i < funcTy.getNumInputs(); i++) {
    auto ty = funcTy.getInputs()[i].dyn_cast<hir::BusType>();
    if (!ty)
      return parser.emitError(parser.getCurrentLocation(),
                              "All input types must be hir.bus type.");
    regionArgs[i].type = ty.getElementType();
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  if (failed(parser.resolveOperands(operands, funcTy.getInputs(),
                                    parser.getNameLoc(), result.operands)))
    return failure();

  result.addTypes(funcTy.getResults());
  return success();
}

void BusTensorMapOp::print(OpAsmPrinter &printer) {
  printer << " (";
  for (size_t i = 0; i < this->getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << this->body().front().getArgument(i) << " = "
            << this->operands()[i];
  }
  printer << ") : (";
  for (size_t i = 0; i < this->getNumOperands(); i++) {
    if (i > 0)
      printer << ",";
    printer << this->operands()[i].getType();
  }
  printer << ") -> (";
  for (size_t i = 0; i < this->getNumResults(); i++) {
    auto result = this->getResult(i);
    if (i > 0)
      printer << ",";
    printer << result.getType();
  }
  printer << ")";
  printer.printRegion(this->body(), false, true);
}

LogicalResult hir::FuncExternOp::verifyType() { return success(); }

LogicalResult hir::FuncOp::verifyType() {

  auto type = this->function_typeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
  return success();
}
