#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include <mlir/IR/IRMapping.h>

using namespace circt;

class FuncToHWModulePortMap {
public:
  void addFuncInput(StringAttr name, hw::PortInfo::Direction direction,
                    Type type);
  void addFuncResult(StringAttr name, Type type);
  void addClk(OpBuilder &);
  void addReset(OpBuilder &);
  ArrayRef<hw::PortInfo> getPortInfoList();
  hw::PortInfo getPortInfoForFuncInput(size_t inputArgNum);

private:
  size_t hwModuleInputArgNum = 0;
  size_t hwModuleResultArgNum = 0;
  SmallVector<hw::PortInfo> portInfoList;
  SmallVector<hw::PortInfo> mapFuncInputToHWPortInfo;
};

bool isRecvBus(DictionaryAttr busAttr);

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, SmallVector<Value, 4> args);

FuncToHWModulePortMap getHWModulePortMap(OpBuilder &builder,
                                         mlir::Location errorLoc,
                                         hir::FuncType funcTy,
                                         ArrayAttr inputNames,
                                         ArrayAttr resultNames);
void copyHIRAttrs(hir::CallOp, hw::InstanceOp);
Operation *constantX(OpBuilder &, Type);
Operation *
getConstantXArray(OpBuilder &builder, Type hwTy,
                  DenseMap<Value, SmallVector<Value>> &mapArrayToElements);

/// Convert to params for hw module. These become parameters in the Verilog
/// module. ignoreDefaultParameters = true is used with the module definition to
/// remove default parameter values. All parameter values are assigned at the
/// call cite.
ArrayAttr getHWParams(Operation *op, bool ignoreDefaultParameters = false);

Value getDelayedValue(OpBuilder &builder, Value input, int64_t delay,
                      std::optional<StringRef> name, Location loc, Value clk,
                      Value reset);

Value convertToNamedValue(OpBuilder &builder, StringRef name, Value val);
Value convertToOptionalNamedValue(OpBuilder &builder,
                                  std::optional<StringRef> name, Value val);
SmallVector<Value> insertBusMapLogic(OpBuilder &builder, Block &bodyBlock,
                                     ArrayRef<Value> operands);

Value insertConstArrayGetLogic(OpBuilder &builder, Value arr, int idx);

Value getClkFromHWModule(hw::HWModuleOp op);
Value getResetFromHWModule(hw::HWModuleOp op);

class HIRToHWMapping {
private:
  DenseMap<Value, Value> mapHIRToHWValue;

public:
  Value lookup(Value hirValue) {
    if (mapHIRToHWValue.find(hirValue) == mapHIRToHWValue.end()) {
      (hirValue.getDefiningOp() ? hirValue.getDefiningOp()
                                : hirValue.getParentRegion()->getParentOp())
          ->emitError(
              "Could not find the corresponding hw Value in HIRToHWMapping.");
      assert(false);
    }
    return mapHIRToHWValue.lookup(hirValue);
  }
  void map(Value hirValue, Value hwValue) {
    assert(hirValue.getParentRegion()->getParentOfType<hir::FuncOp>());
    assert(hwValue.getParentRegion()->getParentOfType<hw::HWModuleOp>());
    mapHIRToHWValue[hirValue] = hwValue;
  }

  IRMapping getBlockAndValueMapping() {
    IRMapping blockAndValueMap;
    for (auto keyValue : mapHIRToHWValue) {
      blockAndValueMap.map(keyValue.getFirst(), keyValue.getSecond());
    }
    return blockAndValueMap;
  }

  // This function ensures that all the 'hirValue -> from' get updated
  // to 'hirValue -> to'.
  // There can be multiple such values because of assign ops in hir dialect.
  void replaceAllHWUses(Value from, Value to) {
    assert(from.getParentRegion()->getParentOfType<hw::HWModuleOp>());
    assert(to.getParentRegion()->getParentOfType<hw::HWModuleOp>());
    assert(from.getType() == to.getType());
    from.replaceAllUsesWith(to);
    for (auto keyValue : mapHIRToHWValue) {
      if (keyValue.getSecond() == from)
        mapHIRToHWValue[keyValue.getFirst()] = to;
    }
  }
};
