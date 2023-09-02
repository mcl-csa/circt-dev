#ifndef HIR_ATTRS
#define HIR_ATTRS

include "mlir/IR/EnumAttr.td"
//Attrs
def MemKindEnum : I32EnumAttr<"MemKindEnum", "Enum to represent kind of hardware memory", [
                            I32EnumAttrCase<"reg", 0>,
                            I32EnumAttrCase<"bram", 1>,
                            I32EnumAttrCase<"lutram", 2>]> {
   let genSpecializedAttr = 0;
 }

def MemKindEnumAttr : EnumAttr<HIR_Dialect, MemKindEnum, "memKindEnum">;

#endif
