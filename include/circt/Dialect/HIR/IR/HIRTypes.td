def HIR_TimeType : DialectType<HIR_Dialect,CPred<"$_self.isa<TimeType>()">, "HIR time type">,
    BuildableType<"TimeType::get($_builder.getContext())">;

def HIR_MemrefType: DialectType<HIR_Dialect,CPred<"$_self.isa<MemrefType>()">, "HIR memref type ">,
    BuildableType<"MemrefType::get($_builder.getContext())">;

def HIR_FuncType: DialectType<HIR_Dialect,CPred<"$_self.isa<FuncType>()">, "HIR function type ">,
    BuildableType<"FuncType::get($_builder.getContext())">;

def HIR_BusType: DialectType<HIR_Dialect, CPred<"$_self.isa<BusType>()">, "HIR bus type ">,
    BuildableType<"BusType::get($_builder.getContext())">;

def HIR_BusTensorType: DialectType<HIR_Dialect,
CPred<"$_self.isa<BusTensorType>()">, "HIR bus_tensor type ">,
    BuildableType<"BusTensorType::get($_builder.getContext())">;
