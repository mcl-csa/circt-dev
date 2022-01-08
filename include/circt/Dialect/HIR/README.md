#HIR
HIR is an mlir dialect for hardware description.

#Notes:

- Time variables can not be captured by a region.
- Two kinds of SSA vars:
  * Values such as integer, float and tuple or tensor of values.
  * Wires and memrefs that contain values.
- Values are always associated with a time.
