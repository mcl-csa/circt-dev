#ifndef HIRPASSIMPL_H
#define HIRPASSIMPL_H
template <typename T>
class HIRPassImplBase {
public:
  HIRPassImplBase(T op) : op(op) {}
  T &getOperation() { return op; }

private:
  T op;
};
#endif