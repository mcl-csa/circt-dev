#include "sharedlib.h"
#include <optional>
#include <vector>

struct SPI {
  SPI() = default;
  virtual ~SPI() = default;
  virtual void eval(){};
  virtual void trace(int time){};
  virtual int *operator[](string &key) { return nullptr; };
};

struct SPIVerilator : public SPI {
  SPIVerilator(vector<string> &verilogFiles);
  ~SPIVerilator() override;
  int *operator[](string &key) override;
  void eval() override;
  void trace(int) override;

private:
  SharedLib *dutLib;
  void (*evalptr)();
  void (*traceptr)(int);
};