#include "spi-verilator.h"

SPIVerilator::SPIVerilator(vector<string> &verilogFiles) : SPI() {

  // Verilate the model into verilator_build/verilated_dut.so
  this->dutLib = new SharedLib("verilator_build/dut.so");

  this->evalptr =
      reinterpret_cast<decltype(evalptr)>(dutLib->getSymbolPtr("eval"));
  this->traceptr =
      reinterpret_cast<decltype(traceptr)>(dutLib->getSymbolPtr("trace"));
}

SPIVerilator::~SPIVerilator() { delete this->dutLib; }
int *SPIVerilator::operator[](string &key) {
  return reinterpret_cast<int *>(dutLib->getSymbolPtr(key.c_str()));
}

void SPIVerilator::eval() { this->evalptr(); }
void SPIVerilator::trace(int time) { this->traceptr(time); }