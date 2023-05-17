#include "module.h"
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

template <typename AddrT, typename DataT>
struct RdPort {
  unsigned char *addrEn;
  AddrT *addr;
  unsigned char *rdEn;
  DataT *rdData;
  DataT value;
  bool hasValue;
};

template <typename AddrT, typename DataT>
struct WrPort {
  unsigned char *addrEn;
  AddrT *addr;
  unsigned char *wrEn;
  DataT *wrData;
};

template <typename AddrT, typename DataT>
class MemRef : public Module {
private:
  vector<DataT> data;
  vector<RdPort<AddrT, DataT>> rdPorts;
  vector<WrPort<AddrT, DataT>> wrPorts;

public:
  MemRef(const char *fileName);
  MemRef(int size);
  void registerRdPort(unsigned char *addrEn, AddrT *addr, unsigned char *rdEn,
                      DataT *rdData);
  void registerWrPort(unsigned char *addrEn, AddrT *addr, unsigned char *wrEn,
                      DataT *wrData);
  void registerRdWrPort(unsigned char *addrEn, AddrT *addr, unsigned char *rdEn,
                        DataT *rdData, unsigned char *wrEn, DataT *wrData);
  // void tick();
  void before_posedge() override;
  void after_posedge() override;
  void *getRawDataPtr();
};

template <typename AddrT, typename DataT>
MemRef<AddrT, DataT>::MemRef(const char *fileName) {
  ifstream myfile(fileName);
  if (!myfile.is_open()) {
    cout << "Could not open file \"" << fileName << "\" in directory "
         << get_current_dir_name() << endl;
    exit(1);
  }

  string line;
  while (getline(myfile, line)) {
    // TODO: assert(sizeof(DataT)<=sizeof(long));
    data.push_back((DataT)stol(line));
  }
}
template <typename AddrT, typename DataT>
MemRef<AddrT, DataT>::MemRef(int size) {
  data = vector<DataT>(size, 0);
}

template <typename AddrT, typename DataT>
void MemRef<AddrT, DataT>::registerRdPort(unsigned char *addrEn, AddrT *addr,
                                          unsigned char *rdEn, DataT *rdData) {

  rdPorts.push_back({addrEn, addr, rdEn, rdData});
}

template <typename AddrT, typename DataT>
void MemRef<AddrT, DataT>::registerWrPort(unsigned char *addrEn, AddrT *addr,
                                          unsigned char *wrEn, DataT *wrData) {
  wrPorts.push_back({addrEn, addr, wrEn, wrData});
}

template <typename AddrT, typename DataT>
void MemRef<AddrT, DataT>::registerRdWrPort(unsigned char *addrEn, AddrT *addr,
                                            unsigned char *rdEn, DataT *rdData,
                                            unsigned char *wrEn,
                                            DataT *wrData) {
  rdPorts.push_back({addrEn, addr, rdEn, rdData});
  wrPorts.push_back({addrEn, addr, wrEn, wrData});
}

template <typename AddrT, typename DataT>
void MemRef<AddrT, DataT>::before_posedge() {
  for (auto &rdPort : rdPorts) {
    if (*rdPort.rdEn == 1) {
      auto value = this->data[*rdPort.addr];
      rdPort.value = value;
      rdPort.hasValue = true;
    } else
      rdPort.hasValue = false;
  }
  for (auto &wrPort : wrPorts) {
    if (*wrPort.wrEn == 1) {
      this->data[*wrPort.addr] = *wrPort.wrData;
    }
  }
}

template <typename AddrT, typename DataT>
void MemRef<AddrT, DataT>::after_posedge() {
  for (auto rdPort : rdPorts) {
    if (rdPort.hasValue) {
      *rdPort.rdData = rdPort.value;
    }
  }
}

/*
template <typename AddrT, typename DataT>
void MemRef<AddrT,DataT>::tick()
{
    //read before write
    for(auto rdPort: rdPorts)
    {
        if(*rdPort.rdEn!=1) continue;
        auto value= this->data[*rdPort.addr];
        *rdPort.rdData=value;
    }
    for(auto wrPort: wrPorts)
    {
        if(*wrPort.wrEn!=1) continue;
        this->data[*wrPort.addr] = *wrPort.wrData;
    }

}
*/

template <typename AddrT, typename DataT>
void *MemRef<AddrT, DataT>::getRawDataPtr() {
  return (void *)data.data();
}
