/** Dynamically load verilated model **/
#ifndef __SHAREDLIB__
#define __SHAREDLIB__

#include <dlfcn.h>
#include <iostream>
#include <string>

using namespace std;

/// Loads dynamic library and allows loading symbols from it.
struct SharedLib {

  SharedLib(const char *libName) {
    this->libHandle = dlopen(libName, RTLD_NOW);

    // FIXME: Use LLVM assertion.
    if (!libHandle) {
      std::cerr << "Error loading the library '" << libName
                << "' : " << dlerror() << "\n";
      exit(EXIT_FAILURE);
    }

    // Print any error.
    dlerror();
    void (*init)();
    init = reinterpret_cast<decltype(init)>(getSymbolPtr("init"));
    init();
  }

  ~SharedLib() {
    void (*deinit)();
    deinit = reinterpret_cast<decltype(deinit)>(getSymbolPtr("deinit"));
    deinit();
    dlclose(this->libHandle);
  }

  void *getSymbolPtr(const char *symName) {
    void *handle = dlsym(this->libHandle, symName);

    // FIXME: Use LLVM assertion.
    if (!handle) {
      std::cerr << "Error accessing function '" << symName << "': " << dlerror()
                << "\n";
      exit(EXIT_FAILURE);
    }

    return handle;
  }

private:
  void *libHandle;
};

#endif