#include <iostream>
#include <stdlib.h>

#include "MemRef.h"
#include "Testbench.h"
#include "Vgesummv_hir.h"

void gesummv_hir(int alpha, int beta, int tmp[8], int A[8][8], int B[8][8],
                 int X[8], int Y[8]);

class gesummv_tb : public Testbench<Vgesummv_hir> {
public:
  MemRef<unsigned char, unsigned int> A;
  MemRef<unsigned char, unsigned int> B;
  MemRef<unsigned char, unsigned int> X;
  MemRef<unsigned char, unsigned int> Y;
  MemRef<unsigned char, unsigned int> TMP;

  gesummv_tb(std::string &&Adata, std::string &&Bdata, std::string &&Xdata)
      : Testbench("waveform.vcd"), A(Adata.c_str()), B(Bdata.c_str()),
        X(Xdata.c_str()), Y(8), TMP(8) {
    A.registerRdPort(&dut.A_p0_addr_en, &dut.A_p0_addr_data, &dut.A_p0_rd_en,
                     &dut.A_p0_rd_data);
    B.registerRdPort(&dut.B_p0_addr_en, &dut.B_p0_addr_data, &dut.B_p0_rd_en,
                     &dut.B_p0_rd_data);
    X.registerRdPort(&dut.X_p0_addr_en, &dut.X_p0_addr_data, &dut.X_p0_rd_en,
                     &dut.X_p0_rd_data);
    Y.registerWrPort(&dut.Y_p0_addr_en, &dut.Y_p0_addr_data, &dut.Y_p0_wr_en,
                     &dut.Y_p0_wr_data);
    TMP.registerWrPort(&dut.tmp_p0_addr_en, &dut.tmp_p0_addr_data,
                       &dut.tmp_p0_wr_en, &dut.tmp_p0_wr_data);
    dut.alpha = 1;
    dut.beta = 2;
    this->registerModule(&A);
    this->registerModule(&B);
    this->registerModule(&X);
    this->registerModule(&Y);
    this->registerModule(&TMP);
  }
};

int main(int argc, char **argv, char **env) {
  gesummv_tb tb(std::string(argv[1]) + std::string("/A.txt"),
                std::string(argv[1]) + std::string("/B.txt"),
                std::string(argv[1]) + std::string("/X.txt"));
  tb.run(1000);
  int tmp[8];
  int Y[8];
  gesummv_hir(1, 2, tmp, (int(*)[8])tb.A.getRawDataPtr(),
              (int(*)[8])tb.B.getRawDataPtr(), (int *)tb.X.getRawDataPtr(), Y);
  int *tmp_dut = (int *)tb.TMP.getRawDataPtr();
  int *Y_dut = (int *)tb.Y.getRawDataPtr();
  for (int i = 0; i < 8; i++) {
    if (tmp_dut[i] != tmp[i]) {
      printf("ERROR: wrong value. tmp_dut[%d]:%d, correct value: %d.\n", i,
             tmp_dut[i], tmp[i]);
      exit(1);
    }
    if (Y_dut[i] != Y[i]) {
      printf("ERROR: wrong value. Y_dut[%d]:%d, correct value: %d.\n", i,
             Y_dut[i], Y[i]);
      exit(1);
    }
  }
  printf("TEST PASS\n");
}
