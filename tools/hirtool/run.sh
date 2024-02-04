verilator --cc $1/*.sv --top-module $1 --prefix Vtop -Mdir build 
verilator --cc $1/*.sv --top-module $1 --prefix Vtop -Mdir build  --xml-only
verilator --getenv VERILATOR_ROOT
