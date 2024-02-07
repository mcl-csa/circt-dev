import os
import subprocess


def findVerilator():
    if "@VERILATOR_PATH@" != "":
        return "@VERILATOR_PATH@"
    elif "VERILATOR_PATH" in os.environ:
        return os.environ["VERILATOR_PATH"]


class Compiler:
    def __init__(self, verilator, cxx):
        self.verilator = verilator
        self.cxx = cxx

    def verilate(self, top, svDir, includeDir):
        cmd_verilate = [self.verilator, '--cc', f"{svDir}/*.sv", '--top-module',
                        top, '--prefix', 'Vtop', '-Mdir', '.']

        if (includeDir != ''):
            pli_header = f'{includeDir}/pli.h'
            if (os.path.isfile()):
                cmd_verilate += ['--FI', pli_header]

        subprocess.run(cmd_verilate, shell=False)

    def compile(self, includeDir):
        cmd_compile = [self.cxx, '--shared', '-fPIC',  '*.cpp', ]
        result = subprocess.run([self.verilator, '--getenv', 'VERILATOR_ROOT'],
                                capture_output=True,  # Python >= 3.7 only
                                text=True  # Python >= 3.7 only
                                )
        verilator_root = result.stdout
        print(verilator_root)
        # subprocess.run(cmd_compile, shell=False)
