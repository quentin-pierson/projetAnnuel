from ctypes import *

path_to_dl = "G:/Programmes/Python/projetAnnuel/CppIA/cmake-build-debug/CppIA.dll"

if __name__ == '__main__':
    mylib = cdll.LoadLibrary(path_to_dl)

    mylib.toto.argype = []
    mylib.toto.restype = c_int

    rslt = mylib.toto()

    print(rslt)