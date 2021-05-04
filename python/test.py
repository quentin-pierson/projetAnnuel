from ctypes import *

path_to_dl = "G:/Programmes/Python/projetAnnuel/CppIA/cmake-build-debug/CppIA.dll"

if __name__ == '__main__':
    mylib = cdll.LoadLibrary(path_to_dl)

    mylib.toto.argype = []
    mylib.toto.restype = c_int

    rslt = mylib.toto()

    print(rslt)

    my_list = [1,2,3,4,5]
    arr_size = len(my_list)

    arr_type = c_int * arr_size
    arr = arr_type(*my_list)

    mylib.array_sum.argype = [arr, c_int]
    mylib.array_sum.restype = c_int

    rslt = mylib.array_sum(arr, arr_size)

    print(rslt)



