from ctypes import *
import path_to_dll as path
import numpy as np


class MLPModel():
    def __init__(self):
        self.mylib = cdll.LoadLibrary(path.path_to_dll)

        self.c_int_p = np.ctypeslib.ndpointer(dtype=np.int64,
                                              flags='C_CONTIGUOUS')

        self.c_float_p = np.ctypeslib.ndpointer(dtype=np.float64,
                                                flags='C_CONTIGUOUS')


        # self.c_float_pp = POINTER(POINTER(c_float))
        # self.c_float_ppp = POINTER(POINTER(POINTER(c_float)))

        self.c_float_pp = POINTER(self.c_float_p)
        self.c_float_ppp = POINTER(self.c_float_pp)


        class Model(Structure):
            _fields_ = [
                ("values", self.c_float_p),
                ("size", c_int)
            ]

        class Model2(Structure):
            _fields_ = [
                ("values", self.c_float_pp),
                ("x", c_int),
                ("y", self.c_int_p)
            ]

        class Model3(Structure):
            _fields_ = [
                ("values", self.c_float_ppp),
                ("y", self.c_int_p),
                ("x", c_int)
            ]

        class ModelMlp(Structure):
            _fields_ = [
                ("W", Model3),
                ("d", Model),
                ("X", Model2),
                ("deltas", Model2),

            ]

        # ----------------------------------------------------------------------------------
        #                       mylib = create MLP model
        # ----------------------------------------------------------------------------------
        self.mylib.create_mlp_model.argtype = [self.c_int_p, c_int]
        self.mylib.create_mlp_model.restype = POINTER(ModelMlp)

        # ----------------------------------------------------------------------------------
        #        mylib = predict linear_model regression unefficient but_more readable
        # ----------------------------------------------------------------------------------
        self.mylib.predict_mlp_model_regression.argtype = [POINTER(ModelMlp), self.c_float_p, c_int]
        self.mylib.predict_mlp_model_regression.restype = POINTER(c_float)

    def create_mlp_model(self, npl):
        npl_size = len(npl)
        npl_cast = cast((c_int * npl_size)(*npl), self.c_int_p)

        result = self.mylib.create_mlp_model(npl_cast, npl_size)
        return result

    def predict_mlp_model_regression(self, model, sample_inputs):
        sample_inputs_size = len(sample_inputs)
        sample_inputs_cast = cast((c_float * sample_inputs_size)(*sample_inputs), POINTER(c_float))

        return self.mylib.predict_mlp_model_regression(model, sample_inputs_cast, sample_inputs_size)
