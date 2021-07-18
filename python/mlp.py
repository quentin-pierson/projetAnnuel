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
                ("values", POINTER(c_float)),
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
                ("W", POINTER(Model3)),
                ("d", POINTER(Model)),
                ("X", POINTER(Model2)),
                ("deltas", POINTER(Model2))
            ]

        # ----------------------------------------------------------------------------------
        #                       mylib = create MLP model
        # ----------------------------------------------------------------------------------
        self.mylib.create_mlp_model.argtype = [self.c_int_p, c_int]
        self.mylib.create_mlp_model.restype = POINTER(ModelMlp)

        # ----------------------------------------------------------------------------------
        #                   mylib = predict_mlp_model_regression
        # ----------------------------------------------------------------------------------
        self.mylib.predict_mlp_model_regression.argtype = [POINTER(ModelMlp), self.c_float_p, c_int]
        self.mylib.predict_mlp_model_regression.restype = POINTER(c_float)

        # ----------------------------------------------------------------------------------
        #                   mylib = predict_mlp_model_classification
        # ----------------------------------------------------------------------------------
        self.mylib.predict_mlp_model_classification.argtype = [POINTER(ModelMlp), self.c_float_p, c_int]
        self.mylib.predict_mlp_model_classification.restype = POINTER(c_float)

        # ----------------------------------------------------------------------------------
        #                   mylib = train_classification_stochastic_gradient_backpropagation_mlp_model
        # ----------------------------------------------------------------------------------
        self.mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.argtype = [POINTER(ModelMlp),
                                                                                                 self.c_float_p, c_int,
                                                                                                 self.c_float_p,
                                                                                                 c_int, c_float, c_int]
        self.mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #                   mylib = train_regression_stochastic_gradient_backpropagation_mlp_model
        # ----------------------------------------------------------------------------------
        self.mylib.train_regression_stochastic_gradient_backpropagation_mlp_model.argtype = [POINTER(ModelMlp),
                                                                                             self.c_float_p, c_int,
                                                                                             self.c_float_p,
                                                                                             c_int, c_float, c_int]
        self.mylib.train_regression_stochastic_gradient_backpropagation_mlp_model.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #                   mylib = free_mlp
        # ----------------------------------------------------------------------------------
        self.mylib.free_MLP.argtype = [POINTER(ModelMlp)]
        self.mylib.free_MLP.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #               mylib = save mlp model for regression
        # ----------------------------------------------------------------------------------

        self.mylib.save_mlp_regression.argtype = [POINTER(ModelMlp)]
        self.mylib.save_mlp_regression.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #               mylib = save mlp model for classification
        # ----------------------------------------------------------------------------------

        self.mylib.save_mlp_classification.argtype = [POINTER(ModelMlp)]
        self.mylib.save_mlp_classification.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #               mylib = load mlp model
        # ----------------------------------------------------------------------------------
        self.mylib.load_mlp_model.argtype = [c_char_p]
        self.mylib.load_mlp_model.restype = POINTER(ModelMlp)

    def create_mlp_model(self, npl):
        npl_size = len(npl)
        npl_cast = cast((c_int * npl_size)(*npl), self.c_int_p)

        result = self.mylib.create_mlp_model(npl_cast, npl_size)
        return result

    def predict_mlp_model_regression(self, model, sample_inputs):
        sample_inputs_size = len(sample_inputs)
        sample_inputs_cast = cast((c_float * sample_inputs_size)(*sample_inputs), POINTER(c_float))

        result = []

        valuesSize = int(model.d[0].size) - 1
        maxi = int(model.d[0].values[valuesSize])

        tab = self.mylib.predict_mlp_model_regression(model, sample_inputs_cast, sample_inputs_size)

        if maxi > 1:

            for i in range(0, maxi):
                result.append(tab[i])
        else:
            result = tab[0]

        return result

    def predict_mlp_model_classification(self, model, sample_inputs):

        sample_inputs_size = len(sample_inputs)
        sample_inputs_cast = cast((c_float * sample_inputs_size)(*sample_inputs), POINTER(c_float))

        result = []

        valuesSize = int(model.d[0].size) - 1
        maxi = int(model.d[0].values[valuesSize])

        tab = self.mylib.predict_mlp_model_classification(model, sample_inputs_cast, sample_inputs_size)

        if maxi > 1:

            for i in range(0, maxi):
                result.append(tab[i])
        else:
            result = tab[0]

        return result

    def train_classification_stochastic_gradient_backpropagation(self, model, flattened_dataset_inputs,
                                                                 flattened_dataset_expected_outputs,
                                                                 alpha=0.001,
                                                                 iterations_count=10000):
        flattened_dataset_inputs_size = len(flattened_dataset_inputs)

        flattened_dataset_inputs_cast = cast((c_float * flattened_dataset_inputs_size)(*flattened_dataset_inputs),
                                             self.c_float_p)

        flattened_dataset_expected_outputs_size = len(flattened_dataset_expected_outputs)

        flattened_dataset_expected_outputs_cast = cast(
            (c_float * flattened_dataset_expected_outputs_size)(*flattened_dataset_expected_outputs),
            self.c_float_p)

        self.mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(model,
                                                                                      flattened_dataset_inputs_cast,
                                                                                      flattened_dataset_inputs_size,
                                                                                      flattened_dataset_expected_outputs_cast,
                                                                                      flattened_dataset_expected_outputs_size,
                                                                                      c_float(alpha), iterations_count)

    def train_regression_stochastic_gradient_backpropagation(self,
                                                             model,
                                                             flattened_dataset_inputs,
                                                             flattened_dataset_expected_outputs,
                                                             alpha=0.001,
                                                             iterations_count=100000):
        flattened_dataset_inputs_size = len(flattened_dataset_inputs)

        flattened_dataset_inputs_cast = cast((c_float * flattened_dataset_inputs_size)(*flattened_dataset_inputs),
                                             self.c_float_p)

        flattened_dataset_expected_outputs_size = len(flattened_dataset_expected_outputs)

        flattened_dataset_expected_outputs_cast = cast(
            (c_float * flattened_dataset_expected_outputs_size)(*flattened_dataset_expected_outputs),
            self.c_float_p)

        self.mylib.train_regression_stochastic_gradient_backpropagation_mlp_model(model, flattened_dataset_inputs_cast,
                                                                                  flattened_dataset_inputs_size,
                                                                                  flattened_dataset_expected_outputs_cast,
                                                                                  flattened_dataset_expected_outputs_size,
                                                                                  c_float(alpha),
                                                                                  iterations_count)

    def free_MLP(self, model):
        self.mylib.free_MLP(model)

    def save_mlp_regression(self, model):
        self.mylib.save_mlp_regression(model)

    def save_mlp_classification(self,model):
        self.mylib.save_mlp_classification(model)

    def load_mlp_model(self, filename):
        filename_b = filename.encode('utf-8')

        return self.mylib.load_mlp_model(filename_b)

