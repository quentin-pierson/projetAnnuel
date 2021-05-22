from ctypes import *
import path_to_dll as path
import numpy as np


class linearModel():
    def __init__(self):
        self.mylib = cdll.LoadLibrary(path.path_to_dll)

        self.c_float_p = np.ctypeslib.ndpointer(dtype=np.float64,
                                           flags='C_CONTIGUOUS')

        self.c_float_ppp = POINTER(POINTER(POINTER(c_float)))

        class Model(Structure):
            _fields_ = [
                ("values", self.c_float_p),
                ("size", c_int)
            ]

        # ----------------------------------------------------------------------------------
        #                       mylib = create linear model
        # ----------------------------------------------------------------------------------

        self.mylib.create_linear_model.argtype = [c_int]
        self.mylib.create_linear_model.restype = POINTER(Model)

        # ----------------------------------------------------------------------------------
        #                       mylib = predict linear model classification
        # ----------------------------------------------------------------------------------

        self.mylib.predict_linear_model_classification.argtype = [POINTER(Model), self.c_float_p]
        self.mylib.predict_linear_model_classification.restype = c_float

        # ----------------------------------------------------------------------------------
        #                       mylib = predict linear model classification
        # ----------------------------------------------------------------------------------

        self.mylib.predict_linear_model_regression.argtype = [POINTER(Model), self.c_float_p]
        self.mylib.predict_linear_model_regression.restype = c_float

        # ----------------------------------------------------------------------------------
        #                       mylib = train rossenblatt classification
        # ----------------------------------------------------------------------------------

        self.mylib.train_classification_rosenblatt_rule_linear_model.argtype = [POINTER(Model), self.c_float_p, c_int,
                                                                           self.c_float_p, c_float, c_int]
        self.mylib.train_classification_rosenblatt_rule_linear_model.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #                       mylib = destroy linear model
        # ----------------------------------------------------------------------------------

        self.mylib.destroy_linear_model.argtype = [POINTER(Model)]
        self.mylib.destroy_linear_model.restype = c_void_p

        # ----------------------------------------------------------------------------------
        #        mylib = predict linear_model regression unefficient but_more readable
        # ----------------------------------------------------------------------------------
        self.mylib.predict_linear_model_regression_unefficient_but_more_readable.argtype = [POINTER(Model), self.c_float_p, c_int]
        self.mylib.predict_linear_model_regression_unefficient_but_more_readable.restype = c_float

        # ----------------------------------------------------------------------------------
        #               mylib = train regression pseudo inverse linear model
        # ----------------------------------------------------------------------------------

        self.mylib.train_regression_pseudo_inverse_linear_model.argtype = [POINTER(Model), c_int, self.c_float_p, c_int,
                                                                                self.c_float_p]
        self.mylib.train_regression_pseudo_inverse_linear_model.restype = c_void_p


    def create_linear_model(self, input_dim):

        result = self.mylib.create_linear_model(input_dim)

        return result

    def train_classification_rosenblatt_rule_linear_model(self, model, flattened_dataset_inputs,
                                                          flattened_dataset_expected_outputs,alpha, iterations_count):
        flattened_dataset_inputs_size = len(flattened_dataset_inputs)

        flattened_dataset_inputs_cast = cast((c_float * len(flattened_dataset_inputs))(*flattened_dataset_inputs),
                                             self.c_float_p)
        flattened_dataset_expected_outputs_cast = cast((c_float * len(flattened_dataset_expected_outputs))(*flattened_dataset_expected_outputs),
                                             self.c_float_p)

        self.mylib.train_classification_rosenblatt_rule_linear_model(model, flattened_dataset_inputs_cast, flattened_dataset_inputs_size,
                                                                     flattened_dataset_expected_outputs_cast, c_float(alpha), iterations_count)

    def predict_linear_model_classification(self, model, sample_inputs):
        sample_inputs_cast = cast((c_float * len(sample_inputs))(*sample_inputs), POINTER(c_float))

        return self.mylib.predict_linear_model_classification(model, sample_inputs_cast)

    def destroy_linear_model(self,model):
        self.mylib.destroy_linear_model(model)

    def predict_linear_model_regression(self, model, sample_inputs):
        sample_inputs_cast = cast((c_float * len(sample_inputs))(*sample_inputs), POINTER(c_float))

        return self.mylib.predict_linear_model_regression(model, sample_inputs_cast)

    def predict_linear_model_regression_unefficient_but_more_readable(self, model, sample_inputs):
        sample_inputs_cast = cast((c_float * len(sample_inputs))(*sample_inputs), POINTER(c_float))
        sample_inputs_size = len(sample_inputs)
        return self.mylib.train_classification_rosenblatt_rule_linear_model(model, sample_inputs_cast, sample_inputs_size)

    def train_regression_pseudo_inverse_linear_model(self, model, flattened_dataset_inputs, flattened_dataset_expected_output):
        flattened_dataset_inputs_size = len(flattened_dataset_inputs)
        flattened_dataset_expected_outputs_size = len(flattened_dataset_expected_output)

        flattened_dataset_inputs_cast = cast((c_float * len(flattened_dataset_inputs))(*flattened_dataset_inputs),
                                             self.c_float_p)
        flattened_dataset_expected_outputs_cast = cast((c_float * len(flattened_dataset_expected_output))(*flattened_dataset_expected_output),
                                             self.c_float_p)

        self.mylib.train_regression_pseudo_inverse_linear_model(model, flattened_dataset_inputs_size, flattened_dataset_inputs_cast,
                                                                flattened_dataset_expected_outputs_size, flattened_dataset_expected_outputs_cast)
