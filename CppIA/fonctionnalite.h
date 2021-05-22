#ifndef CPPIA_FONCTIONNALITE_H
#define CPPIA_FONCTIONNALITE_H

#endif //CPPIA_FONCTIONNALITE_H

#include "header.h"
#include "structCollection.cpp"

float* cut_float_array(float* array, int start, int end);
float* copy_array(float* array, int size);
float* insert_array(float* array, int size,int pos, float value);

// linearModel
DLLEXPORT void destroy_linear_model(Model* model);
DLLEXPORT Model* create_linear_model(int input_dim);
DLLEXPORT float predict_linear_model_regression_unefficient_but_more_readable(Model* model, float* sample_inputs,
                                                                    int sample_inputs_size);

DLLEXPORT float predict_linear_model_regression(Model* model, float* sample_inputs);

DLLEXPORT float predict_linear_model_classification(Model* model, float* sample_inputs);

DLLEXPORT void train_classification_rosenblatt_rule_linear_model(Model* model,
                                                       float* flattened_dataset_inputs,
                                                       int flattened_dataset_inputs_size,
                                                       float* flattened_dataset_expected_outputs,
                                                       float alpha,
                                                       int iterations_count);

DLLEXPORT void train_regression_pseudo_inverse_linear_model(Model* model,
                                                             int flattened_dataset_inputs_size,
                                                  float* flattened_dataset_inputs,
                                                             int flattened_dataset_outputs_size,
                                                  float* flattened_dataset_expected_output);

// MLP
void printW(Model3* W,int * npl, int nplSize);
void freeW(Model3* W,int * npl, int nplSize);
Model2* CreateModel2(int size);
float* TakeLast(Model2* model);