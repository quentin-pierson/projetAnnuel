#ifndef CPPIA_FONCTIONNALITE_H
#define CPPIA_FONCTIONNALITE_H

#endif //CPPIA_FONCTIONNALITE_H

#include "header.h"
#include "structCollection.cpp"
#include "mlp.h"

float* cut_float_array(float* array, int start, int end);
float* copy_array(float* array, int size);
float* insert_array(float* array, int size,int pos, float value);
Model* CreateModel(float* array, int size);
Model2* model2_set_zero(int x, int y);
void printArray(float* array, int size);
void printArray(Model* model);
void printW(Model3* W);
void printX(Model2* X);
void freeX(Model2* W);
void freeW(Model3* W);
int minimumArray(int* array , int sizeArray);
int maximumArray(int* array,int sizeArray);
float maximumArray(float *array, int sizeArray);
float summArray(float** array);
float* mean(float* array);
Model2* repeatArray(Model2* array, int size, int k);

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
Model2* CreateModel2(int size);
float* TakeLast(Model2* model);
Model2* CreateModel2(int size);
float* TakeLast(Model2* model);

// SaveNLoad
void save_in_json(string name, string data);
DLLEXPORT void save_mlp_regression(MLP* mlp);
DLLEXPORT void save_mlp_classification(MLP* mlp);
void save_mlp_model(MLP *mlp, string type);
string save_model(Model* model);
string save_model2(Model2* model);
string save_model3(Model3* model);
string simple_array_to_string(string data, int x, int* y);
string simple_array_to_string(string data, int x, float* y);

string open_json(char *filename);
Model* load_model(string data);
Model2* load_model2(string data);
Model3* load_model3(string data);