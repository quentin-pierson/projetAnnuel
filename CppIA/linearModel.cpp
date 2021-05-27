#include "fonctionnalite.h"
#include "header.h"

DLLEXPORT void destroy_linear_model(Model* model) {
    free(model->values);
    free(model);
}

DLLEXPORT Model* create_linear_model(int input_dim){
    Model* model = (Model*) (malloc(sizeof(Model)));

    model -> size = input_dim+1;
    model -> values = (float*) (malloc(sizeof(float) * model->size));

    for(int i=0 ; i<model->size ; i++){
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<float> distribution(-1.0, 1.0);
        model->values[i] = distribution(generator);
    }

    return model;
}
/*
 *
 * def predict_linear_model_regression_unefficient_but_more_readable(model: [float], sample_inputs: [float]):
  sample_inputs_copy = list(sample_inputs)
  sample_inputs_copy.insert(0, 1.0)

  result = 0.0
  for i in range(len(model)):
    result += model[i] * sample_inputs_copy[i]
  return result
 */

DLLEXPORT float predict_linear_model_regression_unefficient_but_more_readable(Model* model, float* sample_inputs,
                                                                              int sample_inputs_size){

    float* sample_inputs_copy = copy_array(sample_inputs, sample_inputs_size);
    sample_inputs_copy = insert_array(sample_inputs_copy, sample_inputs_size,0, 1.0);

    float result = 0.0;
    for(int i=0; i<model->size ; i++){
        result+= model->values[i] * sample_inputs[i - 1];
    }
    free(sample_inputs_copy);
    return result;
}


DLLEXPORT float predict_linear_model_regression(Model* model, float* sample_inputs){
    float result = (float)model->values[0] * 1.0;

    for (int i = 1; i < model->size; i++){
        result+= model->values[i] * sample_inputs[i - 1];
    }

    return result;
}

DLLEXPORT float predict_linear_model_classification(Model* model, float* sample_inputs){
    float result = predict_linear_model_regression(model, sample_inputs);
    if (result >= 0) {
        return 1.0;
    }
    else {
        return -1.0;
    }
}

DLLEXPORT void train_classification_rosenblatt_rule_linear_model(Model* model,
                                                                 float* flattened_dataset_inputs,
                                                                 int flattened_dataset_inputs_size,
                                                                 float* flattened_dataset_expected_outputs,
                                                                 float alpha=0.0001,
                                                                 int iterations_count=10000){
    int input_dim = model->size -1;
    int samples_count = flattened_dataset_inputs_size / input_dim;

    for (int i = 0; i < iterations_count; i ++){

        int k = rand() % samples_count; //outPut = rand()%((userEnd - userBeg) + 1) + userBeg;

        float* Xk = cut_float_array(flattened_dataset_inputs,k * input_dim, (k + 1) * input_dim);
        float Yk = flattened_dataset_expected_outputs[k];
        float gXk = predict_linear_model_classification(model, Xk);
        model->values[0] += alpha * (Yk - gXk) * 1.0 ;

        for (int j = 1; j < model->size; j ++){
            model->values[j] += alpha * (Yk - gXk) * Xk[j - 1];
        }
        free(Xk);
    }
}


DLLEXPORT void train_regression_pseudo_inverse_linear_model(Model* model, int flattened_dataset_inputs_size, float* flattened_dataset_inputs ,
                                                             int flattened_dataset_expected_outputs_size, float* flattened_dataset_expected_output){
    int input_dim = model->size -2;

    int samples_count = flattened_dataset_inputs_size / input_dim;

    MatrixXf X(flattened_dataset_inputs_size,1); // define matrix
    Vector3f vX (flattened_dataset_inputs);
    X << vX;

    MatrixXf Y(flattened_dataset_expected_outputs_size,1); // define matrix
    Vector3f vY (flattened_dataset_expected_output);
    Y << vY;

    MatrixXf ones(samples_count, 1);
    ones.setOnes();

    MatrixXf X3(X.rows(), X.cols()+ones.cols()); // hstack
    X3 << ones, X ;

    MatrixXf W = X3.transpose() * X3;

    MatrixXf W_inv = W.inverse();

    MatrixXf W2 = W_inv * X3.transpose();

    MatrixXf W3 = W2 * Y;

    ArrayXf a = W3.array();

    for(int i=0; i<model->size-1; i++){
        model->values[i] = a[i];

    }

}