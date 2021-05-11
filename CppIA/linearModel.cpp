#include "header.h"

/*
def create_linear_model(input_dim: int) -> [float]:
    return [random.uniform(-1.0, 1.0) for i in range(input_dim + 1)]
*/

DLLEXPORT float create_linear_model(int input_dim){

}

/*
def predict_linear_model_regression(model: [float], sample_inputs: [float]):
  result = model[0] * 1.0    # bias
  for i in range(1, len(model)):
    result += model[i] * sample_inputs[i - 1]
  return result
*/

DLLEXPORT void predict_linear_model_regression(float model, float sample_inputs){

}

/*
def predict_linear_model_classification(model: [float], sample_inputs: [float]):
  if predict_linear_model_regression(model, sample_inputs) >= 0:
    return 1.0
  else:
    return -1.0
*/

DLLEXPORT void predict_linear_model_classification(float model, float sample_inputs){

}

/*
def train_classification_rosenblatt_rule_linear_model(model: [float],
                                                      flattened_dataset_inputs:[float],
                                                      flattened_dataset_expected_outputs: [float],
                                                      alpha: float = 0.001,
                                                      iterations_count: int = 50):
    input_dim = len(model) - 1
    samples_count = len(flattened_dataset_inputs) // input_dim

    for it in range(iterations_count):
        k = random.randint(0, samples_count - 1)
        Xk = flattened_dataset_inputs[k * input_dim:(k + 1) * input_dim]
        Yk = flattened_dataset_expected_outputs[k]
        gXk = predict_linear_model_classification(model, Xk)
        model[0] += alpha * (Yk - gXk) * 1.0    # bias correction

    for i in range(1, len(model)):
        model[i] += alpha * (Yk - gXk) * Xk[i - 1]
*/


DLLEXPORT void train_classification_rosenblatt_rule_linear_model(float model){

}

/*
def destroy_linear_model(model: [float]):
    del model
 */

DLLEXPORT void destroy_linear_model(float model){

}