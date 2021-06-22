#include "fonctionnalite.h"
#include "header.h"

class MLP{
    /* @dataclass
        class MLP():
        W: [[[float]]]
        d: [int]
        X: [[float]]
        deltas: [[float]]
        */
public:
    Model3* W;
    Model* d;
    Model2* X;
    Model2* deltas;
public:
    MLP(Model3* W, Model* d, Model2* X, Model2* deltas){
        this->W = W;
        this->d = d;
        this->X = X;
        this->deltas = deltas;
    }

    void forward_pass(float* sample_inputs, bool is_classification){
        int L = this->d->size -1;
        for(int j = 1; j < this->d->values[0]+1; j++){
            this->X->values[0][j] = sample_inputs[j-1];
        }

        for(int l = 1; l < L+1; l++){
            for(int j = 1; j < this->d->values[l] +1; j++){
                float sum_result = 0.0;
                for(int i = 0; i < this->d->values[l-1] +1; i++){
                    sum_result += this->W->values[l][i][j] * this->X->values[l-1][i];
                }
                this->X->values[l][j] = sum_result;
                if(is_classification == true || l < L){
                    this->X->values[l][j] = tanhf(this->X->values[l][j]);
                }
            }
        }
    }

    void train_stochastic_gradient_backpropagation(float* flattened_dataset_inputs,
                                                   int flattened_dataset_inputs_size,
                                                   float* flattened_dataset_expected_outputs,
                                                   int flattened_dataset_expected_outputs_size,
                                                   bool is_classification,
                                                   float alpha,
                                                   int iterations_count){
        int L = this->d->size -1;
        float input_dim = this->d->values[0];
        float output_dim = this->d->values[L];
        int samples_count = flattened_dataset_inputs_size / input_dim;

        for(int it=0; it<iterations_count; it++){
            int k = rand() % samples_count;
            float* sample_input = cut_float_array(flattened_dataset_inputs,k * input_dim,(k+1) * input_dim);
            float* sample_expected_output = cut_float_array(flattened_dataset_expected_outputs, k * output_dim, (k+1) * output_dim);
            this->forward_pass(sample_input,is_classification);

            for(int j=1; j < this->d->values[L] +1; j++){
                this->deltas->values[L][j] = (this->X->values[L][j] - sample_expected_output[j-1]);
                if(is_classification){
                    this->deltas->values[L][j] *= (1 - this->X->values[L][j] * this->X->values[L][j]);
                }
            }

            for(int l = L; l >= 1; l--){
                for(int i=1; i < this->d->values[l-1] + 1; i++) {
                    float sum_result = 0.0;
                    for (int j = 1; j < this->d->values[l] +1; j++) {
                        sum_result += this->W->values[l][i][j] * this->deltas->values[l][j];
                    }
                    this->deltas->values[l -1][i] = (1 - this->X->values[l-1][i] * this->X->values[l-1][i]) * sum_result;
                }
            }
            for(int l = 1; l < L +1; l++){
                for(int i = 0; i < this->d->values[l-1]+1; i++){
                    for (int j = 1; j < this->d->values[l]+1; j++) {
                        float result = alpha * this->X->values[l-1][i] * this->deltas->values[l][j];
                        this->W->values[l][i][j] -= result;
                    }
                }
            }
        }
    }

};

DLLEXPORT MLP* create_mlp_model(int * npl, int nplSize) {
    // W
    Model3 *W = (Model3 *) (malloc(sizeof(Model3)));
    W->values = (float ***) (malloc(sizeof(float ***) * nplSize));

    W->x = nplSize;
    W->y = (int *) (malloc(sizeof(int) * nplSize));

    // X
    Model2 *X = CreateModel2(nplSize);

    // Delta
    Model2 *deltas = CreateModel2(nplSize);

    // d
    Model* d = (Model*) (malloc( sizeof(Model)));
    d->values = (float *) (malloc(sizeof(float *) * nplSize));
    d->size = nplSize;

    for (int l = 0; l < nplSize; l++) {
        //d et les y
        int val = npl[l];
        W->y[l] = val;
        d->values[l] = val;

        //X et Deltas
        int iMax2 = npl[l] + 1;

        X->y[l] = iMax2;
        deltas->y[l] = iMax2;

        X->values[l] = (float *) (malloc(sizeof(float *) * iMax2));
        deltas->values[l] = (float *) (malloc(sizeof(float *) * iMax2));

        for (int i = 0; i < iMax2; i++) {

            if (i == 0) {
                X->values[l][i] = 1;
                //append 1
            } else {
                X->values[l][i] = 0;
                //append 0
            }

            deltas->values[l][i] = 0.0;
        }

        //W
        int imax = npl[l - 1] + 1;

        if (l == 0) {
            W->values[l] = (float **) (malloc(sizeof(float **) * 1));
            continue;
        } else {
            W->values[l] = (float **) (malloc(sizeof(float **) * imax));
        }

        for (int i = 0; i < imax; i++) {
            int jmax = npl[l] + 1;
            W->values[l][i] = (float *) (malloc(sizeof(float *) * jmax));
            for (int j = 0; j < jmax; j++) {
                std::random_device rd;
                std::default_random_engine generator(rd());
                std::uniform_real_distribution<float> distribution(-1.0, 1.0);
                W->values[l][i][j] = distribution(generator);
            }
        }
    }

    MLP* mlp = new MLP(W, d, X,deltas);
    return mlp;
}

DLLEXPORT float* predict_mlp_model_regression(MLP* model, float* sample_inputs, int size){
    model-> forward_pass(sample_inputs, false);
    return TakeLast(model->X);
}

DLLEXPORT float* predict_mlp_model_classification(MLP* model, float* sample_inputs, int size){
    model-> forward_pass(sample_inputs, true);
    return TakeLast(model->X);
}


DLLEXPORT void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP* model,
                                                                                  float* flattened_dataset_inputs,
                                                                                  int flattened_dataset_inputs_size,
                                                                                  float* flattened_dataset_expected_outputs,
                                                                                  int flattened_dataset_expected_outputs_size,
                                                                                  float alpha,
                                                                                  int iterations_count){
    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                     flattened_dataset_inputs_size,
                                                     flattened_dataset_expected_outputs,
                                                     flattened_dataset_expected_outputs_size,
                                                     true,
                                                     alpha,
                                                     iterations_count);
}


DLLEXPORT void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP* model,
                                                                                  float* flattened_dataset_inputs,
                                                                                  int flattened_dataset_inputs_size,
                                                                                  float* flattened_dataset_expected_outputs,
                                                                                  int flattened_dataset_expected_outputs_size,
                                                                                  float alpha,
                                                                                  int iterations_count){
    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                     flattened_dataset_inputs_size,
                                                     flattened_dataset_expected_outputs,
                                                     flattened_dataset_expected_outputs_size,
                                                     false,
                                                     alpha,
                                                     iterations_count);

}

DLLEXPORT void free_MLP(MLP* model){
    freeX(model->deltas);
    freeX(model->X);
    freeW(model->W);
    free(model->d->values);
    free(model->d);
    free(model);
}