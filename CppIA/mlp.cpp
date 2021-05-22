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


    /*
     def forward_pass(self, sample_inputs: [float], is_classification: bool):
       L = len(self.d) - 1

       for j in range(1, self.d[0] + 1):
         self.X[0][j] = sample_inputs[j - 1]

       for l in range(1, L + 1):
         for j in range(1, self.d[l] + 1):
           sum_result = 0.0
           for i in range(0, self.d[l - 1] + 1):
             sum_result += self.W[l][i][j] * self.X[l - 1][i]
           self.X[l][j] = sum_result
           if is_classification or l < L:
             self.X[l][j] = math.tanh(self.X[l][j])
    */

    void forward_pass(float* sample_inputs, bool is_classification){
    }

    /*

 def train_stochastic_gradient_backpropagation(self,
                                               flattened_dataset_inputs: [float],
                                               flattened_dataset_expected_outputs: [float],
                                               is_classification: bool,
                                               alpha: float = 0.001,
                                               iterations_count: int = 100000):
   input_dim = self.d[0]
   output_dim = self.d[-1]
   samples_count = len(flattened_dataset_inputs) // input_dim
   L = len(self.d) - 1

   for it in range(iterations_count):
     k = random.randint(0, samples_count - 1)

     sample_input = flattened_dataset_inputs[k * input_dim:(k+1) * input_dim]
     sample_expected_output = flattened_dataset_expected_outputs[k * output_dim:(k+1) * output_dim]

     self.forward_pass(sample_input, is_classification)

     for j in range(1, self.d[L] + 1):
       self.deltas[L][j] = (self.X[L][j] - sample_expected_output[j - 1])
       if is_classification:
         self.deltas[L][j] *= (1 - self.X[L][j] * self.X[L][j])

     for l in reversed(range(1, L + 1)):
       for i in range(1, self.d[l - 1] + 1):
         sum_result = 0.0
         for j in range(1, self.d[l] + 1):
           sum_result += self.W[l][i][j] * self.deltas[l][j]

         self.deltas[l - 1][i] = (1 - self.X[l-1][i] * self.X[l-1][i]) * sum_result

     for l in range(1, L + 1):
       for i in range(0, self.d[l-1] + 1):
         for j in range(1, self.d[l] + 1):
           self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j]

*/

    void train_stochastic_gradient_backpropagation(float* flattened_dataset_inputs,
                                                   int flattened_dataset_inputs_size,
                                                   float* flattened_dataset_expected_outputs,
                                                   int flattened_dataset_expected_outputs_size,
                                                   bool is_classification,
                                                   float alpha,
                                                   int iterations_count){

    }

};

void printW(Model3* W){
    std::cout << "[ ";
    for(int l=0; l< W->x ;l++){ //
        int imax= W->y[l-1]+1 ;
        std::cout << "[ ";
        if(l==0){
            std::cout << "], ";
            continue;
        }
        for (int i = 0; i < imax; i++){
            std::cout << "[ ";
            int jmax = W->y[l] + 1;
            for (int j = 0; j < jmax; j++){
                std::cout << W->values[l][i][j] << ", ";
            }
            std::cout << "], ";
        }
        std::cout << "], "<< "\n";
    }
    std::cout << "] "<< "\n";
}

void printX(Model2* X){
    std::cout << "[ ";
    for(int l=0; l< X->x ;l++){ //
        int iMax = X->y[l];
        std::cout << "[ ";

        for (int i = 0; i < iMax; i++){
            std::cout << X->values[l][i] << ", ";
        }
        std::cout << "], "<< "\n";
    }
    std::cout << "] "<< "\n";
}

void freeW(Model3* W) {
    for(int l=0; l < W->x ;l++){ //
        int imax= W->y[l-1]+1;

        if(l!=0) {
            for (int i = 0; i < imax; i++) {

                free(W->values[l][i]);

            }
        }
        free(W->values[l]);
    }

    free(W->values);
    free(W->y);
    free(W);
}


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


    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);

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
                W->values[l][i][j] = distribution(generator);
            }
        }
    }
    cout << "W: " << "\n";
    printW(W);

    cout << "X: " << "\n";
    printX(X);

    cout << "deltas: " << "\n";
    printX(deltas);
    MLP* mlp = new MLP(W, d, X,deltas);
    return mlp;
}

Model2* CreateModel2(int size){
    Model2* model2 = (Model2*) (malloc( sizeof(Model2)));
    model2->values = (float **) (malloc(sizeof(float **) * size));

    model2->x = size;
    model2->y = (int *) (malloc(sizeof(int) * size));

    return model2;
}

// Model2*
DLLEXPORT float* predict_mlp_model_regression(MLP* model, float* sample_inputs, int size){
    //model-> forward_pass(sample_inputs, false);
    return TakeLast(model->X);
}

/*
def predict_mlp_model_regression(model: MLP, sample_inputs:[float])-> [float]:
  model.forward_pass(sample_inputs, False)
  return model.X[-1][1:]
 */

float* TakeLast(Model2* model){
    int xSize = model->x;
    float * tab = (float *) (malloc(sizeof(float *) * model->y[xSize-1] -1));

    for (int i = 1; i < model->y[xSize-1]; i++){
        tab[i-1] = model->values[xSize-1][i];
    }

    return tab;
}

/*
def predict_mlp_model_classification(model: MLP, sample_inputs:[float])-> [float]:
  model.forward_pass(sample_inputs, True)
  return model.X[-1][1:]
 */


/*
def train_classification_stochastic_gradient_backpropagation_mlp_model(model: MLP,
                                                                       flattened_dataset_inputs: [float],
                                                                       flattened_dataset_expected_outputs: [float],
                                                                       alpha: float = 0.001,
                                                                       iterations_count: int = 100000):
  model.train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                  flattened_dataset_expected_outputs,
                                                  True,
                                                  alpha,
                                                  iterations_count)
 */



/*
def train_regression_stochastic_gradient_backpropagation_mlp_model(model: MLP,
                                                                       flattened_dataset_inputs: [float],
                                                                       flattened_dataset_expected_outputs: [float],
                                                                       alpha: float = 0.001,
                                                                       iterations_count: int = 100000):
  model.train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                  flattened_dataset_expected_outputs,
                                                  False,
                                                  alpha,
                                                  iterations_count)

 */