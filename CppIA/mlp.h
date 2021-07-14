#ifndef CPPIA_MLP_H
#define CPPIA_MLP_H
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
    MLP(Model3* W, Model* d, Model2* X, Model2* deltas);
    void forward_pass(float* sample_inputs, bool is_classification);
    void train_stochastic_gradient_backpropagation(float* flattened_dataset_inputs,
                                                   int flattened_dataset_inputs_size,
                                                   float* flattened_dataset_expected_outputs,
                                                   int flattened_dataset_expected_outputs_size,
                                                   bool is_classification,
                                                   float alpha,
                                                   int iterations_count);
};
#endif //CPPIA_MLP_H
