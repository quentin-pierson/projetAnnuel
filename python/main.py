from ctypes import *
import path_to_dll as path
import matplotlib.pyplot as plt
import numpy as np
from linearModel import linearModel
from mlp import MLPModel
import tqdm

lm = linearModel()
mlp = MLPModel()


if __name__ == '__main__':

    #----------------------------------------------------------------------------------
    #              Utilisation du Modèle linéaire pour la classification
    #----------------------------------------------------------------------------------

    # dataset_inputs = [
    #     [2, 4],
    #     [6, 5],
    #     [4, 7],
    # ]
    #
    # dataset_expected_outputs = [
    #     1,
    #     1,
    #     -1,
    # ]
    #
    #
    # resultat = lm.create_linear_model(2)
    # model = resultat[0]
    #
    # test_dataset = [[float(x1), float(x2)] for x1 in range(-10, 10) for x2 in range(-10, 10)]
    #
    #
    # colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]
    #
    # predicted_outputs = [lm.predict_linear_model_classification(model, p) for p in test_dataset]
    # predicted_outputs_colors = ['blue' if label == 1 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    #
    # lm.train_classification_rosenblatt_rule_linear_model(model, flattened_dataset_inputs,
    #                                                         dataset_expected_outputs, 0.002,10200)
    #
    #
    # predicted_outputs = [lm.predict_linear_model_classification(model, p) for p in test_dataset]
    #
    # predicted_outputs_colors = ['blue' if label == 1 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    #
    # print("Je suis dead ", lm.destroy_linear_model(resultat))


    # ----------------------------------------------------------------------------------
    #               Utilisation du Modèle linéaire pour la régression
    # ----------------------------------------------------------------------------------

    # dataset_inputs = [
    #     [-5],
    #     [4],
    #     [6],
    # ]
    #
    # dataset_expected_outputs = [
    #     5.2,
    #     7,
    #     8.3
    # ]
    #
    # resultat2 = lm.create_linear_model(2)
    # model2 = resultat2[0]
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #
    # test_dataset_inputs = [i for i in range(-10, 11)]
    # predicted_outputs = [lm.predict_linear_model_regression(model2, [p]) for p in test_dataset_inputs]
    #
    #
    # plt.plot(test_dataset_inputs, predicted_outputs)
    # plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    # plt.axis([-10, 10, -10, 10])
    # plt.show()
    #
    # lm.train_regression_pseudo_inverse_linear_model(model2, flattened_dataset_inputs, dataset_expected_outputs)
    #
    # test_dataset_inputs = [i for i in range(-10, 11)]
    # predicted_outputs = [lm.predict_linear_model_regression(model2, [p]) for p in test_dataset_inputs]
    #
    # plt.plot(test_dataset_inputs, predicted_outputs)
    # plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    # plt.axis([-10, 10, -10, 10])
    # plt.show()
    #
    # print("Je suis dead ", lm.destroy_linear_model(resultat2))

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour la classification
    # ----------------------------------------------------------------------------------

    dataset_inputs = [
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ]

    dataset_expected_outputs = [
        -1,
        -1,
        1,
        1,
    ]
    # mlp.create_mlp_model([2, 2, 1])
    resultat3 = mlp.create_mlp_model([2, 2, 1])
    model3 = resultat3[0]


    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]

    predicted_outputs = [mlp.predict_mlp_model_classification(model3, p) for p in test_dataset]

    print(" Je suis predicted_outputs: ", predicted_outputs)

    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    plt.show()

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    mlp.train_classification_stochastic_gradient_backpropagation(model3,
                                                                       flattened_dataset_inputs,
                                                                       dataset_expected_outputs,
                                                                       alpha=0.001,
                                                                       iterations_count=100000)


    predicted_outputs = [mlp.predict_mlp_model_classification(model3, p) for p in test_dataset]
    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    plt.show()

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    mlp.free_MLP(resultat3)

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour la regression
    # ----------------------------------------------------------------------------------

    dataset_inputs = [
        [-5],
        [4],
        [6],
    ]

    dataset_expected_outputs = [
        1.2,
        7,
        8.3
    ]

    resultat5 = mlp.create_mlp_model([1, 3, 1])
    model5 = resultat5[0]

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        flattened_dataset_inputs.append(p[0])

    test_dataset_inputs = [i for i in range(-10, 11)]
    predicted_outputs = [mlp.predict_mlp_model_regression(model5, [p]) for p in test_dataset_inputs]

    plt.plot(test_dataset_inputs, predicted_outputs)
    plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()

    mlp.train_regression_stochastic_gradient_backpropagation(model5,
                                                                   flattened_dataset_inputs,
                                                                   dataset_expected_outputs)

    test_dataset_inputs = [i for i in range(-10, 11)]
    predicted_outputs = [mlp.predict_mlp_model_regression(model5, [p]) for p in test_dataset_inputs]

    plt.plot(test_dataset_inputs, predicted_outputs)
    plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()

    mlp.free_MLP(resultat5)

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour de la classification à 3 classes
    # ----------------------------------------------------------------------------------

    dataset_inputs = [
        [0, 0],
        [0.5, 0.5],
        [1, 0],
    ]

    dataset_expected_outputs = [
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]

    resultat6 = mlp.create_mlp_model([2, 50, 3])
    model6 = resultat6[0]
    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in
              dataset_expected_outputs]

    print(" Je suis dataset_expected_outputs: ", dataset_expected_outputs)
    print(" Je suis colors: ", colors)


    predicted_outputs = [mlp.predict_mlp_model_classification(model6, p) for p in test_dataset]
    print(" Je suis predicted_outputs: ", predicted_outputs)
    predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
                                output in predicted_outputs]
    print(" Je suis predicted_outputs_colors: ", predicted_outputs_colors)
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    plt.show()

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        flattened_dataset_inputs.append(p[0])
        flattened_dataset_inputs.append(p[1])

    flattened_dataset_outputs = []
    for p in dataset_expected_outputs:
        flattened_dataset_outputs.append(p[0])
        flattened_dataset_outputs.append(p[1])
        flattened_dataset_outputs.append(p[2])

    mlp.train_classification_stochastic_gradient_backpropagation(model6,
                                                                       flattened_dataset_inputs,
                                                                       flattened_dataset_outputs)

    predicted_outputs = [mlp.predict_mlp_model_classification(model6, p) for p in test_dataset]
    predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
                                output in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    plt.show()

    mlp.free_MLP(resultat6)

    # dataset_flattened_inputs = [
    #     0, 0,
    #     0.5, 0.5,
    #     0, 1
    # ]
    #
    # dataset_flattened_outputs = [
    #     1, -1, -1,
    #     -1, 1, -1,
    #     -1, -1, 1,
    # ]
    #
    # resultat7 = mlp.create_mlp_model([2, 3, 3])
    # model7 = resultat7[0]
    #
    # points = [[i / 10.0, j / 10.0] for i in range(15) for j in range(15)]
    #
    # predicted_values = [mlp.predict_mlp_model_classification(model7, p) for p in points]
    #
    # print("predicted_values: ", predicted_values)
    #
    # classes = [np.argmax(v) for v in predicted_values]
    #
    # colors = ['blue' if c == 0 else ('red' if c == 1 else 'green') for c in classes]
    #
    # plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)
    # plt.scatter(dataset_flattened_inputs[0], dataset_flattened_inputs[1], c="blue", s=200)
    # plt.scatter(dataset_flattened_inputs[2], dataset_flattened_inputs[3], c="red", s=200)
    # plt.scatter(dataset_flattened_inputs[3], dataset_flattened_inputs[4], c="green", s=200)
    # plt.show()
    #
    # mlp.train_classification_stochastic_gradient_backpropagation(model7, dataset_flattened_inputs, dataset_flattened_outputs)
    # predicted_values = [mlp.predict_mlp_model_classification(model7, p) for p in points]
    #
    # classes = [np.argmax(v) for v in predicted_values]
    #
    # colors = ['blue' if c == 0 else ('red' if c == 1 else 'green') for c in classes]
    #
    # plt.scatter([p[0] for p in points], [p[1] for p in points], c=colors)
    # plt.scatter(dataset_flattened_inputs[0], dataset_flattened_inputs[1], c="blue", s=200)
    # plt.scatter(dataset_flattened_inputs[2], dataset_flattened_inputs[3], c="red", s=200)
    # plt.scatter(dataset_flattened_inputs[4], dataset_flattened_inputs[5], c="green", s=200)
    # plt.show()
    #
    

    # mlp.free_MLP(resultat7)