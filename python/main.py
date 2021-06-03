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
    #     [1, 1],
    #     [2, 3],
    #     [3, 3]
    # ]
    #
    # dataset_expected_outputs = [
    #     1,
    #     -1,
    #     -1
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

    dataset_inputs = [
        [-5],
        [4],
        [4]

    ]

    dataset_expected_outputs = [
        5.2,
        7,
        8.5
    ]

    resultat2 = lm.create_linear_model(2)
    model2 = resultat2[0]

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        flattened_dataset_inputs.append(p[0])

    test_dataset_inputs = [i for i in range(-10, 11)]
    predicted_outputs = [lm.predict_linear_model_regression(model2, [p]) for p in test_dataset_inputs]


    plt.plot(test_dataset_inputs, predicted_outputs)
    plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()

    lm.train_regression_pseudo_inverse_linear_model(model2, flattened_dataset_inputs, dataset_expected_outputs)

    test_dataset_inputs = [i for i in range(-10, 11)]
    predicted_outputs = [lm.predict_linear_model_regression(model2, [p]) for p in test_dataset_inputs]

    plt.plot(test_dataset_inputs, predicted_outputs)
    plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()

    print("Je suis dead ", lm.destroy_linear_model(resultat2))

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour la classification
    # ----------------------------------------------------------------------------------

    # dataset_inputs = [
    #     [0, 0],
    #     [1, 1],
    #     [0, 1],
    #     [1, 0],
    # ]
    #
    # dataset_expected_outputs = [
    #     -1,
    #     -1,
    #     1,
    #     1,
    # ]
    # # mlp.create_mlp_model([2, 2, 1])
    # resultat3 = mlp.create_mlp_model([2,1])
    # model3 = resultat3[0]
    #
    #
    # test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    # colors = ["blue" if output >= 0 else "red" for output in dataset_expected_outputs]
    #
    # predicted_outputs = [mlp.predict_mlp_model_classification(model3, p) for p in test_dataset]
    #
    # print(" Je suis predicted_outputs: ", predicted_outputs)
    #
    # predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # mlp.train_classification_stochastic_gradient_backpropagation(model3,
    #                                                                    flattened_dataset_inputs,
    #                                                                    dataset_expected_outputs,
    #                                                                    alpha=0.001,
    #                                                                    iterations_count=1000000)
    #
    #
    # predicted_outputs = [mlp.predict_mlp_model_classification(model3, p) for p in test_dataset]
    # predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # mlp.free_MLP(resultat3)

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour la regression
    # ----------------------------------------------------------------------------------

    # dataset_inputs = [
    #     [-5],
    #     [4],
    #     [6],
    # ]
    #
    # dataset_expected_outputs = [
    #     1.2,
    #     7,
    #     8.3
    # ]
    #
    # resultat5 = mlp.create_mlp_model([1, 3, 1])
    # model5 = resultat5[0]
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #
    # test_dataset_inputs = [i for i in range(-10, 11)]
    # predicted_outputs = [mlp.predict_mlp_model_regression(model5, [p]) for p in test_dataset_inputs]
    #
    # plt.plot(test_dataset_inputs, predicted_outputs)
    # plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    # plt.axis([-10, 10, -10, 10])
    # plt.show()
    #
    # mlp.train_regression_stochastic_gradient_backpropagation(model5,
    #                                                                flattened_dataset_inputs,
    #                                                                dataset_expected_outputs)
    #
    # test_dataset_inputs = [i for i in range(-10, 11)]
    # predicted_outputs = [mlp.predict_mlp_model_regression(model5, [p]) for p in test_dataset_inputs]
    #
    # plt.plot(test_dataset_inputs, predicted_outputs)
    # plt.scatter([p[0] for p in dataset_inputs], dataset_expected_outputs, s=200)
    # plt.axis([-10, 10, -10, 10])
    # plt.show()
    #
    # mlp.free_MLP(resultat5)

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour de la classification à 3 classes
    # ----------------------------------------------------------------------------------

    # dataset_inputs = [
    #     [0, 0],
    #     [0.5, 0.5],
    #     [1, 0],
    # ]
    #
    # dataset_expected_outputs = [
    #     [1, -1, -1],
    #     [-1, 1, -1],
    #     [-1, -1, 1]
    # ]
    #
    # resultat6 = mlp.create_mlp_model([2, 50, 3])
    # model6 = resultat6[0]
    # test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    # colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in
    #           dataset_expected_outputs]
    #
    # print(" Je suis dataset_expected_outputs: ", dataset_expected_outputs)
    # print(" Je suis colors: ", colors)
    #
    #
    # predicted_outputs = [mlp.predict_mlp_model_classification(model6, p) for p in test_dataset]
    # print(" Je suis predicted_outputs: ", predicted_outputs)
    # predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
    #                             output in predicted_outputs]
    # print(" Je suis predicted_outputs_colors: ", predicted_outputs_colors)
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in dataset_inputs:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # flattened_dataset_outputs = []
    # for p in dataset_expected_outputs:
    #     flattened_dataset_outputs.append(p[0])
    #     flattened_dataset_outputs.append(p[1])
    #     flattened_dataset_outputs.append(p[2])
    #
    # mlp.train_classification_stochastic_gradient_backpropagation(model6,
    #                                                                    flattened_dataset_inputs,
    #                                                                    flattened_dataset_outputs)
    #
    # predicted_outputs = [mlp.predict_mlp_model_classification(model6, p) for p in test_dataset]
    # predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
    #                             output in predicted_outputs]
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in dataset_inputs], [p[1] for p in dataset_inputs], c=colors, s=200)
    # plt.show()
    #
    # mlp.free_MLP(resultat6)

    # ----------------------------------------------------------------------------------
    #               Utilisation du MLP pour de la classification à 3 classes //EXOS
    # ----------------------------------------------------------------------------------

    # X = np.random.random((500, 2)) * 2.0 - 1.0
    # Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
    #               [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
    #               [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
    #               [0, 0, 0] for p in X])
    #
    # resultat7 = mlp.create_mlp_model([2, 3])
    # model7 = resultat7[0]
    # test_dataset = [[float(x1) / 20, float(x2) / 20] for x1 in range(-25, 25) for x2 in range(-25, 25)]
    #
    # colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in
    #           Y]
    #
    # predicted_outputs = [mlp.predict_mlp_model_classification(model7, p) for p in test_dataset]
    # predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
    #                             output in predicted_outputs]
    #
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    # plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    # plt.show()
    #
    # flattened_dataset_inputs = []
    # for p in X:
    #     flattened_dataset_inputs.append(p[0])
    #     flattened_dataset_inputs.append(p[1])
    #
    # flattened_dataset_outputs = []
    # for p in Y:
    #     flattened_dataset_outputs.append(p[0])
    #     flattened_dataset_outputs.append(p[1])
    #     flattened_dataset_outputs.append(p[2])
    #
    # mlp.train_classification_stochastic_gradient_backpropagation(model7,
    #                                                              flattened_dataset_inputs,
    #                                                              flattened_dataset_outputs,
    #                                                              alpha=0.01,
    #                                                              iterations_count=1000000)
    #
    # predicted_outputs = [mlp.predict_mlp_model_classification(model7, p) for p in test_dataset]
    # predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
    #                             output in predicted_outputs]
    #
    # plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors, s=10)
    # plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=100)
    # plt.show()
    #
    # mlp.free_MLP(resultat7)

    # ----------------------------------------------------------------------------------
    #               Utilisation du Modèle linéaire pour la régression //EXOS
    # ----------------------------------------------------------------------------------

    X = np.array([
        [-4],
        [2]
    ])
    Y = np.array([
        2,
        3
    ])

    resultat2 = lm.create_linear_model(1)
    model2 = resultat2[0]

    flattened_dataset_inputs = []
    for p in X:
        flattened_dataset_inputs.append(p[0])

    test_dataset_inputs = [i for i in range(-10, 11)]
    predicted_outputs = [lm.predict_linear_model_regression(model2, [p]) for p in test_dataset_inputs]

    plt.plot(test_dataset_inputs, predicted_outputs)
    plt.scatter([p[0] for p in X], Y, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()

    lm.train_regression_pseudo_inverse_linear_model(model2, flattened_dataset_inputs, Y)

    test_dataset_inputs = [i for i in range(-10, 11)]
    predicted_outputs = [lm.predict_linear_model_regression(model2, [p]) for p in test_dataset_inputs]

    plt.plot(test_dataset_inputs, predicted_outputs)
    plt.scatter([p[0] for p in X], Y, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()

    lm.destroy_linear_model(resultat2)