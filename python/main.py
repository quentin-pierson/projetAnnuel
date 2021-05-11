from ctypes import *
import path_to_dll as path
import matplotlib.pyplot as plt
import numpy as np
from linearModel import linearModel

lm = linearModel()

if __name__ == '__main__':

    # ----------------------------------------------------------------------------------
    #               Utilisation du Modèle linéaire pour la classification
    # ----------------------------------------------------------------------------------

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

    dataset_inputs = [
        [-5],
        [4],
        [6],
    ]

    dataset_expected_outputs = [
        5.2,
        7,
        8.3
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