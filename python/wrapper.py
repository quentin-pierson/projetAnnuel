import numpy as np
from mlp import MLPModel

class MyMLPRawWrapper:
    def __init__(self, npl: [int], is_classification: bool = True,
                 alpha: float = 0.01, iterations_count: int = 1000):
        self.mlp = MLPModel()
        self.result = self.mlp.create_mlp_model(npl)
        self.model = self.result[0]
        self.is_classification = is_classification
        self.alpha = alpha
        self.iterations_count = iterations_count

    def fit(self, X, Y):

        if not hasattr(X, 'shape'):
            X = np.array(X)

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        if not hasattr(Y, 'shape'):
            Y = np.array(Y)

        if len(Y.shape) == 1:
            Y = np.expand_dims(Y, axis=0)

        if self.is_classification:
            self.mlp.train_classification_stochastic_gradient_backpropagation_mlp_model(self.model,
                                                                                        X.flatten(),
                                                                                        Y.flatten(),
                                                                                        self.alpha,
                                                                                        self.iterations_count)
        else:
            self.mlp.train_regression_stochastic_gradient_backpropagation_mlp_model(self.model,
                                                                                    X.flatten(),
                                                                                    Y.flatten(),
                                                                                    self.alpha,
                                                                                    self.iterations_count)

    def predict(self, X):
        if not hasattr(X, 'shape'):
            X = np.array(X)

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        results = []
        for x in X:
            if self.is_classification:
                results.append(self.mlp.predict_mlp_model_classification(self.model, x.flatten()))
            else:
                results.append(self.mpl.predict_mlp_model_regression(self.model, x.flatten()))

        return np.array(results)
