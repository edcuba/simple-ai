# Multi-Class Classification
# Categorical: Iris (from sklearn)

import numpy as np
from sklearn.datasets import load_iris

np.random.seed(1)
iris = load_iris()


# 0 Sepal length in cm
# 1 Sepal width in cm
# 2 Petal length in cm
# 3 Petal width in cm
X_ = iris["data"]
X = np.hstack((np.ones((len(X_), 1)), X_))

labels = {
    0: [0, 0],
    1: [0, 1],
    2: [1, 0]
}

# Class: 0, 1, 2
T_class = iris["target"]
T = np.array([labels[v] for v in T_class])


class CategoricalClassifier:

    def __init__(self, inputs, outputs, nodes=3):
        self.nodes = nodes      # K
        self.inputs = inputs
        self.outputs = outputs  # O

        self.w1 = np.random.random((nodes, inputs))
        self.w2 = np.random.random((outputs, nodes))

    def batch(self, X, T, B):
        """
            Batch generation function
            Arguments:
                X: data
                T: targets
                B: batch size
            Returns:
                Batch generator
            Each generator shuffles the set.
            The last batch is aligned to the batch size by random.
        """
        total = X.shape[0]
        assert total == T.shape[0]

        # generate an array of shuffled indexes
        indexes = np.random.permutation(total)

        i0 = 0
        # generate full batches
        for i0 in range(0, total - B, B):
            batch_indexes = indexes[i0:i0+B]
            yield X[batch_indexes], T[batch_indexes]

        # align the last batch to the batch size
        if i0 != total - 1:
            last_pos = i0+B

            # last batch till the end
            batch_indexes = indexes[last_pos:total]
            new_indexes = np.random.permutation(total)

            # from the end of the current batch until the next B
            remaining_indexes = new_indexes[:B-(total-last_pos)]
            merged_indexes = np.hstack((batch_indexes, remaining_indexes))
            yield X[merged_indexes], T[merged_indexes]

    def forward(self, X):
        """
            Forward the batch through the network
            Arguments
                X: batch (with bias)
            Return
                Hidden layer activation
                Output layer activation
        """
        A = np.dot(self.w1, X)  # (N, K)
        H = self.sigm(A)        # (N, K)
        H[0,:] = 1              # bias
        Z = np.dot(self.w2, H)  # (N, O)
        return Z, H

    def descent(self, X, T, eta):
        # forward pass
        Z, H = self.forward(X)

        # Back propagation
        # 2nd layer gradient
        #   dJ/dW2 = (Y - T).dot(H.T)
        gw2 = np.dot((Z - T), H.T)

        # 1st layer gradient
        #   dj/dW1 = sum_(N,O)((y - t) * w2 * f * (1 - f) * x)

        gw1 = np.dot(np.dot(self.w2.T, (Z - T)) * H * (1. - H), X.T)

        self.w1 -= eta * gw1
        self.w2 -= eta * gw2

        return (gw1, gw2)

    def fit(self, X, T, eta=0.0001, epochs=10000, batch_size=64):
        # Batched stochastic gradient descent
        Y = self.predict(X.T)
        loss = self.loss(T.T, Y)
        print(f"Initial loss {loss}")

        history = [loss]

        for e in range(epochs):
            for b, t in self.batch(X, T, batch_size):
                self.descent(b.T, t.T, eta)
            Y = self.predict(X.T)
            loss = self.loss(T.T, Y)

            accuracy, _ = self.accuracy(X, T)

            if e % 1000 == 0:
                if accuracy is not None:
                    print(f"Epoch {e} loss {loss:1.4f} accuracy {accuracy:1.4f}")
                else:
                    print(f"Epoch {e} loss {loss:1.4f}")

            history.append(loss)
        return history

    def loss(self, T, Y):
        # Mean Squared Error
        return np.mean((T - Y)**2)

    def sigm(self, x):
        # Logistic function applied in the hidden layer
        return 1. / (1. + np.exp(-x))

    def predict(self, X):
        # Regression turned into binary classification
        #   by applying logistic function to the output
        z = self.forward(X)[0]
        return z

    def accuracy(self, X, T):
        Z = self.predict(X.T).T
        C = np.sum(np.argmax(Z, axis=1) == np.argmax(T, axis=1))
        A = C / T.shape[0]
        return A, C

    def loss(self, T, Z):
        # Cross-Entropy Loss of Softmax
        J = - np.sum(np.sum(T * Z, axis=0) - np.log(np.sum(np.exp(Z), axis=0)))
        return float(J)

model = CategoricalClassifier(X.shape[1], T.shape[1], nodes=10)


# Use classic gradient descent (batch size = dataset size)
model.fit(X, T, epochs=3000, eta=0.001, batch_size=T.shape[0])
accuracy, correct = model.accuracy(X, T)

print(f"Accuracy {accuracy:1.4f}, Correct {correct} of {T.shape[0]}")
