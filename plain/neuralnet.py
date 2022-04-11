import numpy as np

class NeuralNet:

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

    def descent(self, X, T, eta, momentum=None, mu=0.999):
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

        if momentum:
            self.w1 -= eta * mu * momentum[0]
            self.w2 -= eta * mu * momentum[1]

        return (gw1, gw2)

    def fit(self, X, T, eta=0.0001, epochs=10000, batch_size=64, use_momentum=True):
        # Batched stochastic gradient descent
        Y = self.predict(X.T)
        loss = self.loss(T.T, Y)
        print(f"Initial loss {loss}")

        history = [loss]
        momentum = None

        for e in range(epochs):
            for b, t in self.batch(X, T, batch_size):
                momentum = self.descent(b.T, t.T, eta, momentum if use_momentum else None)
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
        return self.forward(X)[0]

    def accuracy(self, X, T):
        return None, None
