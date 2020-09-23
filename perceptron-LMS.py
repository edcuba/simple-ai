import matplotlib.pyplot as plt
import numpy as np
import random

data = [
    (1, -1, 3, 0),
    (1, -2, 2, 0),
    (1, -3, 1, 0),
    (1, 2, -2, 1),
    (1, 0, 1, 1),
    (1, -1, 1, 1)
]

def plot(weights, data):
    plt.figure()
    plt.axis([-4,4,-4,4])
    for point in data:
        color = "green"
        if point[-1] == 1:
            color = "red"
        plt.scatter(point[1], point[2], color=color)
    plt.plot(x_l, [(weights[0] + weights[1] * x)/(-weights[2]) for x in x_l])
    plt.show()

weights = [
    random.random(), # bias
    random.random(), # w1
    random.random(), # w2
]
learning_rate = 0.2

x_l = np.linspace(-4, 4, 100)

def predict(weights, x):
    return sum([w * x[i] for i, w in enumerate(weights)])

def update(weights, error, x):
    weights_n = [w + learning_rate * error * x[i] for i, w in enumerate(weights)]
    return weights_n

wrong = 1
while wrong:
    plot(weights, data)
    print(",".join([str(w) for w in weights]))
    wrong = 0
    data_n = data[:]
    random.shuffle(data_n)
    for x in data_n:
        prediction = predict(weights, x)
        pred_cls = 1.0 if prediction >= 0.0 else 0.0
        if pred_cls != x[-1]:
            weights = update(weights, x[-1] - prediction, x)
            wrong += 1



