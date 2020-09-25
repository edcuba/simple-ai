import matplotlib.pyplot as plt
import numpy as np
import random

data = [
    (2, 2, 0.0),
    (-2, 2, 0.0),
    (1, -2, 1.0),
    (-1, 1, 1.0)
]

def plot(weights, bias, data):
    plt.figure()
    plt.axis([-4,4,-4,4])
    plt.scatter([x[0] for x in data[:2]], [x[1] for x in data[:2]], color="green")
    plt.scatter([x[0] for x in data[2:]], [x[1] for x in data[2:]], color="red")
    plt.plot(x_l, [(weights[0] * x + bias)/(-weights[1]) for x in x_l])
    plt.show()

weights = [
    random.random(),
    random.random(),
]
bias = random.random()
learning_rate = 0.2

x_l = np.linspace(-4, 4, 100)

def predict(weights, bias, x):
    act = weights[0] * x[0] + weights[1] * x[1] + bias
    return 1.0 if act >= 0.0 else 0.0

def update(weights, bias, error, x):
    bias_n = bias + learning_rate * error
    weights_n = [w + learning_rate * error * x[i] for i, w in enumerate(weights)]
    return weights_n, bias_n

wrong = 1
while wrong:
    wrong = 0
    plot(weights, bias, data)
    print(weights[0], ",", weights[1], ",", bias)
    for x in data:
        prediction = predict(weights, bias, x)
        if prediction != x[-1]:
            wrong += 1
            weights, bias = update(weights, bias, x[-1] - prediction, x)
