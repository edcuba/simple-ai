import matplotlib.pyplot as plt
import numpy as np
import random

# x0 is always 1 -> bias on the first position of the weights matrix
data = [
    (1, -1, 3, [0,0]),
    (1, -2, 2, [0,0]),
    (1, -3, 1, [0,1]),
    (1, 2, -2, [0,1]),
    (1, 3, 1, [1,0]),
    (1, 2, 1, [1,0])
]

def plot(weights, data):
    plt.figure()
    plt.axis([-4,4,-4,4])
    for point in data:
        color = "green"
        if point[-1] == [0,1]:
            color = "red"
        if point[-1] == [1,0]:
            color = "blue"
        plt.scatter(point[1], point[2], color=color)
    for w in weights:
        plt.plot(x_l, [(w[0] + w[1] * x)/(-w[2]) for x in x_l])
    plt.show()

weights = [
    [random.random(), random.random(), random.random()],
    [random.random(), random.random(), random.random()],
]
learning_rate = 0.2

x_l = np.linspace(-4, 4, 100)

def predict(weights, x):
    return [sum([w * x[i] for i, w in enumerate(W)]) for W in weights]

def update(weights, error, x):
    weights_n = [[w + learning_rate * error[iw] * x[i] for i, w in enumerate(W)] for iw, W in enumerate(weights)]
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
        pred_cls = [1.0 if p >= 0.0 else 0.0 for p in prediction]
        if pred_cls != x[-1]:
            err = [c - pred_cls[i] for i, c in enumerate(x[-1])]
            weights = update(weights, err, x)
            wrong += 1
