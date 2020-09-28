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
learning_rate = 0.1

def shuffle(input_data):
    data_n = input_data[:]
    random.shuffle(data_n)
    return data_n

x_l = np.linspace(-4, 4, 100)

def predict(weights, x):
    return [1 if sum([w * x[i] for i, w in enumerate(W)]) >=0 else 0 for W in weights]

def update(weights, error, x):
    return [[w + learning_rate * e * x[i] for i, w in enumerate(W)] for e, W in zip(error, weights)]

wrong = 1
while wrong:
    plot(weights, data)
    print(",".join([str(w) for w in weights]))
    wrong = 0
    for x in shuffle(data):
        prediction = predict(weights, x)
        err = [c - p for c, p in zip(x[-1], prediction)]
        weights = update(weights, err, x)
        wrong += sum([abs(e) for e in err])
