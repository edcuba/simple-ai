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

def shuffle(input_data):
    data_n = input_data[:]
    random.shuffle(data_n)
    return data_n

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
    return [[w + learning_rate * error[iw] * x[i] for i, w in enumerate(W)] for iw, W in enumerate(weights)]

def get_error(weights, data):
    err = 0
    for x in shuffle(data):
        prediction = predict(weights, x)
        for i, c in enumerate(x[-1]):
            pred_cls = 1.0 if prediction[i] >= 0.0 else 0.0
            if c != pred_cls:
                err += (c - prediction[i])**2
    return err

error = get_error(weights, data)
print(",".join([str(w) for w in weights]))
print("Error:", error)
plot(weights, data)

while error > 0:
    for x in shuffle(data):
        prediction = predict(weights, x)
        err = [0, 0]
        for i, c in enumerate(x[-1]):
            pred_cls = 1.0 if prediction[i] >= 0.0 else 0.0
            if c != pred_cls:
                err[i] = c - prediction[i]
        weights = update(weights, err, x)
    error = get_error(weights, data)

    print(",".join([str(w) for w in weights]))
    print("Error:", error)
    plot(weights, data)
