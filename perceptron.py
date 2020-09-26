import matplotlib.pyplot as plt
import numpy as np
import random

data = [
    (1, 2, 2, 0.0),
    (1, -2, 2, 0.0),
    (1, 1, -2, 1.0),
    (1, -1, 1, 1.0)
]

def shuffle(input_data):
    data_n = input_data[:]
    random.shuffle(data_n)
    return data_n

def plot(weights, data):
    plt.figure()
    plt.axis([-4,4,-4,4])
    plt.scatter([x[1] for x in data[:2]], [x[2] for x in data[:2]], color="green")
    plt.scatter([x[1] for x in data[2:]], [x[2] for x in data[2:]], color="red")
    plt.plot(x_l, [(weights[0] + weights[1] * x)/(-weights[2]) for x in x_l])
    plt.show()

weights = [
    random.random(),
    random.random(),
    random.random(),
]
learning_rate = 0.2

x_l = np.linspace(-4, 4, 100)

def predict(weights, x):
    return 1 if sum([w * x[i] for i, w in enumerate(weights)]) >= 0 else 0

def update(weights, error, x):
    return [w + learning_rate * error * x[i] for i, w in enumerate(weights)]

wrong = 1
while wrong:
    wrong = 0
    plot(weights, data)
    print(",".join([str(w) for w in weights]))
    for x in shuffle(data):
        prediction = predict(weights, x)
        if prediction != x[-1]:
            wrong += 1
            weights = update(weights, x[-1] - prediction, x)
