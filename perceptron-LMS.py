import matplotlib.pyplot as plt
import numpy as np
import random

# x0 is always 1 -> bias on the first position of the weights matrix
data = [
    (1, -1, 3, 0),
    (1, -2, 2, 0),
    (1, -3, 1, 0),
    (1, 2, -2, 1),
    (1, 0, 1, 1),
    (1, -1, 1, 1)
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
    return [w + learning_rate * error * x[i] for i, w in enumerate(weights)]

def get_error(weights, data):
    activations = [predict(weights, x) for x in data]
    predictions = [1 if a >= 0 else 0 for a in activations]
    return sum([(x[-1] - a)**2 if p != x[-1] else 0 for x, p, a in zip(data, predictions, activations)])

error = get_error(weights, data)
print(",".join([str(w) for w in weights]))
print("Error:", error)
plot(weights, data)

while error > 0:
    for x in shuffle(data):
        activation = predict(weights, x)
        prediction = 1 if activation >= 0 else 0
        if prediction != x[-1]:
            weights = update(weights, x[-1] - activation, x)
    error = get_error(weights, data)

    plot(weights, data)
    print(",".join([str(w) for w in weights]))
    print("Error:", error)
