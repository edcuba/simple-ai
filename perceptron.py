import numpy as np
import matplotlib.pyplot as plt

### DATA SETUP ###
# x0 is always 1 -> bias on the first position of the weights matrix
data = np.array([
    (1, -1, 3, 0.0),
    (1, -2, 2, 0.0),
    (1, -3, 1, 0.0),
    (1, 2, -2, 1.0),
    (1, 0, 1, 1.0),
    (1, -1, 1, 1.0)
])

### Plotting ###
x_l = np.linspace(-4, 4, 100)
def plot(ax, weights, data):
    plt.show(block=False)
    ax.clear()
    plt.axis([-4, 4, -4, 4])
    print("Weights:", weights)
    for point in data:
        color = "green" if int(point[-1]) == 1 else "red"
        ax.scatter(point[1], point[2], color=color)
    ax.plot(x_l, [(weights[0] + weights[1] * x)/(-weights[2]) for x in x_l])
    plt.pause(1)


### Implementation ###

learning_rate = 0.2
def update(weights, error, x):
    return [w + learning_rate * error * x[i] for i, w in enumerate(weights)]

# Rosenblatt algorithm
def predict(weights, x):
    return 1 if sum([w * x[i] for i, w in enumerate(weights)]) >= 0 else 0

def train(weights):
    ax = plt.subplot()
    error = 1
    while error:
        error = 0
        plot(ax, weights, data)
        np.random.shuffle(data)
        for x in data:
            err = x[-1] - predict(weights, x)
            weights = update(weights, err, x)
            error += abs(err)
    return weights

# Least Mean Squares Implementation
def get_error_lms(weights, data):
    activations = [predict_lms(weights, x) for x in data]
    predictions = [1 if a >= 0 else 0 for a in activations]
    error = sum([(x[-1] - a)**2 if p != x[-1] else 0 for x, p, a in zip(data, predictions, activations)])
    print("LMS Error:", error)
    return error

def predict_lms(weights, x):
    return sum([w * x[i] for i, w in enumerate(weights)])

def train_lms(weights):
    ax = plt.subplot()
    error = get_error_lms(weights, data)
    while error > 0:
        np.random.shuffle(data)
        plot(ax, weights, data)
        for x in data:
            activation = predict_lms(weights, x)
            prediction = 1 if activation >= 0 else 0
            if prediction != x[-1]:
                weights = update(weights, x[-1] - activation, x)
        error = get_error_lms(weights, data)
    plot(ax, weights, data)
    return weights


### Run Rosenblatt's algorithm ###
fig = plt.figure()

# randomly initialize: bias, w1, w2
weights = np.random.rand(3)
train(weights)
plt.show()


### Run Least Mean Squares ###
fig = plt.figure()

# randomly initialize: bias, w1, w2
weights = np.random.rand(3)
train_lms(weights)
plt.show()
