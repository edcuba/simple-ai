import torch
import torchvision
import torch.nn as nn


# Load the datasets
root = "data"
transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)


# loaders
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=100)

in_dim = 28 * 28


# Create a simple fully connected network
class Classifier(nn.Module):

    def __init__(self, inputs=1, outputs=1, neurons=1):
        super(Classifier, self).__init__()

        self.W1 = nn.Linear(inputs, neurons)
        self.activation = nn.Sigmoid()

        self.W2 = nn.Linear(neurons, outputs)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        a = self.W1(x)
        h = self.activation(a)
        z = self.W2(h)
        y = self.output(z)
        return y


def test(model, loss, test_loader, reshape=True):
    # Evaluation on Test set
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, t in test_loader:
            if reshape:
                x = data.view((data.shape[0], 28*28))
            else:
                x = data
            z = model(x)
            pred = z.argmax(dim=1, keepdim=True)
            correct += pred.eq(t.view_as(pred)).sum().item()
            test_loss += loss(z, t)

    set_size = len(test_loader.dataset)
    test_loss /= set_size

    print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, set_size, 100. * correct / set_size))


def train(model, loss, train_loader, optimizer, epoch, reshape=True):
    model.train()
    for data, t in train_loader:
        if reshape:
            x = data.view((data.shape[0], in_dim))
        else:
            x = data
        optimizer.zero_grad()
        z = model(x)
        J = loss(z, t)
        J.backward()
        optimizer.step()
    print(f'Training Epoch: {epoch} Training Loss: {J.item():.4f}')


model = Classifier(inputs=in_dim, outputs=10, neurons=30)
loss = torch.nn.CrossEntropyLoss()

optim = torch.optim.SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9
)

for epoch in range(20):
    train(model, loss, train_loader, optim, epoch)
    test(model, loss, test_loader)
