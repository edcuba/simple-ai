import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# Load the datasets
root = "data"
transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)


# loaders
train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=100)

in_dim = 28 * 28

# Create a convolutional network
class ConvClassifier(nn.Module):

    def __init__(self, outputs=10):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 128)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


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


model_cnn = ConvClassifier(outputs=10)

loss = torch.nn.CrossEntropyLoss()

optim_cnn = torch.optim.SGD(
    params=model_cnn.parameters(),
    lr=0.01,
    momentum=0.9
)

for epoch in range(5):
    train(model_cnn, loss, train_loader, optim_cnn, epoch, reshape=False)
    test(model_cnn, loss, test_loader, reshape=False)
