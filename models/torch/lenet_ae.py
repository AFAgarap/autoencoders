import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=(2, 2)
                )
        self.conv_layer_2 = nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=(2, 2)
                )

    def forward(self, features):
        activation = self.conv_layer_1(features)
        activation = F.relu(activation)
#        print(activation.size())
        activation = self.conv_layer_2(activation)
        code = F.relu(activation)
#        print(code.size())
        return code


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt_layer_1 = nn.ConvTranspose2d(
                in_channels=16,
                out_channels=6,
                kernel_size=5,
                stride=(2, 2)
                )
        self.convt_layer_2 = nn.ConvTranspose2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=(2, 2)
                )
        self.convt_layer_3 = nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=4,
                stride=(1, 1)
                )

    def forward(self, features):
        activation = self.convt_layer_1(features)
        activation = F.relu(activation)
#        print(activation.size())
        activation = self.convt_layer_2(activation)
        activation = F.relu(activation)
#        print(activation.size())
        activation = self.convt_layer_3(activation)
        reconstructed = F.relu(activation)
#        print(reconstructed.size())
        return reconstructed


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, features):
        code = self.encoder(features)
#        print(code.size())
        reconstructed = self.decoder(code)
#        print(reconstructed.size())
        return reconstructed


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
        root='~/torch_datasets',
        train=True,
        transform=transform,
        download=True
        )

test_dataset = torchvision.datasets.MNIST(
        root='~/torch_datasets',
        train=False,
        transform=transform,
        download=True
        )

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
        )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss().to(device)

print(model)
print([parameter.size() for parameter in model.parameters()])

epochs = 20

for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
    loss = loss / len(train_loader)
    print('epoch : {}/{}, loss = {:.6f}'.format(epoch + 1, epochs, loss))

with torch.no_grad():
    number = 10
    plt.figure(figsize=(20, 4))
    for index in range(number):
        # display original
        ax = plt.subplot(2, number, index + 1)
        plt.imshow(test_dataset.data[index].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, number, index + 1 + number)
        test_data = test_dataset.data[index]
        test_data = test_data.to(device)
        test_data = test_data.float()
        test_data = test_data.view(-1, 1, 28, 28)
        output = model(test_data)
        plt.imshow(output.cpu().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
