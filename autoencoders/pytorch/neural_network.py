import torch


class NeuralNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in kwargs["units"]
            ]
        )

    def forward(self, features):
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == 0:
                activations[index] = torch.relu(layer(activations[index - 1]))
        logits = activations[len(activations) - 1]
        return logits


def epoch_train(
    model: torch.nn.Module,
    optimizer: object,
    data_loader: torch.utils.data.DataLoader,
    criterion: object,
    device: object,
) -> float:
    epoch_loss = 0
    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.view(batch_features.shape[0], -1)
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_labels)
        train_loss.backward()
        optimizer.step()
        epoch_loss += train_loss.item()
    return epoch_loss


def train(
    model: torch.nn.Module,
    optimizer: object,
    data_loader: torch.utils.data.DataLoader,
    criterion: object,
    epochs: int,
    device: object,
) -> object:
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = epoch_train(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
        )
        epoch_loss /= len(data_loader)
        train_loss.append(epoch_loss)
        print(f"epoch {epoch + 1}/{epochs} : mean loss = {epoch_loss:.6f}")
    return train_loss
