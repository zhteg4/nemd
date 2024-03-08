import torch
from torch import nn  # building blocks for neural network
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path('Models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = '01_pytorch_wokflow_model_0.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=-1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='train data')
    plt.scatter(test_data, test_labels, c='g', label='test data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    plt.legend(prop={'size': 12})
    plt.show()


torch.manual_seed(42)


class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(
            torch.rand(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


model_0 = LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

print(model_0.state_dict())
epochs = 1000
for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)
    print(f'Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}')
print(model_0.state_dict())
plot_predictions(predictions=test_pred)

torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
