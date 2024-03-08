import torch

device = 'cuda'

weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=-1).to('cuda')
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


class LinearRegression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features=1, out_features=2)

    def forward(self, x: torch.Tensor):
        return self.linear_layer(x)


torch.manual_seed(42)
model_1 = LinearRegression()
model_1.state_dict()
model_1.to(device)
next(model_1.parameters()).device

loss_fn = torch.nn.L1Loss()  # same as MAE
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

for epoch in range(100):
    model_1.train()
    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')

print(model_1.state_dict())
