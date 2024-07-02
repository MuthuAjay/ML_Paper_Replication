import torch
from tqdm.auto import tqdm
from dataset import dataset
from nn import Sequential, MaxPool2d, Flatten, Linear
from conv_layer import Conv2d
from activations import Relu
from loss import CrossEntropyLoss
from optimizer import OptimizerSG

if __name__ == "__main__":

    train_dataloader, test_dataloader = dataset()
    device = 'cpu'
    print(torch.cuda.is_available())
    model = Sequential([
        Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1, padding=1, dilation=1),
        Relu(),
        Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, dilation=1),
        Relu(),
        MaxPool2d(2, 2),
        Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, dilation=1),
        Relu(),
        Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, dilation=1),
        Relu(),
        MaxPool2d(2, 2),
        Flatten(),
        Linear(fan_in=980,
               fan_out=10)
    ])

    loss = CrossEntropyLoss()
    losses = []
    parameters = model.parameters()
    print(sum(p.nelement() for p in parameters))  # number of parameters in total
    for p in parameters:
        p.requires_grad = True
    optimizer = OptimizerSG(params=parameters, lr=0.1)
    # print(parameters[0])
    train_loss, train_acc = 0.0, 0.0

    for i in tqdm(range(1)):
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            lossi = loss(y_pred_logits, y)
            for layer in model.layers:
                layer.out.retain_grad()
            for p in parameters:
                p.grad = None
            grads = model.backward(y_pred_logits, y, loss)
            lossi.backward()
            loss.backward(y_pred_logits, y)
            optimizer.step()
            train_acc = (y_pred_logits.argmax(dim=1) == y).sum().item() / len(y)

            losses.append(lossi)
            # print(lossi.item())

        train_loss /= len(train_dataloader)

        print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f}")
