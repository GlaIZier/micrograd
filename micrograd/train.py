import random

from micrograd.core import Scalar, MLP


def gen_data(n=128):
    for _ in range(n):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)
        y = 1.0 if x1**2 + x2**2 > 1.0 else 0.0
        yield x1, x2, y

def train(mlp, data, epochs=3, lr=0.001):
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x1, x2, y in data:
            y_pred = mlp([Scalar(x1), Scalar(x2)])[0]
            y = Scalar(y)
            loss = (y_pred - y) ** 2
            total_loss += loss.v
            # loss = sum((y_pred_i - y_i) ** 2 for y_pred_i, y_i in zip(y_pred, y))
            loss.backprop()
            for p in mlp.parameters():
                p.v -= lr * p.grad
            mlp.zero_grad()
        print(f"Epoch={epoch}. Total loss {total_loss}")

def train_batched(mlp, data, epochs=3, lr=0.001):
    for epoch in range(1, epochs + 1):
        total_loss = 0
        cur_loss = Scalar(0)
        for i, d in enumerate(data):
            x1, x2, y = d
            y_pred = mlp([Scalar(x1), Scalar(x2)])[0]
            y = Scalar(y)
            loss = (y_pred - y) ** 2
            total_loss += loss.v
            cur_loss += loss
            if i + 1 % 32 == 0:
                cur_loss.backprop()
                for p in mlp.parameters():
                    p.v -= lr * p.grad
                mlp.zero_grad()
                cur_loss = Scalar(0)
            # loss = sum((y_pred_i - y_i) ** 2 for y_pred_i, y_i in zip(y_pred, y))

        print(f"Epoch={epoch}. Total loss {total_loss}")

def simple_forward():
    mlp = MLP(2, 3, 1)
    x1, x2 = Scalar(1.0), Scalar(2.0)
    print(mlp([x1, x2]))

def run():
    mlp = MLP(2, 3, 1)
    data = list(gen_data(128))
    train_batched(mlp, data, epochs=100, lr=0.01)
    print(f"1, 2: {mlp([Scalar(1.0), Scalar(2.0)])}")
    print(f"0, 0: {mlp([Scalar(0.0), Scalar(0.0)])}")
    print(f"-0.5, -0.5: {mlp([Scalar(-0.5), Scalar(-0.5)])}")
    print(f"-3, -3: {mlp([Scalar(-3), Scalar(-3)])}")
    print(f"-1, 2: {mlp([Scalar(-1), Scalar(2)])}")


if __name__ == '__main__':
    run()
