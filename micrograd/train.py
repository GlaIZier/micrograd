from micrograd.core import Scalar, MLP

if __name__ == '__main__':
    mlp = MLP(2, 3, 1)
    x1, x2 = Scalar(1.0), Scalar(2.0)
    print(mlp([x1, x2]))
