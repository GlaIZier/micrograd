
class Scalar:

    def __init__(self, v, grad=0, children=tuple(), back=lambda: None, op=""):
        self.v = v
        self.children = children
        self.back = back
        self.grad = grad
        self.op = op

    def __repr__(self):
        return f"Scalar(v={self.v}, grad={self.grad})"

    def __add__(self, o):
        s = Scalar(self.v + o.v, children=(self, o), op="+")

        def __back():
            self.grad += s.grad
            o.grad += s.grad

        s.back = __back
        return s

    def __mul__(self, o):
        s = Scalar(self.v * o.v, children=(self, o), op="*")

        def __back():
            self.grad += s.grad * o.v
            o.grad += s.grad * self.v

        s.back = __back
        return s

    def __pow__(self, o):
        assert not isinstance(o, Scalar)
        s = Scalar(self.v ** o, children=self.children, op="^")

        def __back():
            self.grad += s.grad * o * self.v ** (o - 1)

        s.back = __back
        return s

    def __neg__(self):
        # return Scalar(v=-self.v, grad=-self.grad, children = self.children, back=self.back, op = self.op)
        return Scalar(-1) * self

    def __sub__(self, o):
        return self + (-o)

    # def __sub__(self, o):
    #     s = Scalar(self.v - o.v, children=(self, o), op="-")
    #     def __back():
    #         self.grad += s.grad
    #         o.grad -= s.grad
    #     s.back = __back
    #     return s

    def __truediv__(self, o):
        t = o ** -1
        return self * t

    def backprop(self):
        from collections import deque
        children = deque()
        self.grad = 1
        children.append(self)
        while children:
            c = children.popleft()
            c.back()
            children.extend(c.children)