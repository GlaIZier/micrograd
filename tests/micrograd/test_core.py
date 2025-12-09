from micrograd.core import Scalar


def test_add_mul():
    a = Scalar(1)
    b = Scalar(2)
    c = Scalar(3)
    d = Scalar(4)
    e = Scalar(5)

    f = d * e
    g = c + f
    h = b * g
    z = a + h
    z.backprop()
    print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}, g={g}, h={h}, z={z}")

    eps = 0.00001
    grad_a = (((a.v + eps) + b.v * (c.v + d.v * e.v)) - (a.v + b.v * (c.v + d.v * e.v))) / eps
    grad_b = ((a.v + (b.v + eps) * (c.v + d.v * e.v)) - (a.v + b.v * (c.v + d.v * e.v))) / eps
    grad_c = ((a.v + b.v * (c.v + eps + d.v * e.v)) - (a.v + b.v * (c.v + d.v * e.v))) / eps
    grad_d = ((a.v + b.v * (c.v + (d.v + eps) * e.v)) - (a.v + b.v * (c.v + d.v * e.v))) / eps
    grad_e = ((a.v + b.v * (c.v + d.v * (e.v + eps))) - (a.v + b.v * (c.v + d.v * e.v))) / eps
    print(f"grad_a={grad_a}, grad_b={grad_b}, grad_c={grad_c}, grad_d={grad_d}, grad_e={grad_e}")

    assert abs(a.grad - grad_a) < 1e-5
    assert abs(b.grad - grad_b) < 1e-5
    assert abs(c.grad - grad_c) < 1e-5
    assert abs(d.grad - grad_d) < 1e-5
    assert abs(e.grad - grad_e) < 1e-5

def test_sub():
    a = Scalar(2)
    b = Scalar(1)
    z = a - b
    z.backprop()
    print(f"a={a}, b={b}, z={z}")

    eps = 0.00001
    grad_a = ((a.v + eps - b.v) - (a.v - b.v)) / eps
    grad_b = ((a.v - (b.v + eps)) - (a.v - b.v)) / eps
    print(f"grad_a={grad_a}, grad_b={grad_b}")

    assert abs(a.grad - grad_a) < 1e-5
    assert abs(b.grad - grad_b) < 1e-5

def test_div():
    a = Scalar(12)
    b = Scalar(4)
    z = a / b
    z.backprop()
    print(f"a={a}, b={b}, z={z}")

    eps = 0.00001
    grad_a = (((a.v + eps) / b.v) - (a.v / b.v)) / eps
    grad_b = ((a.v / (b.v + eps)) - (a.v / b.v)) / eps
    print(f"grad_a={grad_a}, grad_b={grad_b}")

    assert abs(a.grad - grad_a) < 1e-5
    assert abs(b.grad - grad_b) < 1e-5

def test_complex():
    # -a * (b^3 - c * d / (e + f^-2))
    a = Scalar(1)
    b = Scalar(2)
    c = Scalar(3)
    d = Scalar(4)
    e = Scalar(5)
    f = Scalar(6)

    z = -a * (b**3 - c * d / (e + f**-2))
    z.backprop()
    print(f"a={a}, b={b}, c={c}, d={d}, e={e}, f={f}, z={z}")

    eps = 1e-10
    grad_a = (((-(a.v + eps)) * (b.v**3 - c.v * d.v / (e.v + f.v**-2))) - (-a.v * (b.v**3 - c.v * d.v / (e.v + f.v**-2)))) / eps
    grad_b = (-a.v * (((b.v + eps) ** 3) - c.v * d.v / (e.v + f.v ** -2)) - (-a.v * (b.v ** 3 - c.v * d.v / (e.v + f.v ** -2)))) / eps
    grad_c = (-a.v * (b.v ** 3 - (c.v + eps) * d.v / (e.v + f.v ** -2)) - (-a.v * (b.v ** 3 - c.v * d.v / (e.v + f.v ** -2)))) / eps
    grad_d = (-a.v * (b.v ** 3 - c.v * (d.v + eps) / (e.v + f.v ** -2)) - (-a.v * (b.v ** 3 - c.v * d.v / (e.v + f.v ** -2)))) / eps
    grad_e = (-a.v * (b.v ** 3 - c.v * d.v / ((e.v + eps) + f.v ** -2)) - (-a.v * (b.v ** 3 - c.v * d.v / (e.v + f.v ** -2)))) / eps
    grad_f = (-a.v * (b.v ** 3 - c.v * d.v / (e.v + (f.v + eps) ** -2)) - (-a.v * (b.v ** 3 - c.v * d.v / (e.v + f.v ** -2)))) / eps

    print(f"grad_a={grad_a}, grad_b={grad_b}, grad_c={grad_c}, grad_d={grad_d}, grad_e={grad_e}, grad_f={grad_f}")
    assert abs(a.grad - grad_a) < 1e-5
    assert abs(b.grad - grad_b) < 1e-5
    assert abs(c.grad - grad_c) < 1e-5
    assert abs(d.grad - grad_d) < 1e-5
    assert abs(e.grad - grad_e) < 1e-5
    assert abs(f.grad - grad_f) < 1e-5


def test_relu():
    a = Scalar(-1)
    b = Scalar(2)

    z = a.relu() + b.relu()
    z.backprop()
    print(f"a={a}, b={b}, z={z}")
    eps = 1e-10

    grad_a = ((((a.v + eps) if (a.v + eps) > 0 else 0) + (b.v if b.v > 0 else 0)) - ((a.v if a.v > 0 else 0) + (b.v if b.v > 0 else 0))) / eps
    grad_b = (((a.v if a.v > 0 else 0) + ((b.v + eps) if (b.v + eps) > 0 else 0)) - ((a.v if a.v > 0 else 0) + (b.v if b.v > 0 else 0))) / eps
    print(f"grad_a={grad_a}, grad_b={grad_b}")
    assert abs(a.grad - grad_a) < 1e-5
    assert abs(b.grad - grad_b) < 1e-5

