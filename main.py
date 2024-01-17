import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt

def partial_function(f___, input, pos, value):
    tmp = input[pos]
    input[pos] = value
    ret = f___(*input)
    input[pos] = tmp
    return ret

def gradient(function, input):
    """Частная произвоздная по каждому из параметров функции f(т.е. градиент)"""

    ret = np.empty(len(input))
    for i in range(len(input)):
        fg = lambda x: partial_function(function, input, i, x)
        ret[i] = nd.Derivative(fg)(input[i])
    return ret

def next_point(x, y, gx, gy, step) -> tuple:
    return x - step * gx, y - step * gy

def gradient_descent(function, x0, y0, tk, M):
    yield x0, y0, 0, function(x0, y0)
    e1 = 0.0001
    e2 = 0.0001
    k = 0
    while True:
        (gx, gy) = gradient(function, [x0, y0])  # 3
        if np.linalg.norm((gx, gy)) < e1:  # Шаг 4. Проверить выполнение критерия окончания
            break
        if k >= M:  # Шаг 5
            break
        x1, y1 = next_point(x0, y0, gx, gy, tk)  # 7
        f1 = function(x1, y1)
        f0 = function(x0, y0)
        while not f1 < f0:  # 8 условие
            tk = tk / 2
            x1, y1 = next_point(x0, y0, gx, gy, tk)
            f1 = function(x1, y1)
            f0 = function(x0, y0)
        if np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < e2 and abs(f1 - f0) < e2:  # 9
            x0, y0 = x1, y1
            break
        else:
            k += 1
            x0, y0 = x1, y1
            yield x0, y0, k, f1

def function(x, y):
    return np.sin(x) + np.cos(y)
def draw_gradient_descent(func, grad):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-5, 5, 50)
    y = x
    x, y = np.meshgrid(x, y)
    z = func(x, y)
    ax.plot_surface(x, y, z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    point, = ax.plot([], [], [], 'ro', markersize=7, zorder=5)
    grad = list(grad)
    frame = 0

    min_point = min(grad, key=lambda t: t[3])
    min_x, min_y, min_f = min_point[0], min_point[1], min_point[3]
    ax.plot([min_x], [min_y], [min_f], 'go', markersize=10, zorder=4)

    while frame < len(grad):
        x_point, y_point, _, _ = grad[frame]
        point.set_data([x_point], [y_point])
        point.set_3d_properties(func(x_point, y_point))
        plt.pause(0.01)
        frame += 1
    plt.show()

x0, y0 = 4, 4
tk = 0.1
M = 50

grad = gradient_descent(function, x0, y0, tk, M)
draw_gradient_descent(function, grad)