import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x, y):
    return 2 * np.power(x, 2) + 3 * np.power(y, 2) + (4 * x * y) - (6 * x) - (3 * y)

def simplex_method(x1, x2):
    triangle = []
    x0 = float(x1)
    z0 = float(x2)
    e = 5
    alpha = 2
    points = [[x0 - alpha / 2, z0 - 0.29 * alpha],
              [x0 + alpha / 2, z0 - 0.29 * alpha],
              [x0, z0 + 0.58 * alpha]]
    func = [f(points[0][0], points[0][1]),
            f(points[1][0], points[1][1]),
            f(points[2][0], points[2][1])]
    triangle.append(list(points))
    x_min = x0
    z_min = z0
    y_min = f(x0, z0)
    flag = 0
    x_max = x0
    z_max = z0
    while abs(f(x_max, z_max) - min(func)) > e:
        if flag:
            flag = 0
            x0, z0 = points[func.index(min(func))]
            x0 += points[func.index(max(func))][0]
            z0 += points[func.index(max(func))][1]
            x0 /= 2
            z0 /= 2
            points.remove(points[func.index(max(func))])
            func.remove(max(func))
            x1, z1 = points[func.index(min(func))]
            x1 += points[func.index(max(func))][0]
            z1 += points[func.index(max(func))][1]
            x1 /= 2
            z1 /= 2
            points.remove(points[func.index(max(func))])
            func.remove(max(func))
            func.append(f(x0, z0))
            points.append([x0, z0])
            func.append(f(x1, z1))
            points.append([x1, z1])
        else:
            x_max, z_max = points[func.index(max(func))]
            points.remove(points[func.index(max(func))])
            func.remove(max(func))
            x0 = 0 - x_max
            z0 = 0 - z_max
            for value in points:
                x0 += value[0]
                z0 += value[1]
            if f(x0, z0) > max(func):
                func.append(f(x_max, z_max))
                points.append([x_max, z_max])
                flag = 1
            else:
                func.append(f(x0, z0))
                points.append([x0, z0])

        if f(x_min, z_min) > min(func):
            x_min, z_min = points[func.index(min(func))]
            y_min = min(func)

        triangle.append(list(points))

    x_min = round(x_min, 2)
    z_min = round(z_min, 2)
    y_min = round(y_min, 2)
    return triangle, x_min, y_min, z_min

def makeData():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xgrid, ygrid = np.meshgrid(x, y)
    z = f(xgrid, ygrid)
    return xgrid, ygrid, z

x, y, z = makeData()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=10, cstride=10, alpha=0.4, cmap="seismic")

x1 = 3
x2 = -1
triangle, X, Z, Y = simplex_method(x1, x2)
for tr in triangle:
    if min(tr)[1] > Y:
        ax.scatter(tr[0][0], tr[0][1], min(tr), c="red", s=10)
    else:
        ax.scatter(X, Y, Z, c="red", s=10)
plt.show()
