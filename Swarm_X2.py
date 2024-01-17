from math import *
import random as rnd
import matplotlib.pyplot as plt
import  numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def Rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def func(x, y):
    return x**2 + y**2

def func_ROMA(x, y):
    return 3*x*y - x**2*y - x*y**2

def rastrigin(x, y):
    A = 10
    return 20 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

class Unit:
    def __init__(self, start, end, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, function):
        self.start = start
        self.end = end
        self.currentVelocityRatio = currentVelocityRatio
        self.localVelocityRatio = localVelocityRatio
        self.globalVelocityRatio = globalVelocityRatio
        self.function = function
        self.localBestPos = self.getFirsPos()
        self.localBestScore = self.function(*self.localBestPos)
        self.currentPos = self.localBestPos[:]
        self.score = self.function(*self.localBestPos)
        self.globalBestPos = []
        self.velocity = self.getFirstVelocity()

    def getFirstVelocity(self):
        minval = -(self.end - self.start)
        maxval = self.end - self.start
        return [rnd.uniform(minval, maxval), rnd.uniform(minval, maxval)]

    def getFirsPos(self):
        return [rnd.uniform(self.start, self.end), rnd.uniform(self.start, self.end)]

    def nextIteration(self):
        rndCurrentBestPosition = [rnd.random(), rnd.random()]
        rndGlobalBestPosition = [rnd.random(), rnd.random()]
        velocityRatio = self.localVelocityRatio + self.globalVelocityRatio
        commonVelocityRatio = 2 * self.currentVelocityRatio / abs(2 - velocityRatio - sqrt(velocityRatio ** 2 - 4 * velocityRatio))
        multLocal = list(map(lambda x: x * commonVelocityRatio * self.localVelocityRatio, rndCurrentBestPosition))
        betweenLocalAndCurPos = [self.localBestPos[0] - self.currentPos[0], self.localBestPos[1] - self.currentPos[1]]
        betweenGlobalAndCurPos = [self.globalBestPos[0] - self.currentPos[0], self.globalBestPos[1] - self.currentPos[1]]
        multGlobal = list(map(lambda x: x * commonVelocityRatio * self.globalVelocityRatio, rndGlobalBestPosition))
        newVelocity1 = list(map(lambda coord: coord * commonVelocityRatio, self.velocity))
        newVelocity2 = [coord1 * coord2 for coord1, coord2 in zip(multLocal, betweenLocalAndCurPos)]
        newVelocity3 = [coord1 * coord2 for coord1, coord2 in zip(multGlobal, betweenGlobalAndCurPos)]
        self.velocity = [coord1 + coord2 + coord3 for coord1, coord2, coord3 in zip(newVelocity1, newVelocity2, newVelocity3)]
        self.currentPos = [coord1 + coord2 for coord1, coord2 in zip(self.currentPos, self.velocity)]
        newScore = self.function(*self.currentPos)
        if newScore < self.localBestScore:
            self.localBestPos = self.currentPos[:]
            self.localBestScore = newScore
        return newScore

class Swarm:
    def __init__(self, sizeSwarm, currentVelocityRatio, localVelocityRatio, globalVelocityRatio, numbersOfLife, function, start, end):
        self.sizeSwarm = sizeSwarm
        self.currentVelocityRatio = currentVelocityRatio
        self.localVelocityRatio = localVelocityRatio
        self.globalVelocityRatio = globalVelocityRatio
        self.numbersOfLife = numbersOfLife
        self.function = function
        self.start = start
        self.end = end
        self.swarm = []
        self.globalBestPos = []
        self.globalBestScore = float('inf')
        self.createSwarm()

    def createSwarm(self):
        pack = [self.start, self.end, self.currentVelocityRatio, self.localVelocityRatio, self.globalVelocityRatio, self.function]
        self.swarm = [Unit(*pack) for _ in range(self.sizeSwarm)]
        for unit in self.swarm:
            if unit.localBestScore < self.globalBestScore:
                self.globalBestScore = unit.localBestScore
                self.globalBestPos = unit.localBestPos

    def update_plot(self, i, data, sc, ax):
        x, y = data[i]
        sc.set_offsets(list(zip(x, y)))
        ax.set_title(f'Iteration {i + 1}')
        return sc,

    def startSwarm(self):
        dataForGIF = []
        for _ in range(self.numbersOfLife):
            oneDataX = []
            oneDataY = []
            for unit in self.swarm:
                oneDataX.append(unit.currentPos[0])
                oneDataY.append(unit.currentPos[1])
                unit.globalBestPos = self.globalBestPos
                score = unit.nextIteration()
                if score < self.globalBestScore:
                    self.globalBestScore = score
                    self.globalBestPos = unit.localBestPos
            dataForGIF.append([oneDataX, oneDataY])

        # 3D Визуализация движения частиц
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Создание сетки для графика функции
        x_vals = np.linspace(self.start, self.end, 100)
        y_vals = np.linspace(self.start, self.end, 100)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        z_grid = self.function(x_grid, y_grid)

        # Построение графика функции
        ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', alpha=0.5)

        sc = ax.scatter([], [], [], c='b', marker='o')

        def update_plot(frame):
            x, y = dataForGIF[frame]
            objective_values = [self.function(xi, yi) for xi, yi in zip(x, y)]
            sc._offsets3d = (x, y, objective_values)
            ax.set_title(f'Iteration {frame + 1}')
            return sc,

        ani = FuncAnimation(fig, update_plot, frames=len(dataForGIF), interval=200, repeat=False)

        plt.show()
        print(self.globalBestPos)
        print(self.globalBestScore)

    # Пример использования
a = Swarm(50, 0.1, 1, 5, 500, rastrigin, -5, 5)
a.startSwarm()
