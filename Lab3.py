import numpy as np
import matplotlib.pyplot as plt


def rosenbrock_func(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

class GeneticAlgorithm:
    def __init__(self, generations=50, mut_chance=0.8, survive_cof=0.8, pop_number=100):
        self.population = dict()
        self.mut_chance = mut_chance
        self.survive_cof = survive_cof
        self.generations = generations
        self.pop_number = pop_number

    def generate_start_population(self, x, y):
        for i in range(self.pop_number):
            po_x = np.random.uniform(-x, x)
            po_y = np.random.uniform(-y, y)
            self.population[i] = [po_x, po_y, rosenbrock_func(po_x, po_y)]

    def get_best_individual(self):
        return min(self.population.items(), key=lambda item: item[1][2])

    def select(self):
        sorted_pop = dict(sorted(self.population.items(), key=lambda item: item[1][2], reverse=True))

        cof = int(self.pop_number * (1 - self.survive_cof))
        parents1 = list(sorted_pop.items())[cof: cof * 2]
        parents2 = list(sorted_pop.items())[self.pop_number - cof: self.pop_number]

        i = 0
        for pop in sorted_pop.values():
            if np.random.random() > 0.5:
                pop[0] = parents1[i][1][0]
                pop[1] = parents2[i][1][1]
                pop[2] = rosenbrock_func(parents1[i][1][0], parents2[i][1][1])
            else:
                pop[0] = parents2[i][1][0]
                pop[1] = parents1[i][1][1]
                pop[2] = rosenbrock_func(parents2[i][1][0], parents1[i][1][1])
            i += 1
            if i >= cof:
                break

        self.population = sorted_pop

    def mutation(self, cur_gen):
        for pop in self.population.values():
            if np.random.random() < self.mut_chance:
                pop[0] += (np.random.random() - 0.5) * ((self.generations - cur_gen) / self.generations)
            if np.random.random() < self.mut_chance:
                pop[1] += (np.random.random() - 0.5) * ((self.generations - cur_gen) / self.generations)
            pop[2] = rosenbrock_func(pop[0], pop[1])

def visualize_evolution(genetic, X, Y, Z, ax, pop_number, iter_number):
    for i in range(iter_number):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

        for j in range(pop_number):
            ax.scatter(genetic.population[j][0], genetic.population[j][1], genetic.population[j][2], c="red", s=1, marker="s")

        best_individual = genetic.get_best_individual()
        ax.scatter(best_individual[1][0], best_individual[1][1], best_individual[1][2], c="blue")

        plt.draw()
        plt.pause(0.1)

        genetic.select()
        genetic.mutation(i)
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    for j in range(pop_number):
        ax.scatter(genetic.population[j][0], genetic.population[j][1], genetic.population[j][2], c="red", s=1, marker="s")
    best_individual = genetic.get_best_individual()
    ax.scatter(best_individual[1][0], best_individual[1][1], best_individual[1][2], c="blue")
    plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock_func(X, Y)

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

generations = 20
mut_chance = 0.8
survive_cof = 0.8
pop_number = 100
iter_number = 20

genetic = GeneticAlgorithm(generations, mut_chance, survive_cof, pop_number)
genetic.generate_start_population(5, 5)

visualize_evolution(genetic, X, Y, Z, ax, pop_number, iter_number)
