import numpy as np
import matplotlib.pyplot as plt

class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.5, q=100):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

    def initialize_pheromones(self, graph):
        return np.ones_like(graph) * 0.1

    def heuristic(self, graph):
        return 1 / (graph + 1e-10)

    def run(self, graph, start, goal):
        pheromone = self.initialize_pheromones(graph)
        heuristic = self.heuristic(graph)
        best_path = None
        best_length = float('inf')

        for _ in range(self.num_iterations):
            paths = []
            path_lengths = []

            for ant in range(self.num_ants):
                path = [start]
                current = start

                while current != goal:
                    neighbors = np.where(graph[current] > 0)[0]
                    pheromone_values = pheromone[current, neighbors] ** self.alpha
                    heuristic_values = heuristic[current, neighbors] ** self.beta
                    probabilities = pheromone_values * heuristic_values
                    probabilities /= probabilities.sum()
                    next_node = np.random.choice(neighbors, p=probabilities)
                    path.append(next_node)
                    current = next_node

                paths.append(path)
                path_length = sum(graph[path[i], path[i+1]] for i in range(len(path)-1))
                path_lengths.append(path_length)

                if path_length < best_length:
                    best_length = path_length
                    best_path = path

            pheromone *= (1 - self.rho)
            for path, length in zip(paths, path_lengths):
                delta_pheromone = self.q / length
                for i in range(len(path)-1):
                    pheromone[path[i], path[i+1]] += delta_pheromone

        return best_path

if __name__ == "__main__":
    graph = np.array([
        [0, 1, 4, 0, 0],
        [1, 0, 4, 2, 7],
        [4, 4, 0, 3, 5],
        [0, 2, 3, 0, 1],
        [0, 7, 5, 1, 0]
    ])
    aco = AntColonyOptimization(num_ants=10, num_iterations=100)
    best_path = aco.run(graph, start=0, goal=4)
    print("Best path found:", best_path)