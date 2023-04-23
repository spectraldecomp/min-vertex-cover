import math
import random


class simulated_annealing:


    def __init__(self, graph_file_path, T=100, alpha=0.99, max_iter=10000):
        self.graph = self.import_graph(graph_file_path)
        self.T = T
        self.alpha = alpha
        self.max_iter = max_iter
        self.vertex_cover = self.generate_vertex_cover()


    '''IMPORT GRAPH'''

    # Read graph from file
    def import_graph(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        graph = {}
        for i, line in enumerate(lines):
            neighbors = set(map(int, line.strip().split()))
            graph[i] = neighbors
        return graph

    '''FITNESS'''

    # Calculate fitness of an individual (bitstring representation of a vertex cover)
    def fitness(self, individual):
        # Check if the individual is a valid vertex cover
        if not self.is_feasible(individual):
            return -1000
        # Individual is a valid vertex cover
        else:
           return -self.num_vertices(individual)

    def is_feasible(self, vertex_cover_):
        vertex_cover = set()
        for i, bit in enumerate(vertex_cover_):
            if bit == 1:
                vertex_cover.add(i)
                for neighbor in self.graph[i]:
                    vertex_cover.add(neighbor)
        if len(vertex_cover) == len(self.graph):
            return True
        else:
            return False

    def num_vertices(self, individual):
        num_vertices = 0
        for bit in individual:
            if bit == 1:
                num_vertices += 1
        return num_vertices

    '''GENERATE INITIAL VERTEX COVER'''
    def generate_vertex_cover(self):
        individual = []
        for i in range(len(self.graph)):
            individual.append(random.randint(0, 1))
        return individual

    def perturb(self, vertex_cover):
        # Save copy of current vertex cover
        old_vertex_cover = vertex_cover.copy()
        # Choose a random bit to flip
        vertex = random.randint(0, len(self.graph) - 1)
        # Flip the bit
        vertex_cover[vertex] = 1 - vertex_cover[vertex]

        if not self.is_feasible(vertex_cover):
            vertex_cover = old_vertex_cover
        return vertex_cover

    # Simulated annealing
    def simulated_annealing(self):

        # Initialize current vertex cover
        current_vertex_cover = self.vertex_cover
        current_fitness = self.fitness(current_vertex_cover)
        best_vertex_cover = current_vertex_cover
        best_fitness = current_fitness

        # Loop until temperature is 0
        while self.T > 25:
            print(self.T)
            # Loop for max iterations
            for i in range(self.max_iter):
                # Perturb the current vertex cover
                new_vertex_cover = self.perturb(current_vertex_cover)
                # Calculate fitness of new vertex cover
                new_fitness = self.fitness(new_vertex_cover)
                # If the new vertex cover is better, accept it
                if new_fitness > current_fitness:
                    current_vertex_cover = new_vertex_cover
                    current_fitness = new_fitness
                # If the new vertex cover is worse, accept it with probability p
                else:
                    p = self.acceptance_probability(current_fitness, new_fitness, self.T)
                    if random.random() < p:
                        current_vertex_cover = new_vertex_cover
                        current_fitness = new_fitness
                # Update best vertex cover
                if current_fitness > best_fitness:
                    best_vertex_cover = current_vertex_cover
                    best_fitness = current_fitness
            # Decrease temperature
            self.T *= self.alpha
        return best_vertex_cover

    def acceptance_probability(self, current_fitness, new_fitness, T):
        return math.exp((current_fitness - new_fitness) / T)

class main:

    sa = simulated_annealing("graphs/adj_list.txt")
    vertex_cover = sa.simulated_annealing()
    print(vertex_cover)




