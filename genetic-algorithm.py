import random
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class genetic_algorithm:

    # Initialize
    def __init__(self, graph_file_path, crossover_method, selection_method, mutation_method, num_generations=5000,
                 mutation_rate=0.05, crossover_rate=0.9, population_size=100, has_elitism=True):
        self.graph = self.import_graph(graph_file_path)
        self.num_generations = num_generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.has_elitism = has_elitism
        self.selection_method = selection_method
        self.mutation_method = mutation_method
        self.crossover_method = crossover_method

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
        # Unnecessary since all individuals are feasible by design
        if not self.is_feasible(individual):
            return -1000
        # Individual is a valid vertex cover
        else:
           return -self.num_vertices(individual)

    def is_feasible(self, individual):
        vertex_cover = set()
        for i, bit in enumerate(individual):
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

    '''GENERATE INDIVIDUAL AND POPULATION'''

    # Generate a random individual (bitstring representation of a vertex cover)
    def generate_individual(self):
        individual = []
        for i in range(len(self.graph)):
            individual.append(random.randint(0, 1))
        return individual

    # Generate a random population
    def generate_population(self):
        population = []
        for i in range(self.population_size):
            individual = self.generate_individual()
            # Generate only feasible individuals
            while not self.is_feasible(individual):
                individual = self.generate_individual()
            population.append(individual)
        return population

    '''SELECTION'''

    def select_parent_pool(self, population):
        parent_pool = []
        for i in range(self.population_size):
            selected_individual = self.selection(population)
            while not self.is_feasible(selected_individual):
                selected_individual = self.selection(population)
            parent_pool.append(selected_individual)

        counts = Counter(tuple(individual) for individual in population)
        most_common = counts.most_common(1)[0][0]
        return parent_pool

    # Selection (tournament and roulette)
    def selection(self, population):
        if self.selection_method == "tournament":
            return self.tournament_selection(population)
        elif self.selection_method == "roulette":
            return self.roulette_selection(population)
        else:
            raise ValueError("Invalid selection method")

    # Tournament selection
    def tournament_selection(self, population):
        tournament_size = 2
        # Randomly select two individuals from the population
        tournament = random.sample(population, tournament_size)

        # Choose the individual with higher fitness with probability tournament_probability.
        # Else, choose the individual with lower fitness.
        tournament_probability = 0.75
        if random.random() < tournament_probability:
            return max(tournament, key=self.fitness)
        else:
            return min(tournament, key=self.fitness)

    # Roulette selection
    def roulette_selection(self, population):
        fitness_scores = [self.fitness(individual) for individual in population]
        total_fitness = sum(fitness_scores)
        probabilities = [fitness_score / total_fitness for fitness_score in fitness_scores]
        return random.choices(population, weights=probabilities)[0]

    '''CROSSOVER'''

    def generate_children(self, parent_pool):
        children = []
        for i in range(self.population_size // 2):
            parents = random.sample(parent_pool, 2)
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parents[0], parents[1])
                if not self.is_feasible(child1):
                    child1 = parents[0]
                if not self.is_feasible(child2):
                    child2 = parents[1]
                children.append(child1)
                children.append(child2)
            else:
                children.append(parents[0])
                children.append(parents[1])
        return children

    # Crossover (uniform and single-point)
    def crossover(self, parent1, parent2):
        if self.crossover_method == "uniform":
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_method == "single-point":
            return self.single_point_crossover(parent1, parent2)
        else:
            raise ValueError("Invalid crossover method")

    # Uniform crossover
    def uniform_crossover(self, parent1, parent2):

        children = []
        for i in range(2):
            child = []
            for j in range(len(parent1)):
                # Randomly choose a gene from either parent
                child.append(random.choice([parent1[j], parent2[j]]))
            children.append(child)
        return children

    # Single-point crossover
    def single_point_crossover(self, parent1, parent2):
        children = []
        for i in range(2):
            child = []
            # Randomly select a crossover point
            crossover_point = random.randint(0, len(parent1) - 1)
            for j in range(len(parent1)):
                if j < crossover_point:
                    child.append(parent1[j])
                else:
                    child.append(parent2[j])
            children.append(child)
        return children

    '''MUTATION'''

    def mutate_children(self, children):
        for i in range(len(children)):
            if random.random() < self.mutation_rate:
                child_copy = children[i].copy()
                mutated_child = self.mutation(child_copy)
                if self.is_feasible(mutated_child):
                    children[i] = mutated_child
        return children

    # Mutation (bitflip and swap)
    def mutation(self, child):
        if self.mutation_method == "bitflip":
            return self.bitflip_mutation(child)
        elif self.mutation_method == "swap":
            return self.swap_mutation(child)
        else:
            raise ValueError("Invalid mutation method")

    # Bitflip mutation
    def bitflip_mutation(self, child):
        # Randomly select a bit and flip it
        bit = random.randint(0, len(child) - 1)
        child[bit] = 1 - child[bit]
        return child

    # Swap mutation
    def swap_mutation(self, child):
        # Randomly select two bits and swap them
        bits = random.sample(range(len(child)), 2)
        child[bits[0]], child[bits[1]] = child[bits[1]], child[bits[0]]
        return child

    '''ELITISM'''

    def elitism(self, population, children):
        if self.has_elitism:
            population.extend(children)
            population = sorted(population, key=self.fitness)
            return population[self.population_size:]
        else:
            raise ValueError("Elitism is always enabled")

    '''RUN'''

    def run(self):
        population = self.generate_population()
        for i in range(self.num_generations):
            parent_pool = self.select_parent_pool(population)
            children = self.generate_children(parent_pool)
            children = self.mutate_children(children)
            population = self.elitism(population, children)
        return max(population, key=self.fitness)

'''MAIN'''

class main:

    # Test the algorithm on a graph
    def test(self, graph_file_path, num_generations, mutation_rate, crossover_rate, selection_method, mutation_method, crossover_method):
        ga = genetic_algorithm(graph_file_path, crossover_method, selection_method, mutation_method, num_generations, mutation_rate, crossover_rate)
        vertex_cover = ga.run()
        return vertex_cover, ga.num_vertices(vertex_cover)

    def draw_graph(self, filename, vertex_cover=None):
        # Load the adjacency list from the file
        adj_list = {}
        with open(filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                neighbors = list(map(int, line.strip().split()))
                adj_list[i] = neighbors

        # Create the graph from the adjacency list
        G = nx.Graph(adj_list)

        # Set seed
        np.random.seed(0)
        pos = nx.spring_layout(G)

        # Draw the graph
        nx.draw(G, pos=pos, with_labels=False, node_size = 25)

        if vertex_cover:
            # Draw each vertex in the vertex cover in red
            vertex_cover_nodes = [i for i, x in enumerate(vertex_cover) if x == 1]
            nx.draw_networkx_nodes(G, pos=pos, nodelist=vertex_cover_nodes, node_color="r", node_size=75)
        plt.gca().set_title("Genetic Algorithm")
        plt.show()



if __name__ == "__main__":
    for i in range(5):
        vertex_cover, size = main().test("graphs/adj_list.txt", num_generations=5000,
                                   mutation_rate=0.05, crossover_rate=0.9,selection_method="roulette",
                                   mutation_method="bitflip", crossover_method="uniform")
        print("Vertex cover:", vertex_cover, "Size:", size)
        main().draw_graph("graphs/adj_list.txt", vertex_cover)
