import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

''' Generate a random simple connected graph and save it to a file'''

# number of nodes
n = 15
# probability of an edge
p = 0.2

# Generate a simple graph until it is connected
while True:
    G = nx.erdos_renyi_graph(n, p)
    if nx.is_connected(G):
        break

# Set seed
np.random.seed(0)
pos = nx.spring_layout(G)

# Display the graph
nx.draw(G, pos=pos, with_labels=True)
plt.show()

# Create an adjacency list
adj_list = {}
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    adj_list[node] = neighbors

# Save the graph and adjacency list to a file
with open("graphs/adj_list.txt", "w") as f:
    for node, neighbors in adj_list.items():
        f.write(" ".join(map(str, neighbors)) + "\n")