import networkx as nx
import matplotlib.pyplot as plt

''' Generate a random simple connected graph and save it to a file'''

# number of nodes
n = 10
# probability of an edge
p = 0.3

# Generate a simple graph until it is connected
while True:
    G = nx.erdos_renyi_graph(n, p)
    if nx.is_connected(G):
        break

# Display the graph
nx.draw(G, with_labels=True)
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