import json
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load the data
with open('graphify-out/graph.json', 'r') as f:
    data = json.load(f)

# 2. Safety Check: Determine the correct keys
# NetworkX often uses "links" but Graphify might output "edges"
nodes_key = "nodes" if "nodes" in data else "nodes" # usually consistent
edges_key = "links" if "links" in data else "edges"

# 3. Create the graph object with explicit keys
try:
    G = nx.node_link_graph(data, attrs={'source': 'source', 'target': 'target', 'name': 'id', 'key': 'key', 'link': edges_key})
except TypeError:
    # For older NetworkX versions or different schemas
    G = nx.node_link_graph(data, edges=edges_key)

# 4. Draw it
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, edge_color="gray", alpha=0.5)

plt.title("RAG Implementation V2 - Knowledge Graph")
plt.show()