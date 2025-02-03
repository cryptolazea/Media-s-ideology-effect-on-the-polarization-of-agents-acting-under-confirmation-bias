import networkx as nx
import matplotlib.pyplot as plt

# Create directed graphs
G1 = nx.DiGraph()
G2 = nx.DiGraph()

# Add nodes with initial beliefs
initial_beliefs = {
    'A': 0.2,
    'B': 0.5,
    'C': 0.75,
    'D': 0.9
}

for node, belief in initial_beliefs.items():
    G1.add_node(node, belief=belief)
    G2.add_node(node, belief=belief)

# Add edges with weights for the first network T
edges_T = [
    ('A', 'B', 0.55), ('B', 'A', 0.2),
    ('A', 'C', 0.25),
    ('A', 'D', 0.7),
    ('B', 'C', 0.8),
    ('C', 'B', 0.7),
    ('C', 'D', 0.3),
    ('D', 'A', 0.3),
    ('D', 'C', 0.2),
    ('D', 'D', 0.9)
]

for u, v, w in edges_T:
    G1.add_edge(u, v, weight=w)

# Add edges with weights for the second network T*
edges_T_star = [
    ('A', 'A', 0.45),
    ('A', 'B', 0.55),
    ('B', 'C', 0.8),
    ('C', 'B', 0.7),
    ('C', 'D', 0.3),
    ('D', 'A', 0.7),
    ('D', 'D', 0.9)
]

for u, v, w in edges_T_star:
    G2.add_edge(u, v, weight=w)

# Define positions of nodes
pos = {
    'A': (0, 1),
    'B': (1, 1),
    'C': (1, 0),
    'D': (0, 0)
}

# Draw the first network
plt.figure(figsize=(14, 6))

plt.subplot(121)
nx.draw(G1, pos, with_labels=True, node_size=3000, node_color='white', edgecolors='black', font_size=15)
nx.draw_networkx_edge_labels(G1, pos, edge_labels={(u, v): f'{d["weight"]}' for u, v, d in G1.edges(data=True)})

for u, v in G1.edges():
    if (u, v) in G1.edges and (v, u) in G1.edges:
        nx.draw_networkx_edges(G1, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad=0.2', arrowstyle='-|>', edge_color='black')
    else:
        nx.draw_networkx_edges(G1, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad=0.0', arrowstyle='-|>', edge_color='black')

for node, (x, y) in pos.items():
    plt.text(x-0.2, y+0.1, f'$x_{{{node}0}} = {initial_beliefs[node]}$', fontsize=12, ha='center')

plt.title("(a) Network $T$")
plt.text(0.5, -0.2, "$s = (0.353, 0.328, 0.192, 0.128)'$", fontsize=12, ha='center')

# Draw the second network
plt.subplot(122)
nx.draw(G2, pos, with_labels=True, node_size=3000, node_color='white', edgecolors='black', font_size=15)
nx.draw_networkx_edge_labels(G2, pos, edge_labels={(u, v): f'{d["weight"]}' for u, v, d in G2.edges(data=True)})

for u, v in G2.edges():
    if (u, v) in G2.edges and (v, u) in G2.edges:
        nx.draw_networkx_edges(G2, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad=0.2', arrowstyle='-|>', edge_color='black')
    else:
        nx.draw_networkx_edges(G2, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad=0.0', arrowstyle='-|>', edge_color='black')

for node, (x, y) in pos.items():
    plt.text(x-0.2, y+0.1, f'$x_{{{node}0}} = {initial_beliefs[node]}$', fontsize=12, ha='center')

plt.title("(b) Network $T^*$")
plt.text(0.5, -0.2, "$s^* = (0.481, 0.330, 0.094, 0.094)'$", fontsize=12, ha='center')

plt.tight_layout()
plt.show()