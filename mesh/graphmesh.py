import meshio
import networkx as nx
import matplotlib.pyplot as plt

filename = "unitSquare2.msh"
mesh = meshio.read(filename)

points = mesh.points[:, :2]

if "triangle" in mesh.cells_dict:
    cells = mesh.cells_dict["triangle"]
else:
    raise ValueError("Mesh file does not contain triangular elements.")
G = nx.Graph()

# Add nodes
for i in range(len(points)):
    G.add_node(i, pos=points[i])

# Add edges (from triangles)
for tri in cells:
    i, j, k = tri
    G.add_edge(i, j)
    G.add_edge(j, k)
    G.add_edge(k, i)

pos = nx.get_node_attributes(G, "pos")

plt.figure(figsize=(6, 6))
nx.draw(
    G,
    pos,
    node_size=25,
    edge_color="black",
    width=0.5,
    with_labels=False,
)
plt.title("Mesh Connectivity Graph")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
