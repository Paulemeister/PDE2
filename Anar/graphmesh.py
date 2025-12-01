import meshio
import networkx as nx
import matplotlib.pyplot as plt

mesh = meshio.read("unitSquare2_P2.msh")
print("Cells in mesh:", mesh.cells_dict.keys())

if "triangle" in mesh.cells_dict:
    cells = mesh.cells_dict["triangle"]
    print("P1 mesh detected")
elif "triangle6" in mesh.cells_dict:
    cells = mesh.cells_dict["triangle6"]
    print("P2 mesh detected")
else:
    raise ValueError("Mesh file does not contain triangular elements.")

points = mesh.points[:, :2]
G = nx.Graph()
for i in range(len(points)):
    G.add_node(i)

for tri in cells:
    for i in range(len(tri)):
        for j in range(i + 1, len(tri)):
            G.add_edge(tri[i], tri[j])

pos = {i: points[i] for i in range(len(points))}
plt.figure(figsize=(6, 6))
nx.draw(G, pos, node_size=25, with_labels=False)
plt.gca().set_aspect("equal")
plt.show()
