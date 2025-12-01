import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def load_mesh(filename="unitSquareStokes.msh"):
    msh = meshio.read(filename) 
    points = msh.points[:, :2]

    if "triangle6" in msh.cells_dict:
        triangles = msh.cells_dict["triangle6"]
        is_p2 = True 
    elif "triangle" in msh.cells_dict:
        triangles = msh.cells_dict["triangle"]
        is_p2 = False
    else:
        raise ValueError("No triangular elements!") 

    print(f"Mesh: {len(points)} nodes, {len(triangles)} P{'2' if is_p2 else '1'} elements")

    x, y = points[:, 0], points[:, 1]
    r = np.sqrt(x**2 + y**2)
    tol = 1e-6

    inflow  = np.abs(x + 1.0) < tol
    top     = np.abs(y - 1.0) < tol
    bottom  = np.abs(y + 1.0) < tol

    radii = np.unique(np.round(r[(r > 0.02) & (r < 0.5)], decimals=4))
    if len(radii) > 0:
        cyl_r = radii[0]
        cylinder = np.abs(r - cyl_r) < 5e-3
        print(f"Detected cylinder radius ≈ {cyl_r:.5f}")
    else:
        cyl_r = 0.2
        cylinder = np.abs(r - cyl_r) < tol
        print("Cylinder radius fallback to 0.2")

    wall = cylinder | top | bottom
    print(f"Inflow nodes: {inflow.sum()}, No-slip nodes (walls+cylinder): {wall.sum()}")

    return points, triangles, inflow, wall, is_p2

def get_P2_basis_and_grad(xi, eta):
    l1 = 1 - xi - eta
    l2 = xi
    l3 = eta

    phi = np.array([
        l1*(2*l1 - 1), l2*(2*l2 - 1), l3*(2*l3 - 1),
        4*l1*l2, 4*l2*l3, 4*l3*l1
    ])

    dphi_dxi  = np.array([4*xi + 4*eta - 3, 4*xi - 1, 0,          4 - 8*xi - 4*eta, 4*eta,     -4*eta])
    dphi_deta = np.array([4*xi + 4*eta - 3, 0,         4*eta - 1, -4*xi,           4*xi,      4 - 4*xi - 8*eta])

    return phi, dphi_dxi, dphi_deta

def get_P1_basis(xi, eta):
    return np.array([1 - xi - eta, xi, eta])

def gauss_points_triangle():
    gp = np.array([
        [1/3,       1/3],
        [0.059715871789770, 0.470142064105115],
        [0.470142064105115, 0.059715871789770],
        [0.470142064105115, 0.470142064105115],
        [0.797426985353087, 0.101286507323456],
        [0.101286507323456, 0.797426985353087],
        [0.101286507323456, 0.101286507323456]
    ])
    gw = np.array([0.225,
                   0.132394152788506, 0.132394152788506, 0.132394152788506,
                   0.125939180544827, 0.125939180544827, 0.125939180544827])
    return gp, gw

def assemble_stokes_system(points, triangles, is_p2):
    n_nodes = len(points)
    p1_nodes = np.unique(triangles[:, :3])
    n_p1 = len(p1_nodes)
    n_dofs = 2*n_nodes + n_p1

    p1_map = np.zeros(n_nodes, dtype=int)
    p1_map[p1_nodes] = np.arange(n_p1)

    A = lil_matrix((n_dofs, n_dofs))
    b = np.zeros(n_dofs)

    gp, gw = gauss_points_triangle()
    print(f"Assembling {n_dofs} DOFs ...")

    for elem in triangles:
        if not is_p2:
            continue
        nodes = elem

        x_e = points[nodes, 0]
        y_e = points[nodes, 1]

        x21, y21 = x_e[1] - x_e[0], y_e[1] - y_e[0]
        x31, y31 = x_e[2] - x_e[0], y_e[2] - y_e[0]
        detJ = x21*y31 - x31*y21
        area = abs(detJ)/2
        if area < 1e-15:
            continue

        invJ = np.array([[ y31, -y21],
                         [-x31,  x21]]) / detJ

        Kloc = np.zeros((6,6))
        Bloc = np.zeros((3,12))

        for xi, eta, w in zip(gp[:,0], gp[:,1], gw):
            _, dphi_xi, dphi_eta = get_P2_basis_and_grad(xi, eta)
            phi_p1 = get_P1_basis(xi, eta)
            dphi = invJ @ np.vstack([dphi_xi, dphi_eta])
            Kloc += w * area * (dphi.T @ dphi)
            for i in range(3):
                for j in range(6):
                    Bloc[i, j]    += w * area * phi_p1[i] * dphi[0, j]
                    Bloc[i, j+6]  += w * area * phi_p1[i] * dphi[1, j]

        for i in range(6):
            for j in range(6):
                A[nodes[i], nodes[j]]                     += Kloc[i,j]
                A[n_nodes + nodes[i], n_nodes + nodes[j]] += Kloc[i,j]

        for i in range(3):
            p_idx = 2*n_nodes + p1_map[nodes[i]]
            for j in range(6):
                A[nodes[j],           p_idx] += Bloc[i, j]
                A[n_nodes + nodes[j], p_idx] += Bloc[i, j+6]
                A[p_idx, nodes[j]]           += Bloc[i, j]
                A[p_idx, n_nodes + nodes[j]] += Bloc[i, j+6]

    return A, b, n_nodes, n_p1, p1_map, p1_nodes

def apply_boundary_conditions(A, b, points, inflow, wall, n_nodes):
    x, y = points[:, 0], points[:, 1]

    for i in np.where(inflow)[0]:
        u_bc = y[i]**2 - 1
        A[i, :] = 0;  A[i, i] = 1;  b[i] = u_bc
        A[n_nodes+i, :] = 0; A[n_nodes+i, n_nodes+i] = 1; b[n_nodes+i] = 0

    for i in np.where(wall)[0]:
        A[i, :] = 0; A[i, i] = 1; b[i] = 0
        A[n_nodes+i, :] = 0; A[n_nodes+i, n_nodes+i] = 1; b[n_nodes+i] = 0

    A[2*n_nodes, :] = 0
    A[2*n_nodes, 2*n_nodes] = 1
    b[2*n_nodes] = 0

    return A, b

def solve_stokes(mesh_file="unitSquareStokes.msh"):
    points, triangles, inflow, wall, is_p2 = load_mesh(mesh_file)
    A, b, n_nodes, _, _, _ = assemble_stokes_system(points, triangles, is_p2)
    A, b = apply_boundary_conditions(A, b, points, inflow, wall, n_nodes)
    sol = spsolve(csr_matrix(A), b)

    u1 = sol[:n_nodes]
    u2 = sol[n_nodes:2*n_nodes]
    p  = sol[2*n_nodes:]

    p1_nodes = np.unique(triangles[:, :3])
    p_full = np.zeros(n_nodes)
    p_full[p1_nodes] = p

    print(f"u₁ ∈ [{u1.min():.4f}, {u1.max():.4f}]  ← negative = recirculation!")
    print(f"u₂ ∈ [{u2.min():.4f}, {u2.max():.4f}]")
    print(f"Max |u| = {np.sqrt(u1**2+u2**2).max():.4f}")

    plot_solution(points, triangles, u1, u2, p_full)

def plot_solution(points, triangles, u1, u2, p):
    x, y = points[:, 0], points[:, 1]
    tri_plot = triangles[:, :3] if triangles.shape[1] == 6 else triangles
    vel_mag = np.sqrt(u1**2 + u2**2)

    # Plot 1: u1
    fig, ax = plt.subplots(figsize=(8, 8))
    u1_tri = np.mean(u1[tri_plot], axis=1)
    c1 = ax.tripcolor(x, y, tri_plot, facecolors=u1_tri, shading='flat', cmap='viridis',
                  edgecolor='black', linewidth=0.5, alpha=0.9)
    cbar = plt.colorbar(c1, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('u₁', labelpad=15, fontsize=12)
    ax.set_aspect('equal')
    ax.set_title('Velocity Component u₁ (Heatmap)\nu₁ = (y-1)(y+1)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig('u1_velocity.png', dpi=300, bbox_inches='tight')
    print("Saved: u1_velocity.png")
    plt.close()

    # Plot 2: u2
    fig, ax = plt.subplots(figsize=(8, 8))
    u2_tri = np.mean(u2[tri_plot], axis=1)
    c2 = ax.tripcolor(x, y, tri_plot, facecolors=u2_tri, shading='flat', cmap='viridis',
                  edgecolor='black', linewidth=0.5, alpha=0.9)
    cbar = plt.colorbar(c2, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('u₂', labelpad=15, fontsize=12)
    ax.set_aspect('equal')
    ax.set_title('Velocity Component u₂ (Heatmap)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig('u2_velocity.png', dpi=300, bbox_inches='tight')
    print("Saved: u2_velocity.png")
    plt.close()

    # Plot 3: Pressure
    fig, ax = plt.subplots(figsize=(8, 8))
    p_tri = np.mean(p[tri_plot], axis=1)
    c3 = ax.tripcolor(x, y, tri_plot, facecolors=p_tri, shading='flat', cmap='viridis',
                  edgecolor='black', linewidth=0.5, alpha=0.9)

    cbar = plt.colorbar(c3, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('p', labelpad=15, fontsize=12)
    ax.set_aspect('equal')
    ax.set_title('Pressure Field (Heatmap)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig('pressure.png', dpi=300, bbox_inches='tight')
    print("Saved: pressure.png")
    plt.close()

    # Plot 4: Velocity magnitude
    fig, ax = plt.subplots(figsize=(8, 8))
    vel_tri = np.mean(vel_mag[tri_plot], axis=1)
    c4 = ax.tripcolor(x, y, tri_plot, facecolors=vel_tri, shading='flat', cmap='viridis',
                  edgecolor='black', linewidth=0.5, alpha=0.9)

    cbar = plt.colorbar(c4, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('|u|', labelpad=15, fontsize=12)
    ax.set_aspect('equal')
    ax.set_title('Velocity Magnitude (Heatmap)', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig('velocity_magnitude.png', dpi=300, bbox_inches='tight')
    print("Saved: velocity_magnitude.png")
    plt.close()

if __name__ == "__main__":
    solve_stokes(mesh_file="unitSquareStokes.msh")