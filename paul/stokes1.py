import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mesh_ops import MeshOps

from typing import Callable
from matplotlib.collections import LineCollection
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy as sp

ParamDict = dict[
    str, Callable[[np.floating, np.floating], NDArray[np.floating]] | int | np.floating
]


def assemble_stokes(
    mesh: MeshOps, param: ParamDict
) -> tuple[lil_matrix, NDArray[np.floating], dict, dict]:
    """
    Assemble the Stokes system with P2-P1 elements.
    Returns: (system matrix, rhs vector, p2_global_to_dof, p1_global_to_dof)
    """

    num_elements = mesh.getNumberOfTriangles()
    
    # Shape functions on P1 reference element
    p1_points = [[0, 0], [1, 0], [0, 1]]
    p_1_n = len(p1_points)
    shape_1: list[Callable[[np.floating, np.floating], np.floating]] = [
        lambda x, y: 1 - x - y,
        lambda x, y: x,
        lambda x, y: y,
    ]
    
    # Shape functions on P2 reference element
    p2_points: list[list[float]] = [
        [0, 0], [1, 0], [0, 1],
        [0.5, 0], [0.5, 0.5], [0, 0.5],
    ]
    p_2_n = len(p2_points)
    
    dshape_2: list[Callable[[int, int], NDArray[np.floating]]] = [
        lambda x, y: np.array([4 * x + 4 * y - 3, 4 * y + 4 * x - 3]),
        lambda x, y: np.array([4 * x - 1, 0]),
        lambda x, y: np.array([0, 4 * y - 1]),
        lambda x, y: np.array([4 - 4 * y - 8 * x, -4 * x]),
        lambda x, y: np.array([4 * y, 4 * x]),
        lambda x, y: np.array([-4 * y, 4 - 4 * x - 8 * y]),
    ]

    source: Callable[[np.floating, np.floating], NDArray[np.floating]] = param["source"]

    # Get unique nodes and create DOF mappings
    p2_all_nodes = np.unique(mesh.triangles6.flatten())
    p1_all_nodes = np.unique(mesh.triangles.flatten())
    
    N_u = len(p2_all_nodes)  # P2 nodes for velocity
    N_p = len(p1_all_nodes)   # P1 nodes for pressure

    print(f"Assembly: N_u={N_u}, N_p={N_p}, Total points={len(mesh.points)}")
    print(f"P2 nodes: {len(p2_all_nodes)} nodes, range [{p2_all_nodes.min()}, {p2_all_nodes.max()}]")
    print(f"P1 nodes: {len(p1_all_nodes)} nodes, range [{p1_all_nodes.min()}, {p1_all_nodes.max()}]")
    
    # Create mapping from global node indices to DOF indices
    p2_global_to_dof = {node_idx: dof for dof, node_idx in enumerate(p2_all_nodes)}
    p1_global_to_dof = {node_idx: dof for dof, node_idx in enumerate(p1_all_nodes)}

    A = lil_matrix((2 * N_u, 2 * N_u))
    B = lil_matrix((2 * N_u, N_p))
    f: NDArray[np.floating] = np.zeros(2 * N_u)

    wts, pts, N_quadr = mesh.IntegrationRuleOfTriangle()

    shape1_ref = np.array([[phi(x, y) for phi in shape_1] for x, y in pts])
    dshape2_ref = np.array([[phi(x, y) for phi in dshape_2] for x, y in pts])

    # Iterate over all elements
    for e in range(num_elements):

        elemA = np.zeros((p_2_n, p_2_n))

        invJ = mesh.calcInverseJacobianOfTriangle(e)
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)

        # Assemble A matrix (velocity Laplacian term)
        for i_func in range(p_2_n):
            for j_func in range(p_2_n):
                temp = 0
                for i_quad in range(N_quadr):
                    dphi_i = dshape2_ref[i_quad, i_func] @ invJ.T
                    dphi_j = dshape2_ref[i_quad, j_func] @ invJ.T
                    temp += wts[i_quad] * (dphi_i @ dphi_j)
                elemA[i_func, j_func] += temp * detJ

        con6 = mesh.getNodeNumbersOfTriangle(e, order=2)

        # Convert global node indices to DOF indices
        con6_dof = [p2_global_to_dof[node] for node in con6]
        
        # Add to global A matrix for both velocity components
        for i in range(p_2_n):
            for j in range(p_2_n):
                A[con6_dof[i], con6_dof[j]] += elemA[i, j]
                A[con6_dof[i] + N_u, con6_dof[j] + N_u] += elemA[i, j]

        # Assemble B matrix (divergence/gradient coupling term)
        con3 = mesh.getNodeNumbersOfTriangle(e, order=1)
        con3_dof = [p1_global_to_dof[node] for node in con3]

        elemBx = np.zeros((p_2_n, p_1_n))
        elemBy = np.zeros((p_2_n, p_1_n))

        for i_func in range(p_2_n):
            for j_func in range(p_1_n):
                tempx = 0
                tempy = 0
                for i_quad in range(N_quadr):
                    phi1 = shape1_ref[i_quad, j_func]
                    dphi2 = dshape2_ref[i_quad, i_func] @ invJ.T
                    x_dphi2 = dphi2[0]
                    y_dphi2 = dphi2[1]

                    tempx += wts[i_quad] * x_dphi2 * phi1
                    tempy += wts[i_quad] * y_dphi2 * phi1
                    
                # Correct weak form: -div operator
                elemBx[i_func, j_func] -= detJ * tempx
                elemBy[i_func, j_func] -= detJ * tempy

        # Add to global B matrix
        for i in range(p_2_n):
            for j in range(p_1_n):
                B[con6_dof[i], con3_dof[j]] += elemBx[i, j]
                B[con6_dof[i] + N_u, con3_dof[j]] += elemBy[i, j]

    # Assemble block system
    M = sp.sparse.block_array([[A, B], [B.T, None]], format="lil")
    g = np.append(f, np.zeros(N_p))

    return M, g, p2_global_to_dof, p1_global_to_dof


def apply_bc_stokes(
    mesh: MeshOps, M: lil_matrix, f: NDArray[np.floating], param, 
    p2_global_to_dof: dict, p1_global_to_dof: dict
) -> tuple[lil_matrix, NDArray[np.floating]]:
    """Apply boundary conditions using DOF mappings"""

    N_u = len(p2_global_to_dof)
    N_p = len(p1_global_to_dof)
    lines3 = mesh.lines3

    # Inflow boundary condition
    omega2_f = lambda x, y: np.array([(y - 1) * (y + 1), 0])

    bc_count = 0
    nodes_processed = set()  # Track which nodes we've set BCs on
    
    for i, (line3, tag) in enumerate(zip(lines3, mesh.lineTags)):

        if tag == 2:  # Inflow boundary (left, x = -1)
            print(f"Processing inflow boundary (tag {tag}), {len(line3)} nodes")
            for p_ix in line3:
                if p_ix in nodes_processed:
                    continue
                # Convert global node index to DOF index
                if p_ix not in p2_global_to_dof:
                    continue
                dof_idx = p2_global_to_dof[p_ix]
                
                point = mesh.points[p_ix]
                u = omega2_f(*point)
                f[dof_idx] = u[0]
                f[dof_idx + N_u] = u[1]
                # Clear rows and columns
                M[dof_idx, :] = 0
                M[:, dof_idx] = 0
                M[dof_idx + N_u, :] = 0
                M[:, dof_idx + N_u] = 0
                # Set diagonal
                M[dof_idx, dof_idx] = 1
                M[dof_idx + N_u, dof_idx + N_u] = 1
                nodes_processed.add(p_ix)
                bc_count += 1

        elif tag == 3:  # Outflow boundary (right, x = 1)
            # Do nothing boundary condition
            print(f"Processing outflow boundary (tag {tag}), {len(line3)} nodes - do nothing")
            pass
            
        elif (tag == 4) or (tag == 5):  # Wall boundaries (cylinder, top, bottom)
            print(f"Processing wall boundary (tag {tag}), {len(line3)} nodes")
            for p_ix in line3:
                if p_ix in nodes_processed:
                    continue
                # Convert global node index to DOF index
                if p_ix not in p2_global_to_dof:
                    continue
                dof_idx = p2_global_to_dof[p_ix]
                
                # Zero velocity on boundary
                f[dof_idx] = 0
                f[dof_idx + N_u] = 0
                # Clear rows and columns
                M[dof_idx, :] = 0
                M[:, dof_idx] = 0
                M[dof_idx + N_u, :] = 0
                M[:, dof_idx + N_u] = 0
                # Set diagonal
                M[dof_idx, dof_idx] = 1
                M[dof_idx + N_u, dof_idx + N_u] = 1
                nodes_processed.add(p_ix)
                bc_count += 1

    print(f"Applied boundary conditions to {bc_count} nodes out of {N_u} velocity nodes")
    print(f"Interior nodes (no BC): {N_u - bc_count}")
    
    # Fix pressure singularity by setting p=0 at one node
    f[2 * N_u] = 0
    M[2 * N_u, :] = 0
    M[:, 2 * N_u] = 0
    M[2 * N_u, 2 * N_u] = 1

    return M, f


def split_6triangles3(mesh: MeshOps):
    """Split P2 triangles into 4 P1 triangles for visualization"""
    new_tri = []
    for k in range(mesh.getNumberOfTriangles()):
        con = mesh.getNodeNumbersOfTriangle(k, order=2)
        p1, p2, p3, p12, p23, p31 = con
        new_tri.append([p1, p12, p31])
        new_tri.append([p2, p23, p12])
        new_tri.append([p3, p31, p23])
        new_tri.append([p12, p23, p31])
    return new_tri


def solve_stokes(meshfile: str, param: ParamDict) -> None:
    mesh = MeshOps(meshfile)

    # Assemble system (this also creates the mappings)
    M, g, p2_global_to_dof, p1_global_to_dof = assemble_stokes(mesh, param)
    
    N_u = len(p2_global_to_dof)
    N_p = len(p1_global_to_dof)
    
    print(f"Number of velocity DOFs: {2*N_u}")
    print(f"Number of pressure DOFs: {N_p}")
    print(f"Total system size: {2*N_u + N_p}")
    
    # Apply boundary conditions
    M, g = apply_bc_stokes(mesh, M, g, param, p2_global_to_dof, p1_global_to_dof)

    # Solve system
    print("Solving linear system...")
    mn = sp.sparse.linalg.spsolve(M.tocsr(), g)

    # Extract solution
    un = mn[:(N_u * 2)]
    pn = mn[(N_u * 2):]
    un_x = un[:N_u]
    un_y = un[N_u:]

    print(f"\nVelocity range: u_x ∈ [{un_x.min():.4f}, {un_x.max():.4f}]")
    print(f"                u_y ∈ [{un_y.min():.4f}, {un_y.max():.4f}]")
    print(f"Pressure range: p ∈ [{pn.min():.4f}, {pn.max():.4f}]")

    print("\n=== Boundary Condition Check ===")
    
    # Check each boundary type
    for tag_check, tag_name in [(2, "Inflow"), (4, "Cylinder"), (5, "Walls")]:
        for i, (line3, tag) in enumerate(zip(mesh.lines3, mesh.lineTags)):
            if tag == tag_check:
                p_ix_global = line3[0]  # Check first node (global index)
                if p_ix_global not in p2_global_to_dof:
                    continue
                dof_ix = p2_global_to_dof[p_ix_global]
                point = mesh.points[p_ix_global]
                if tag == 2:
                    expected_u1 = (point[1] - 1) * (point[1] + 1)
                    error = abs(un_x[dof_ix] - expected_u1)
                    print(f"{tag_name} node {p_ix_global} (DOF {dof_ix}) at ({point[0]:.3f}, {point[1]:.3f}): "
                          f"u1={un_x[dof_ix]:.6f} (expected {expected_u1:.6f}, error={error:.2e}), "
                          f"u2={un_y[dof_ix]:.6f}")
                else:
                    print(f"{tag_name} node {p_ix_global} (DOF {dof_ix}) at ({point[0]:.3f}, {point[1]:.3f}): "
                          f"u1={un_x[dof_ix]:.6e}, u2={un_y[dof_ix]:.6e}")
                break

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Create arrays indexed by ACTUAL NODE NUMBERS (not DOF indices)
    # This is the key: the array size matches mesh.points, and we index by global node number
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    
    # Velocity arrays: use the solution values directly at their node positions
    un_x_plot = np.zeros(len(mesh.points))
    un_y_plot = np.zeros(len(mesh.points))
    for global_node, dof_idx in p2_global_to_dof.items():
        un_x_plot[global_node] = un_x[dof_idx]
        un_y_plot[global_node] = un_y[dof_idx]
    
    triangles3 = split_6triangles3(mesh)

    # Plot u_x
    ax = axes[0, 0]
    m1 = ax.tripcolor(x, y, un_x_plot, triangles=triangles3, cmap="RdBu_r", shading='flat')
    ax.set_title("Velocity $u_1$ (x-component)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.colorbar(m1, ax=ax)

    # Plot u_y
    ax = axes[0, 1]
    m2 = ax.tripcolor(x, y, un_y_plot, triangles=triangles3, cmap="RdBu_r", shading='flat')
    ax.set_title("Velocity $u_2$ (y-component)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.colorbar(m2, ax=ax)

    # Plot pressure (on P1 nodes only)
    ax = axes[1, 0]
    
    # Get P1 nodes and create pressure array indexed by DOF
    p1_nodes_list = np.array(sorted(p1_global_to_dof.keys()))
    p_values = np.array([pn[p1_global_to_dof[node]] for node in p1_nodes_list])
    
    # Create local triangle connectivity
    p1_dof_to_local = {node: i for i, node in enumerate(p1_nodes_list)}
    triangles_p1_local = np.array([[p1_dof_to_local[n] for n in tri] for tri in mesh.triangles])
    
    m3 = ax.tripcolor(mesh.points[p1_nodes_list, 0], 
                      mesh.points[p1_nodes_list, 1], 
                      p_values, 
                      triangles=triangles_p1_local,
                      cmap="viridis", shading='flat')
    ax.set_title("Pressure $p$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.colorbar(m3, ax=ax)

    # Plot velocity magnitude and streamlines
    ax = axes[1, 1]
    velocity_mag = np.sqrt(un_x_plot**2 + un_y_plot**2)
    m4 = ax.tripcolor(x, y, velocity_mag, triangles=triangles3, cmap="plasma", shading='flat')
    
    x_grid = np.linspace(x.min(), x.max(), 30)
    y_grid = np.linspace(y.min(), y.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate velocity to grid
    from scipy.interpolate import LinearNDInterpolator
    p2_nodes = np.array(sorted(p2_global_to_dof.keys()))
    un_x_for_interp = np.array([un_x[p2_global_to_dof[n]] for n in p2_nodes])
    un_y_for_interp = np.array([un_y[p2_global_to_dof[n]] for n in p2_nodes])
    
    interp_ux = LinearNDInterpolator(mesh.points[p2_nodes], un_x_for_interp)
    interp_uy = LinearNDInterpolator(mesh.points[p2_nodes], un_y_for_interp)
    U_grid = interp_ux(X_grid, Y_grid)
    V_grid = interp_uy(X_grid, Y_grid)
    
    ax.streamplot(X_grid, Y_grid, U_grid, V_grid, color='white', 
                  density=1.5, linewidth=0.5, arrowsize=0.5)
    ax.set_title("Velocity Magnitude with Streamlines")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.colorbar(m4, ax=ax)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    param_stokes: ParamDict = dict(
        laplaceCoeff=1,
        source=lambda x, y: np.array([0.0, 0.0]),
        dirichlet=0,
        neumann=0,
        order=1,
    )
    
    meshfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "mesh", "unitSquareStokes.msh")
    solve_stokes(meshfile, param_stokes)