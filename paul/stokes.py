import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import mesh_ops
from mesh_ops import MeshOps

from typing import Callable
from matplotlib.collections import LineCollection # type: ignore
import matplotlib.lines # type: ignore
from meshio import Mesh # type: ignore
from mpl_toolkits.mplot3d.art3d import Line3DCollection # type: ignore
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix # type: ignore
#from mesh_ops import MeshOps
import matplotlib # type: ignore

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt # type: ignore
import scipy as sp # type: ignore

ParamDict = dict[
    str, Callable[[np.floating, np.floating], NDArray[np.floating]] | int | np.floating
]


def assemble_stokes(
    mesh: MeshOps, param: ParamDict
) -> tuple[lil_matrix, NDArray[np.floating]]:

    num_elements = mesh.getNumberOfTriangles()
    # Shape functions on P1 reference element, assuming counter clockwise from (0,0)
    p1_points = [[0, 0], [1, 0], [0, 1]]
    p_1_n = len(p1_points)
    shape_1: list[Callable[[np.floating, np.floating], np.floating]] = [
        lambda x, y: 1 - x - y,
        lambda x, y: x,
        lambda x, y: y,
    ]
    dshape_1: list[Callable[[np.floating, np.floating], NDArray[np.floating]]] = [
        lambda x, y: np.array([-1, -1]),
        lambda x, y: np.array([1, 0]),
        lambda x, y: np.array([0, 1]),
    ]
    # Shape functions on P2 reference element, assuming counter clockwise from (0,0)
    p2_points: list[list[float]] = [
        [0, 0],
        [1, 0],
        [0, 1],
        [0.5, 0],
        [0.5, 0.5],
        [0, 0.5],
    ]
    p_2_n = len(p2_points)
    shape_2: list[Callable[[np.floating, np.floating], np.floating]] = [
        lambda x, y: 1 - 3 * x - 3 * y + 4 * x * y + 2 * x**2 + 2 * y**2,
        lambda x, y: -x + 2 * x**2,
        lambda x, y: -y + 2 * y**2,
        lambda x, y: 4 * x - 4 * x * y - 4 * x**2,
        lambda x, y: 4 * x * y,
        lambda x, y: 4 * y - 4 * x * y - 4 * y**2,
    ]
    dshape_2: list[Callable[[int, int], NDArray[np.floating]]] = [
        lambda x, y: np.array([4 * x + 4 * y - 3, 4 * y + 4 * x - 3]),
        lambda x, y: np.array([4 * x - 1, 0]),
        lambda x, y: np.array([0, 4 * y - 1]),
        lambda x, y: np.array([4 - 4 * y - 8 * x, -4 * x]),
        lambda x, y: np.array([4 * y, 4 * x]),
        lambda x, y: np.array([-4 * y, 4 - 4 * x - 8 * y]),
    ]

    # points for quadrature on reference element from hw5
    source: Callable[[np.floating, np.floating], NDArray[np.floating]] = param["source"]  # type: ignore

    N_u = len(
        np.unique(mesh.triangles6)
    )  # class treats midpoints as points, for p2 mesh
    N_p = len(np.unique(mesh.triangles))

    A = lil_matrix((2 * N_u, 2 * N_u))
    B = lil_matrix((2 * N_u, N_p))
    f: NDArray[np.floating] = np.zeros(2 * N_u)

    wts, pts, N_quadr = mesh.IntegrationRuleOfTriangle()

    shape1_ref = np.array([[phi(x, y) for phi in shape_1] for x, y in pts])
    dshape2_ref = np.array([[phi(x, y) for phi in dshape_2] for x, y in pts])

    p1_map = dict()

    for i, point in enumerate(np.unique(mesh.triangles)):
        p1_map[point] = i

    # Iterate over all elements
    for e in range(num_elements):

        elemA = np.zeros((p_2_n, p_2_n))

        invJ = mesh.calcInverseJacobianOfTriangle(e)
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)

        for i_func in range(p_2_n):
            for j_func in range(p_2_n):
                temp = 0
                for i_quad in range(N_quadr):
                    dphi_i = dshape2_ref[i_quad, i_func] @ invJ.T
                    dphi_j = dshape2_ref[i_quad, j_func] @ invJ.T
                    temp += wts[i_quad] * dphi_i @ dphi_j.T
                elemA[i_func, j_func] += temp * detJ

        con6 = mesh.getNodeNumbersOfTriangle(e, order=2)

        for i in range(p_2_n):
            for j in range(p_2_n):
                A[con6[i], con6[j]] += elemA[i, j]
                A[con6[i] + N_u, con6[j] + N_u] += elemA[i, j]

        con3 = mesh.getNodeNumbersOfTriangle(e, order=1)
        con3_mapped = np.array([p1_map[p] for p in con3])

        elemBx = np.zeros((p_2_n, p_1_n))
        elemBy = np.zeros((p_2_n, p_1_n))
        # Iterate over all points of the quadrature

        for i_func in range(p_2_n):
            for j_func in range(p_1_n):
                tempx = 0
                tempy = 0
                for i_quad in range(N_quadr):
                    # get gradients in actual element
                    phi1 = shape1_ref[i_quad, j_func]  # 1 x N_p1

                    dphi2 = dshape2_ref[i_quad, i_func] @ invJ.T
                    x_dphi2 = dphi2[0]
                    y_dphi2 = dphi2[1]

                    # also apply weights from quadrature
                    tempx += wts[i_quad] * x_dphi2 * phi1
                    tempy += wts[i_quad] * y_dphi2 * phi1
                elemBx[i_func, j_func] += detJ /2 * tempx
                elemBy[i_func, j_func] += detJ /2 * tempy

        for i in range(p_2_n):
            for j in range(p_1_n):
                B[con6[i], con3_mapped[j]] += elemBx[i, j]
                B[con6[i] + N_u, con3_mapped[j]] += elemBy[i, j]
        # B[np.ix_(con6, con3_mapped)] += elemBx  # store first component of u_i at i
        # B[
        #     np.ix_(con6 + N_u, con3_mapped)
        # ] += elemBy  # store second component of u_i at i+N_u

        ###########################
        # calculate contributions of each element to the integrals of the linear functional
        # for each shape function on the vertices
        ###########################
        n1 = con3[0]
        p1 = mesh.points[n1]

        # for i, node in enumerate(con6):
        #     temp_x = 0
        #     temp_y = 0
        #     phi = shape2_ref[:, i]
        #     for j in range(N_quadr):
        #         pt = pts[j]
        #         # point in actual element
        #         x = p1 + J @ pt
        #         source_val = source(x[0], x[1])
        #         temp_x += wts[j] * phi[j] * source_val[0]
        #         temp_y += wts[j] * phi[j] * source_val[1]
        #     f[node] += temp_x * detJ
        #     f[node + N_u] += temp_y * detJ
        pass

    M = sp.sparse.block_array([[A, B], [B.T, np.zeros((N_p, N_p))]], format="lil")
    g = np.append(f, np.zeros(N_p))

    return M, g


def apply_bc_stokes(
    mesh: MeshOps, M: lil_matrix, f: NDArray[np.floating], param
) -> tuple[lil_matrix, NDArray[np.floating]]:

    N_u = len(
        np.unique(mesh.triangles6)
    )  # class treats midpoints as points, for p2 mesh
    N_p = len(np.unique(mesh.triangles))
    lines3 = mesh.lines3

    omega2_f = lambda x, y: np.array([(y - 1) * (y + 1), 0])

    test = 0
    for i, (line3, tag) in enumerate(zip(lines3, mesh.lineTags)):

        if tag == 2:

            for p_ix in line3:
                point = mesh.points[p_ix]
                u = omega2_f(*point)
                f[p_ix] = u[0]
                f[p_ix + N_u] = u[1]
                # clear rows
                M[p_ix, :] = 0
                M[:, p_ix] = 0
                M[p_ix + N_u, :] = 0
                M[:, p_ix + N_u] = 0
                # set diagonal
                M[p_ix, p_ix] = 1
                M[p_ix + N_u, p_ix + N_u] = 1
                test += 1

        elif tag == 3:
            for p_ix in line3:
                test += 1
            # Do nothing boundary / surface integral is 0
            pass
        elif (tag == 4) or (tag == 5):

            for p_ix in line3:
                # zero velocity on bdy
                point = mesh.points[p_ix]

                f[p_ix] = 0
                f[p_ix + N_u] = 0
                # clear rows
                M[p_ix, :] = 0
                M[:, p_ix] = 0
                M[p_ix + N_u, :] = 0
                M[:, p_ix + N_u] = 0
                # set diagonal
                M[p_ix, p_ix] = 1
                M[p_ix + N_u, p_ix + N_u] = 1
                test += 1

    # Impose zero pressure on first node of first triangle
    # f[2 * N_u] = 0
    # # clear rows
    # M[2 * N_u, :] = 0
    # M[:, 2 * N_u] = 0
    # # set diagonal
    # M[2 * N_u, 2 * N_u] = 1

    print(test)

    return M, f


def get_midpoint(
    p1: NDArray[np.floating], p2: NDArray[np.floating]
) -> NDArray[np.floating]:
    return 0.5 * (p1 + p2)


def check_mesh(mesh: MeshOps):
    n = 0
    for i, e in enumerate(mesh.triangles6):
        J = mesh.calcJacobianDeterminantOfTriangle(i)
        if J < 0:
            n += 1
    if n > 0:
        print(f"{n} Triangles inverted!")


def preprocess(mesh: MeshOps):
    check_mesh(mesh)
    if np.min(mesh.triangles6[:, 3:]) > np.max(mesh.triangles):
        # the triangle list is not sorted
        pass


def calc_p1_map(mesh: MeshOps) -> dict[int, int]:

    p1_map = dict()

    for i, point in enumerate(np.unique(mesh.points)):
        p1_map[point] = i

    return p1_map


def sanity_checks(mesh: MeshOps):

    # All triangles used?
    if len(np.unique(mesh.triangles6)) != len(mesh.points):
        print("not all points in mesh used in triangle!")

    # All are point indeces skipped?
    if np.max(np.unique(mesh.triangles6)) != len(mesh.points) - 1:
        print("not all indeces in mesh are in points list")

    # Are triangles inverted?
    num_inverted = 0
    for i, con in enumerate(mesh.triangles):
        detJ = np.linalg.det(mesh.calcJacobianOfTriangle(i))
        if detJ < 0:
            num_inverted += 1
    if num_inverted != 0:
        print(f"{num_inverted} triangles inverted!")

    # where are the midpoints?
    p1_points = np.unique(mesh.triangles)
    p2_points = np.unique(mesh.triangles6)
    if np.max(p1_points) > len(p1_points) - 1:
        print("p1 points are not at the start of the point list")


def solve_stokes(meshfile: str, param: ParamDict) -> None:
    meshfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mesh", "unitSquareStokes.msh")
    mesh = MeshOps(meshfile)
    #mesh: MeshOps = MeshOps(meshfile)
    # sanity_checks(mesh)

    # preprocess(mesh)

    print_mat: bool = False

    N_u = len(
        np.unique(mesh.triangles6)
    )  # class treats midpoints as points, for p2 mesh
    N_p = len(np.unique(mesh.triangles))

    M, g = assemble_stokes(mesh, param)
    if False:
        fig, ax1 = plt.subplots(nrows=1)
        ax1.spy(M, markersize=1)
        # ax2.imshow(g[:, None])
        plt.show()
    # M, g = apply_bc_stokes(mesh, M, g, param)

    if print_mat:
        print(M.toarray())
        print(g)

    if False:
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.spy(M, markersize=1)
        ax2.imshow(g[:, None])
        plt.show()

    mn = sp.sparse.linalg.spsolve(M.tocsr(), g)

    un = mn[: (N_u * 2)]
    pn = mn[(N_u * 2) :]
    un_x = un[:N_u]
    un_y = un[N_u:]

    print(
        "N_u",
        N_u,
        "mesh.points.shape[0]",
        mesh.points.shape[0],
        "un_x.shape",
        un_x.shape,
    )
    nonzero_idxs = np.where(np.abs(un_x) > 1e-12)[0]
    print(
        "Nicht-nulle Geschwindigkeit an Knoten: ",
        nonzero_idxs,
        mesh.points[nonzero_idxs],
    )

    if print_mat:
        print(un)

    # visualize on values on the nodes
    # ( accurate for S1 and S2 approximate space)

    fig, ax = plt.subplots(nrows=2)
    ax1, ax2 = ax
    # ax = fig.add_subplot(projection="3d")
    # ax1 = fig.add_subplot()
    # ax2 = fig.add_subplot()

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = un_x

    # display linear interpolation between between approximate solution on nodes,
    # meaning vertexes with P1 or vertices and midpoints on P2
    triangles3 = split_6triangles3(mesh)
    # ax.plot_trisurf(x, y, z, triangles=triangles, cmap="viridis")
    m1 = ax1.tripcolor(
        x,
        y,
        un_x,
        triangles=triangles3,
        cmap="viridis",
        edgecolor="gray",
    )
    # m2 = ax2.tripcolor(
    #     x, y, un_y, triangles=triangles3, cmap="viridis", edgecolor="gray"
    # )
    # m3 = ax1.scatter(x, y, c=un_x)
    m4 = ax2.scatter(x, y, c=un_y)

    segments = np.array([[mesh.points[i], mesh.points[j]] for i, j in mesh.lines])
    # colors = plt.colormaps.get_cmap()

    cmap = matplotlib.colormaps.get_cmap("tab10")  # 10 indexed colors
    cols = [cmap(i) for i in mesh.lineTags]
    lc = LineCollection(segments, colors=cols)  # type: ignore
    lc2 = LineCollection(segments, colors=cols)  # type: ignore
    ax1.add_collection(lc)
    ax2.add_collection(lc2)

    fig.colorbar(m1)
    fig.colorbar(m4)

    plt.show()


def split_6triangles3(mesh: MeshOps):
    new_tri = []
    for k in range(mesh.getNumberOfTriangles()):
        con = mesh.getNodeNumbersOfTriangle(k, order=2)
        p1, p2, p3, p12, p23, p31 = con
        new_tri.append([p1, p12, p31])
        new_tri.append([p2, p23, p12])
        new_tri.append([p3, p31, p23])
        new_tri.append([p12, p23, p31])
    return new_tri


param_stokes: ParamDict = dict(
    laplaceCoeff=1,
    # source=lambda x, y: np.float64(1.0),
    source=lambda x, y: np.array([0, 0]),
    dirichlet=0,
    neumann=0,
    order=1,
)
#solve_stokes("mesh/unitSquareStokes.msh", param_stokes)
meshfile = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mesh", "unitSquareStokes.msh")
solve_stokes(meshfile, param_stokes)
