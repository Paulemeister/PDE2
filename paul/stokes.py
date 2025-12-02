from typing import Callable
from matplotlib.collections import LineCollection
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
from mesh_ops import MeshOps
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import scipy as sp

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

    # for each point of the quadrature, calculate for each point in reference element (column vector )
    # of the gradients (row vector) inside the reference element
    shape1_ref = np.array([[phi(x, y) for phi in shape_1] for x, y in pts])
    shape2_ref = np.array([[phi(x, y) for phi in shape_2] for x, y in pts])
    dshape2_ref = np.array([[phi(x, y) for phi in dshape_2] for x, y in pts])

    p1_map = dict()

    for i, point in enumerate(np.unique(mesh.triangles)):
        p1_map[point] = i

    # Iterate over all elements
    for e in range(num_elements):
        ########################
        # calculate contributions of every element to each integral for the
        # shape functions on the vertecies of the element  (for bilinear form / matrix)
        #########################3
        elemA = np.zeros((p_2_n, p_2_n))

        invJ = mesh.calcInverseJacobianOfTriangle(e)
        J = mesh.calcJacobianOfTriangle(e)
        # calculates (B_K ^-T grad phi_n)^T = grad phi_n^T B_K^-1
        # this is done all at once by stacking the transposed gradients on top of each other
        # to a mx2 matrix resulting in stacked result row vectors in a mx2 matrix
        # where m is the amount of points in the reference element
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)
        temp = np.zeros_like(elemA)
        # Iterate over all points of the quadrature
        for i in range(N_quadr):
            # get gradients in actual element
            dphi = dshape2_ref[i] @ invJ
            # (dphi @ dphi.T) will calculate a matrix with a_ij = dphi_i * dphi_j <- dot product
            # also apply weights from quadrature
            temp += wts[i] * dphi @ dphi.T

        elemA += temp * detJ
        con6 = mesh.getNodeNumbersOfTriangle(e, order=2)

        # np.ix_(a,b) will create indexes p,q which, when indexing a matrix with it will select
        # every element in rows a that are at columns b
        # eg A([1,3,5],[6,7,8]) will select (1,6) (1,7) (3,8) (3,6) (3,7) (5,8) (5,6) (5,7) (1,8)
        # here we acess the elements corresponding to the node numbers
        A[np.ix_(con6, con6)] += elemA  # store first component of u_i at i
        A[
            np.ix_(con6 + N_u, con6 + N_u)
        ] += elemA  # store second component of u_i at i+N_u

        con3 = mesh.getNodeNumbersOfTriangle(e, order=1)
        con3_mapped = np.array([p1_map[p] for p in con3])

        elemBx = np.zeros((p_2_n, p_1_n))
        elemBy = np.zeros((p_2_n, p_1_n))
        tempx = np.zeros_like(elemBx)
        tempy = np.zeros_like(elemBy)
        # Iterate over all points of the quadrature
        for i in range(N_quadr):
            # get gradients in actual element
            phi1 = shape1_ref[i, None]  # 1 x N_p1

            # invJ 2x2
            # dshape2_ref N_quadr x N_p2 x 2
            # dphi2 = dshape2_ref[i] @ invJ

            x_dphi = dshape2_ref[i] @ (invJ @ np.array([[1], [0]]))
            y_dphi = dshape2_ref[i] @ (invJ @ np.array([[0], [1]]))
            # x_dphi = dphi2[:, 0][:, None]
            # y_dphi = dphi2[:, 1][:, None]

            # also apply weights from quadrature
            tempx += wts[i] * x_dphi @ phi1
            tempy += wts[i] * y_dphi @ phi1

        elemBx = -tempx * detJ
        elemBy = -tempy * detJ

        B[np.ix_(con6, con3_mapped)] += elemBx  # store first component of u_i at i
        B[
            np.ix_(con6 + N_u, con3_mapped)
        ] += elemBy  # store second component of u_i at i+N_u

        ###########################
        # calculate contributions of each element to the integrals of the linear functional
        # for each shape function on the vertices
        ###########################
        n1 = con3[0]
        p1 = mesh.points[n1]

        for i, node in enumerate(con6):
            temp_x = 0
            temp_y = 0
            phi = shape2_ref[:, i]
            for j in range(N_quadr):
                pt = pts[j]
                # point in actual element
                x = p1 + J @ pt
                source_val = source(x[0], x[1])
                temp_x += wts[j] * phi[j] * source_val[0]
                temp_y += wts[j] * phi[j] * source_val[1]
            f[node] += temp_x * detJ
            f[node + N_u] += temp_y * detJ
        pass

    M = sp.sparse.bmat([[A, B], [B.T, np.zeros((N_p, N_p))]], format="lil")
    f2 = np.append(f, np.zeros(N_p))
    return M, f2


def apply_bc_stokes(
    mesh: MeshOps, M: lil_matrix, f: NDArray[np.floating], param
) -> tuple[lil_matrix, NDArray[np.floating]]:

    N_u = len(
        np.unique(mesh.triangles6)
    )  # class treats midpoints as points, for p2 mesh
    N_p = len(np.unique(mesh.triangles))
    lines3 = mesh.lines3

    p1_map = dict()

    for i, point in enumerate(np.unique(mesh.triangles)):
        p1_map[point] = i

    omega2_f = param["omega2"]

    for i, (line3, tag) in enumerate(zip(lines3, mesh.lineTags)):

        if tag == 2:

            for p_ix in line3:
                point = mesh.points[p_ix]
                u = omega2_f(*point)
                f[p_ix] = u[0]
                f[p_ix + N_u] = u[1]
                # clear rows
                M[p_ix, :] = 0
                M[p_ix + N_u, :] = 0
                # set diagonal
                M[p_ix, p_ix] = 1
                M[p_ix + N_u, p_ix + N_u] = 1

        elif tag == 3:
            # Do nothing boundary / surface integral is 0
            pass
        elif tag == 4 or tag == 5:

            for p_ix in line3:
                # zero velocity on bdy
                f[p_ix] = 0
                f[p_ix + N_u] = 0
                # clear rows
                M[p_ix, :] = 0
                M[p_ix + N_u, :] = 0
                # set diagonal
                M[p_ix, p_ix] = 1
                M[p_ix + N_u, p_ix + N_u] = 1

    # Impose zero pressure on first node of first triangle
    f[2 * N_u] = 0
    # clear rows
    M[2 * N_u, :] = 0
    M[:, 2 * N_u] = 0
    # set diagonal
    M[2 * N_u, 2 * N_u] = 1

    return M, f


def add_tri6_line3(mesh: MeshOps) -> None:

    extra_points = []
    points_idx = mesh.getNumberNodes() - 1
    triangles = []
    # map that maps edge to new edge midpoint in point index
    vert_p_map = {}
    points = mesh.getNodeList()
    # get all triangles
    for e in range(mesh.getNumberOfTriangles()):
        # extract all edges
        con = mesh.getNodeNumbersOfTriangle(e)
        edge1 = (con[0], con[1])
        edge2 = (con[1], con[2])
        edge3 = (con[2], con[0])

        midpoints = []
        # assemble new connectivity list for triangle6
        for edge in [edge1, edge2, edge3]:
            ep1_ix, ep2_ix = edge
            sorted_edge = tuple(sorted(edge))

            if sorted_edge in vert_p_map.keys():
                emp_idx = vert_p_map[sorted_edge]
            else:
                # add new point to point list
                points_idx += 1
                emp_idx = points_idx
                vert_p_map[sorted_edge] = emp_idx
                emp = get_midpoint(points[ep1_ix], points[ep2_ix])
                extra_points.append(emp)
            midpoints.append(emp_idx)
        triangle = np.append(con, midpoints)
        triangles.append(triangle)

    new_triangles = np.array(triangles)
    new_points = np.append(points, extra_points, axis=0)

    lines3 = []
    for line in mesh.lines:
        sorted_edge = tuple(sorted(line))
        # Assume all lines are part of an element
        mp_ix = vert_p_map[sorted_edge]
        lines3.append(np.append(line, mp_ix))

    new_lines = np.array(lines3)

    mesh.triangles6 = new_triangles
    mesh.points = new_points
    mesh.lines3 = new_lines


def get_midpoint(
    p1: NDArray[np.floating], p2: NDArray[np.floating]
) -> NDArray[np.floating]:
    return 0.5 * (p1 + p2)


def solve_stokes(meshfile: str, param: ParamDict) -> None:
    mesh: MeshOps = MeshOps(meshfile)
    p1_map = dict()

    for i, point in enumerate(np.unique(mesh.triangles)):
        p1_map[point] = i

    print_mat: bool = False

    M, g = assemble_stokes(mesh, param)
    M, g = apply_bc_stokes(mesh, M, g, param)

    if print_mat:
        print(M.toarray())
        print(g)

    N_p2 = mesh.getNumberNodes()

    mn = sp.sparse.linalg.spsolve(M.tocsr(), g)

    un = mn[: (N_p2 * 2)]
    pn = mn[(N_p2 * 2) :]
    un_x = un[:N_p2]
    un_y = un[N_p2:]

    if print_mat:
        print(un)

    # visualize on values on the nodes
    # ( accurate for S1 and S2 approximate space)

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    (ax1, ax2), (ax3, ax4) = ax

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]

    # display linear interpolation between between approximate solution on nodes,
    # meaning vertexes with P1 or vertices and midpoints on P2
    tri_p2 = split_6triangles3(mesh)
    tri_p1 = mesh.triangles
    # ax.plot_trisurf(x, y, z, triangles=triangles, cmap="viridis")
    m1 = ax1.tripcolor(
        x,
        y,
        un_x,
        triangles=tri_p2,
        cmap="viridis",
        edgecolor="black",
        shading="flat",
        linewidth=0.2,
    )
    m2 = ax2.tripcolor(
        x,
        y,
        un_y,
        triangles=tri_p2,
        cmap="viridis",
        edgecolor="black",
        shading="flat",
        linewidth=0.2,
    )

    # make new vector for p that has len of points list,
    # that can be indexed by the global point index

    p_ixs = np.unique(mesh.triangles)

    pn_help = np.zeros_like(un_x)

    for p_ix in p_ixs:
        loc_p_ix = p1_map[p_ix]
        pn_help[p_ix] = pn[loc_p_ix]

    m3 = ax3.tripcolor(
        x,
        y,
        pn_help,
        triangles=tri_p1,
        cmap="viridis",
        edgecolor="black",
        shading="flat",
        linewidth=0.2,
    )

    ax4.quiver(x, y, un_x, un_y, angles="xy", scale=10.0)

    ax1.set_title("$u_1$")
    ax2.set_title("$u_2$")
    ax3.set_title("$p$")
    ax4.set_title("$u$")

    segments = np.array([[mesh.points[i], mesh.points[j]] for i, j in mesh.lines])

    cmap = matplotlib.colormaps.get_cmap("tab10")  # 10 indexed colors
    cols = [cmap(i) for i in mesh.lineTags]

    for a in [ax1, ax2, ax3, ax4]:
        a.set_xlabel("x")
        a.set_ylabel("y")
        lc = LineCollection(segments, colors=cols)  # type: ignore
        a.add_collection(lc)

    fig.colorbar(m1, ax=[ax1, ax2, ax3, ax4])

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
    source=lambda x, y: np.array([0, 0]),
    omega2=lambda x, y: np.array([-(y - 1) * (y + 1), 0]),
    # omega2=lambda x, y: np.array([y * (y - 1) * (y + 1), 0]),
)
solve_stokes("mesh/unitSquareStokes.msh", param_stokes)
