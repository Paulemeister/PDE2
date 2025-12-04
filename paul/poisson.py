#Loads a mesh
#Constructs linear or quadratic finite elements
#Assembles stiffness matrix and load vector
#Applies Dirichlet and Neumann boundary conditions
#Solves the Poisson equation
#Visualizes the solution

from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix # type: ignore
from paul_mesh_ops import MeshOps
import matplotlib # type: ignore

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt # type: ignore
import scipy as sp # type: ignore 

ParamDict = dict[
    str, Callable[[np.floating, np.floating], np.floating] | int | np.floating
]

# Assemble the stiffness matrix and load vector for the Poisson equation
def assemble_poisson(
    mesh: MeshOps, param: ParamDict
) -> tuple[lil_matrix, NDArray[np.floating]]:

# extract parameters
    num_elements = mesh.getNumberOfTriangles()
    # Shape functions on P1 reference element, assuming counter clockwise from (0,0)
    p1_points = [[0, 0], [1, 0], [0, 1]]
    shape_1: list[Callable[[np.floating, np.floating], np.floating]] = [
        lambda x, y: 1 - x - y,
        lambda x, y: x,
        lambda x, y: y,
    ]
    # Derivatives of shape functions on P1 reference element
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
    # Shape functions on P2 reference element
    shape_2: list[Callable[[np.floating, np.floating], np.floating]] = [
        lambda x, y: 1 - 3 * x - 3 * y + 4 * x * y + 2 * x**2 + 2 * y**2,
        lambda x, y: -x + 2 * x**2,
        lambda x, y: -y + 2 * y**2,
        lambda x, y: 4 * x - 4 * x * y - 4 * x**2,
        lambda x, y: 4 * x * y,
        lambda x, y: 4 * y - 4 * x * y - 4 * y**2,
    ]
    # Derivatives of shape functions on P2 reference element
    dshape_2: list[Callable[[int, int], NDArray[np.floating]]] = [
        lambda x, y: np.array([4 * x + 4 * y - 3, 4 * y + 4 * x - 3]),
        lambda x, y: np.array([4 * x - 1, 0]),
        lambda x, y: np.array([0, 4 * y - 1]),
        lambda x, y: np.array([4 - 4 * y - 8 * x, -4 * x]),
        lambda x, y: np.array([4 * y, 4 * x]),
        lambda x, y: np.array([-4 * y, 4 - 4 * x - 8 * y]),
    ]

    # points for quadrature on reference element from hw5
    source: Callable[[np.floating, np.floating], np.floating] = param["source"]  # type: ignore
    # depending on order, set shape functions and quadrature points
    order: int = param_poisson["order"]  # type: ignore
    if order == 2:
        phi = shape_2
        # derivatives of shape functions
        dphi = dshape_2
        # number of local nodes per element
        local_n = 6
        N = len(mesh.points)
        # quadrature points and weights
        wts, pts, N_quadr = mesh.IntegrationRuleOfTriangle()

    else:
        phi = shape_1
        dphi = dshape_1
        local_n = 3
        N = mesh.getNumberNodes()  # only works when not not updating mesh.nbNod
        pts = np.array([[1 / 6, 1 / 6], [4 / 6, 1 / 6], [1 / 6, 4 / 6]])
        wts = np.array([1 / 6] * 3)
        N_quadr = 3

    A = lil_matrix((N, N))
    f: NDArray[np.floating] = np.zeros(N)

    # for each point of the quadrature, calculate for each point in reference element (column vector )
    # of the gradients (row vector) inside the reference element
    dphi_ref = np.array([[dphi[i](x, y) for i in range(local_n)] for x, y in pts])

    # Iterate over all elements
    for e in range(num_elements):
        ########################
        # calculate contributions of every element to each integral for the
        # shape functions on the vertecies of the element  (for bilinear form / matrix)
        #########################3
        elemA = np.zeros((local_n, local_n))

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
            dphi = dphi_ref[i] @ invJ
            # (dphi @ dphi.T) will calculate a matrix with a_ij = phi_i * phi_j <- dot product
            # also apply weights from quadrature
            temp += wts[i] * dphi @ dphi.T

        elemA += temp * detJ
        con = mesh.getNodeNumbersOfTriangle(e, order=order)
        # np.ix_(a,b) will create indexes p,q which, when indexing a matrix with it will select
        # every element in rows a that are at columns b
        # eg A([1,3,5],[6,7,8]) will select (1,6) (1,7) (3,8) (3,6) (3,7) (5,8) (5,6) (5,7) (1,8)
        # here we acess the elements corresponding to the node numbers
        A[np.ix_(con, con)] += elemA

        ###########################
        # calculate contributions of each element to the integrals of the linear functional
        # for each shape function on the vertices
        ###########################
        n1 = con[0]
        p1 = mesh.points[n1]

        for i, node in enumerate(con):
            temp = 0
            for j in range(N_quadr):
                pt = pts[j]
                # point in actual element
                x = p1 + J @ pt
                temp += wts[j] * phi[i](pt[0], pt[1]) * source(x[0], x[1])
            f[node] += temp * detJ

    return A, f


def apply_bc_poisson(
    mesh: MeshOps, A: lil_matrix, f: NDArray[np.floating], param
) -> tuple[lil_matrix, NDArray[np.floating]]:

    dirichlet = param["dirichlet"]
    neumann = param["neumann"]

    phi_1 = [lambda x: x, lambda x: 1 - x]
    pts_1 = [0, 1]
    phi_2 = [
        lambda x: 0.5 * x**2 - 0.5 * x,
        lambda x: 0.5 * x**2 + 0.5 * x,
        lambda x: -1 * x**2 + 1,
    ]
    pts_2 = [-1, 1, 0]

    if param["order"] == 2:
        phi = phi_2
        local_n = 3
        lines = mesh.lines3
    else:
        phi = phi_1
        local_n = 2
        lines = mesh.lines

    wts, pts, _ = mesh.IntegrationRuleOfLine()

    phi_ref = np.array([[phi[i](x) for i in range(local_n)] for x in pts])

    dirichlet_points = []
    for i, (line, tag) in enumerate(zip(lines, mesh.lineTags)):

        if tag == 2:
            for p in line:
                # keep for later, as points sharing a neuman and dirichlet boundary conditions
                # could be overwritten by neumann.
                # dirichlet takes precedence, as neumann is part of the actual formulation,
                # and dirichlet is enforced artificially
                dirichlet_points.append(p)
        elif tag == 3:
            # TODO: arbitrary neuman conditions via function
            detJ = mesh.calcJacobianDeterminantOfLine(i)

            # assume that neumann = n grad u, meaning it is the normal component on the surface
            # then we can ignore the normal vector
            f[line] += phi_ref.T @ wts * neumann * detJ

    # force dirichlet last
    for p in dirichlet_points:
        f[p] = dirichlet
        A[p, :] = 0
        # TODO: figure out if setting this to zero makes a difference
        # it should be preferable, as it will result in a diagonal matrix,
        # making solving faster
        # A[:, p] = 0
        A[p, p] = 1

    return A, f


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


def solve_poisson(meshfile: str, param: ParamDict) -> None:
    mesh: MeshOps = MeshOps(meshfile)

    #mesh.getNodeNumbersOfTriangle(0, 4)

    # simple file doesn't include the neumann boundary with id==3,
    # so we just add it here ( boundary condition code depends on it)
    if meshfile.endswith("unitSquare1.msh"):
        mesh.lineTags = np.append(mesh.lineTags, 3)
        mesh.lines = np.append(mesh.lines, [[1, 2]], axis=0)

    # Make the mesh into P2 elements for second order
    # mesh normally consists of P1 elements

    if param["order"] != 1:
        add_tri6_line3(mesh)

    print_mat: bool = False

    A, f = assemble_poisson(mesh, param)
    A, f = apply_bc_poisson(mesh, A, f, param)
    if print_mat:
        print(A.toarray())
        print(f)
    un = sp.sparse.linalg.spsolve(A.tocsr(), f)

    if print_mat:
        print(un)

    # visualize on values on the nodes
    # ( accurate for S1 and S2 approximate space)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = un

    # display linear interpolation between between approximate solution on nodes,
    # meaning vertexes with P1 or vertices and midpoints on P2
    triangles = mesh.triangles if param["order"] == 1 else split_6triangles3(mesh)
    ax.plot_trisurf(x, y, z, triangles=triangles, cmap="viridis")
    ax.scatter(x, y, un)

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


param_poisson: ParamDict = dict(
    laplaceCoeff=1,
    #source=lambda x, y: np.float64(1.0),
    source=lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
    dirichlet=0,
    neumann=2,
    order=1,  # Change order number to 1 & 2 for P1 & P2 elements respectively
)

solve_poisson("../mesh/unitSquare2.msh", param_poisson)
#solve_poisson("../mesh/unitSquare1.msh", param_poisson)
