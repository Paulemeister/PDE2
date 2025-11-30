import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, spsolve_triangular
from mesh_ops import MeshOps
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def assemble_poisson(mesh: MeshOps, param):
    # TODO!
    N = mesh.getNumberNodes()
    A = lil_matrix((N, N))
    f = np.zeros(N)

    wts, pts, nums = mesh.IntegrationRuleOfTriangle()
    phi_ref = lambda x,y: np.array([1-x-y,x,y])
    rhs_ref = np.zeros(3)
    for i in range(nums):
        x, y = pts[i][0], pts[i][1]
        rhs_ref += wts[i] * param["source"](x,y) * phi_ref(x,y)

    dphi_ref = np.array([[-1, -1],
                          [1, 0],
                          [0, 1]])

    num_elements = mesh.getNumberOfTriangles()
    for e in range(num_elements):
        elemA = np.zeros((3, 3))
        invJ = mesh.calcInverseJacobianOfTriangle(e)
        J = mesh.calcJacobianOfTriangle(e)
        dphi = dphi_ref @ invJ
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)
        elemA = (dphi @ dphi.T) * detJ * 1/2
        con = mesh.getNodeNumbersOfTriangle(e)
        A[np.ix_(con, con)] += elemA
        f[con] += detJ * rhs_ref
    return A, f
    # return A, f


def apply_bc_poisson(mesh: MeshOps, A, f, param):
    # TODO
    numTagLines = mesh.getNumberOfTaggedLines()
    dirichlet = param["dirichlet"]
    neumann = param["neumann"]
    for i in range(numTagLines):
        if mesh.getTagOfLine(i) == 2:
            for node in mesh.getNodeNumbersOfLine(i):
                A.rows[node] = [node]
                A.data[node] = [1.0]
                f[node] = dirichlet
        elif mesh.getTagOfLine(i) == 3:
            for node in mesh.getNodeNumbersOfLine(i):
                f[node] += neumann*mesh.calcJacobianDeterminantOfLine(i)
    return A, f


def solve_poisson(meshfile, param):
    mesh = MeshOps(meshfile)

    A, f = assemble_poisson(mesh, param)
    A, f = apply_bc_poisson(mesh, A, f, param)
    u = spsolve(A,f)
    print(u)

param_poisson = dict(
    laplaceCoeff=1,
    source=lambda x, y: 1, # np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
    dirichlet=0,
    neumann=1,
    order=1
)
solve_poisson("mesh/unitSquare2.msh", param_poisson)

mesh = MeshOps("mesh/unitSquare2.msh")

listGlobNodes = mesh.getNodeList()
xs = listGlobNodes[:,0]
ys = listGlobNodes[:,1]

fig, ax = plt.subplots()

ax.scatter(xs,ys)
ax.set_xlim(0-0.2,1+0.2)
ax.set_ylim(0-0.2,1+0.2)
ax.axis('square')

plt.show()
