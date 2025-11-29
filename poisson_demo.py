import numpy as np
from scipy.sparse import lil_matrix
from mesh_ops import MeshOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def assemble_poisson(mesh: MeshOps, param):
    # TODO!
    N = mesh.getNumberNodes()
    A = lil_matrix((N, N))
    num_elements = mesh.getNumberOfTriangles()
    e = 0 # first element
    elemA = np.zeros((3, 3))
    dphi_ref = np.array([[-1, -1],
                          [1, 0],
                          [0, 1]])
    invJ = mesh.calcInverseJacobianOfTriangle(e)
    J = mesh.calcJacobianOfTriangle(e)
    dphi = dphi_ref @ invJ
    detJ = mesh.calcJacobianDeterminantOfTriangle(e)
    elemA = (dphi @ dphi.T) * detJ * 1/2
    con = mesh.getNodeNumbersOfTriangle(e)
    A[np.ix_(con, con)] += elemA
    return A, None
    # return A, f


def apply_bc_poisson(mesh: MeshOps, A, f, param):
    # TODO
    return A, f


def solve_poisson(meshfile, param):
    mesh = MeshOps(meshfile)

    A, f = assemble_poisson(mesh, param)
    A, f = apply_bc_poisson(mesh, A, f, param)
    # Solve u = ...

param_poisson = dict(
    laplaceCoeff=1,
    source=lambda x, y: 1, # np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
    dirichlet=0,
    neumann=1,
    order=1
)
solve_poisson("mesh/unitSquare1.msh", param_poisson)