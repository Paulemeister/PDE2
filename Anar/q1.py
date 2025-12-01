import sys
import os

sys.path.append('/Users/brianhuynh/PDE2')

import numpy as np
from scipy.sparse import lil_matrix #type: ignore
from scipy.sparse.linalg import spsolve, spsolve_triangular #type: ignore
from mesh_ops import MeshOps
import matplotlib #type: ignore
# matplotlib.use('Agg')
import matplotlib.pyplot as plt #type: ignore

def assemble_poisson(mesh: MeshOps, param):
    # TODO!
    Pn = param["order"]
    wts, pts, nums = mesh.IntegrationRuleOfTriangle()
    T = mesh.getNumberOfTriangles() # Number of edges
    N = mesh.getNumberNodes()
    A = lil_matrix((N, N))
    f = np.zeros(N)

    phi_ref_1 = lambda x,y: np.array([1-x-y,x,y])
    phi_ref_2 = lambda x,y: np.array([(1-x-y)*(1-2*x-2*y),
                                      x*(2*x-1),
                                      y*(2*y-1),
                                      4*x*(1-x-y),
                                      4*x*y,
                                      4*y*(1-x-y)])

    dphi_ref_1 = np.array([[-1, -1], [1, 0], [0, 1]])
    dphi_ref_2 = lambda x,y: np.array([[4*x+4*y-3, 4*x+4*y-3],
                                       [4*x-1, 0],
                                       [0, 4*y-1],
                                       [-8*x-4*y+4, -4*x],
                                       [4*y, 4*x],
                                       [-4*y, -4*x-8*y+4]])


    if Pn == 1:
        rhs_ref = np.zeros(3)
        for i in range(nums):
            x, y = pts[i][0], pts[i][1]
            rhs_ref += wts[i] * param["source"](x,y) * phi_ref_1(x,y)
        rhs_ref[np.abs(rhs_ref) < 1e-16] = 0

        for e in range(T):
            elemA = np.zeros((3,3))
            invJ = mesh.calcInverseJacobianOfTriangle(e)
            detJ = mesh.calcJacobianDeterminantOfTriangle(e)

            dphi = dphi_ref_1 @ invJ
            elemA = (dphi @ dphi.T) * detJ * 1/2
            elemA[np.abs(elemA) < 1e-16] = 0

            con = mesh.getNodeNumbersOfTriangle(e)
            A[np.ix_(con, con)] += elemA
            f[con] += detJ * rhs_ref

    elif Pn == 2:
        rhs_ref = np.zeros(6)
        for i in range(nums):
            x, y = pts[i][0], pts[i][1]
            rhs_ref += wts[i] * param["source"](x,y) * phi_ref_2(x,y)
        rhs_ref[np.abs(rhs_ref) < 1e-16] = 0

        for e in range(T):
            elemA = np.zeros((6,6))
            invJ = mesh.calcInverseJacobianOfTriangle(e)
            detJ = mesh.calcJacobianDeterminantOfTriangle(e)

            for i in range(nums):
                x, y = pts[i][0], pts[i][1]
                dphi = dphi_ref_2(x,y) @ invJ
                elemA += (dphi @ dphi.T) * wts[i]
            elemA *= elemA * detJ
            elemA[np.abs(elemA) < 1e-16] = 0

            con = mesh.getNodeNumbersOfTriangle(e,Pn)
            A[np.ix_(con, con)] += elemA
            f[con] += detJ * rhs_ref

    return A, f
    # return A, f


def apply_bc_poisson(mesh: MeshOps, A, f, param):
    # TODO
    numTagLines = mesh.getNumberOfTaggedLines()
    dirichlet = param["dirichlet"]
    neumann = param["neumann"]

    fun_ref_1 = np.array([1,1])
    fun_ref_2 = np.array([1/3,1/3,4/3])

    Pn = param["order"]
    for i in range(numTagLines):
        if mesh.getTagOfLine(i) == 2:
            for node in mesh.getNodeNumbersOfLine(i,Pn):
                A.rows[node] = [node]
                A.data[node] = [1.0]
                f[node] = dirichlet
        elif mesh.getTagOfLine(i) == 3:
            detJ = mesh.calcJacobianDeterminantOfLine(i)
            if Pn == 1:
                con = mesh.getNodeNumbersOfLine(i,Pn)
                f[con] += neumann*fun_ref_1*detJ
            elif Pn == 2:
                con = mesh.getNodeNumbersOfLine(i,Pn)
                f[con] += neumann*fun_ref_2*detJ
    return A, f


def solve_poisson(meshfile, param):
    mesh = MeshOps(meshfile)

    A, f = assemble_poisson(mesh, param)
#     plt.spy(A)

#     A_dense = A.toarray()
#     print("A sparse matrix:\n")
#     for row in A_dense:
#         print(" ".join(f"{val:6.3f}" for val in row))
#     print("\nf right hand side:\n")
#     print(" ".join(f"{val:6.3f}" for val in f))

    print("\n\nApplying boundary conditions...\n\n")

    A, f = apply_bc_poisson(mesh, A, f, param)
#     A_dense = A.toarray()
#     print("A sparse matrix:\n")
#     for row in A_dense:
#         print(" ".join(f"{val:6.3f}" for val in row))
#     print("\nf right hand side:\n")
#     print(" ".join(f"{val:6.3f}" for val in f))

    print("\n\nSolving the linear system...\n\n")

    u = spsolve(A,f)
    print("\nalpha_i coefficients for shape functions:")
    for val in u:
        print(f"{val:12.9f}")
    print("\n")

param_poisson = dict(
    laplaceCoeff=1,
    source=lambda x, y: 1, # np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
#     source=lambda x, y: np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
    dirichlet=0,
    neumann=0,
    order=2 # Change order number to 1 & 2 for P1 & P2 elements respectively
)

solve_poisson("unitSquare2_P2.msh", param_poisson) # Change mesh name accordingly for P1 and P2 elements.

plt.show()
