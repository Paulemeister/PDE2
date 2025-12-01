import numpy as np
from scipy.sparse import lil_matrix #type: ignore
from scipy.sparse.linalg import spsolve #type: ignore
from mesh_ops import MeshOps
import matplotlib #type: ignore
# matplotlib.use('Agg')
import matplotlib.pyplot as plt #type: ignore

def assemble_poisson(mesh: MeshOps, param):
    # TODO!
    Pn = param["order"]
    source = lambda point: param["source"](point[0],point[1])
    wts, pts, nums = mesh.IntegrationRuleOfTriangle()
    T = mesh.getNumberOfTriangles() # Number of edges
    N = mesh.getNumberNodes()
    A = lil_matrix((N, N))
    f = np.zeros(N)

    def phi_ref_1(pts):
        x, y = pts[:,0], pts[:,1]
        return np.column_stack((1-x-y,x,y))

    def phi_ref_2(pts):
        x, y = pts[:,0], pts[:,1]
        return np.column_stack(((1-x-y)*(1-2*x-2*y),
                                x*(2*x-1),
                                y*(2*y-1),
                                4*x*(1-x-y),
                                4*x*y,
                                4*y*(1-x-y)))

    dphi_ref_1 = np.array([[-1, -1], [1, 0], [0, 1]])
    def dphi_ref_2(pts):
        x, y = pts[:,0], pts[:,1]
        out = np.zeros((len(pts),6,2))

        out[:,0,0] = 4*x+4*y-3
        out[:,0,1] = 4*x+4*y-3
        out[:,1,0] = 4*x-1
        out[:,1,1] = 0
        out[:,2,0] = 0
        out[:,2,1] = 4*y-1
        out[:,3,0] = -8*x-4*y+4
        out[:,3,1] = -4*x
        out[:,4,0] = 4*y
        out[:,4,1] = 4*x
        out[:,5,0] = -4*y
        out[:,5,1] = -4*x-8*y+4

        return out

    if Pn == 1:
        phi_ref = phi_ref_1(pts)

        for e in range(T):
            invJ = mesh.calcInverseJacobianOfTriangle(e)
            detJ = mesh.calcJacobianDeterminantOfTriangle(e)
            point = np.apply_along_axis(lambda x: mesh.calcMappedIntegrationPointOfTriangle(e,x), 1, pts)
            func = np.apply_along_axis(source, 1, point)
            rhs = detJ*np.einsum('i,i,ij->ij',wts,func,phi_ref)

            dphi = dphi_ref_1 @ invJ
            elemA = (dphi @ dphi.T) * detJ * 1/2
            elemA[np.abs(elemA) < 1e-16] = 0

            con = mesh.getNodeNumbersOfTriangle(e)
            A[np.ix_(con, con)] += elemA
            f[con] += rhs.sum(axis=0)

    elif Pn == 2:
        phi_ref = phi_ref_2(pts)
        dphi_ref = dphi_ref_2(pts)

        for e in range(T):
            invJ = mesh.calcInverseJacobianOfTriangle(e)
            detJ = mesh.calcJacobianDeterminantOfTriangle(e)
            point = np.apply_along_axis(lambda x: mesh.calcMappedIntegrationPointOfTriangle(e,x), 1, pts)
            func = np.apply_along_axis(source, 1, point)
            rhs = detJ*np.einsum('i,i,ij->ij',wts,func,phi_ref)

            dphi = dphi_ref @ invJ
            elemA = detJ*np.einsum('i,ijk->ijk',wts,dphi@dphi.transpose(0,2,1))

            con = mesh.getNodeNumbersOfTriangle(e,Pn)
            A[np.ix_(con, con)] += elemA.sum(axis=0)
            f[con] += rhs.sum(axis=0)

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

    print("\nAssembling stiffness matrix and etc...\n\n")
    A, f = assemble_poisson(mesh, param)

    print("\n\nApplying boundary conditions...\n\n")
    A, f = apply_bc_poisson(mesh, A, f, param)

    print("\n\nSolving the linear system...\n\n")
    A = A.tocsr()
    u = spsolve(A,f)

    xs = mesh.points[:,0]
    ys = mesh.points[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_trisurf(xs, ys, u, cmap="viridis", linewidth=0.2, antialiased=True)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y)")


param_poisson = dict(
    laplaceCoeff=1,
#    source=lambda x, y: 1, # np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
    source=lambda x, y: np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
    dirichlet=0,
    neumann=2,
    order=2 # Change order number to 1 & 2 for P1 & P2 elements respectively
)

solve_poisson("unitSquare2_P2.msh", param_poisson) # Change mesh name accordingly for P1 and P2 elements.

plt.show()
