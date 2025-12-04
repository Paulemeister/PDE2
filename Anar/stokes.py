import numpy as np
from scipy.sparse import lil_matrix
from scipy.linalg import inv
from scipy.sparse.linalg import spsolve
from mesh_ops import MeshOps
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def assemble_stokes(mesh: MeshOps, param):
    source = lambda pts: param["source"](pts[0],pts[1])
    wts, pts, nums = mesh.IntegrationRuleOfTriangle()
    T = mesh.getNumberOfTriangles() # Number of elements
    N = mesh.getNumberNodes() # Number of nodes

    N_P1 = -1 # Number of P1 nodes
    P1_nodes = []
    for i in range(T):
        nodes = mesh.getNodeNumbersOfTriangle(i)
        for node in nodes:
            if node not in P1_nodes:
                P1_nodes.append(int(node))
        N_P1 = max(N_P1,np.max(mesh.getNodeNumbersOfTriangle(i)))
    N_P1 += 1
    P1_nodes.sort()
    P1_nodes = np.array(P1_nodes)
    All_nodes = np.concatenate((np.arange(2*N),2*N+P1_nodes))

    stiffness = lil_matrix((2*N+N_P1, 2*N+N_P1))
    rhs = np.zeros(2*N+N_P1)

    x, y = pts[:,0], pts[:,1]
    psi_ref = np.column_stack((1-x-y,x,y))
    phi_ref =  np.column_stack(((1-x-y)*(1-2*x-2*y),
                                x*(2*x-1),
                                y*(2*y-1),
                                4*x*(1-x-y),
                                4*x*y,
                                4*y*(1-x-y)))
    dpsi_ref = np.array([[-1, -1], [1, 0], [0, 1]])
    dphi_ref = np.zeros((nums,6,2))
    dphi_ref[:,0,0] = 4*x+4*y-3
    dphi_ref[:,0,1] = 4*x+4*y-3
    dphi_ref[:,1,0] = 4*x-1
    dphi_ref[:,1,1] = 0
    dphi_ref[:,2,0] = 0
    dphi_ref[:,2,1] = 4*y-1
    dphi_ref[:,3,0] = -8*x-4*y+4
    dphi_ref[:,3,1] = -4*x
    dphi_ref[:,4,0] = 4*y
    dphi_ref[:,4,1] = 4*x
    dphi_ref[:,5,0] = -4*y
    dphi_ref[:,5,1] = -4*x-8*y+4

    mx = -1
    mn = N_P1 + 2*N
    for e in range(T):
        invJ = mesh.calcInverseJacobianOfTriangle(e)
        detJ = mesh.calcJacobianDeterminantOfTriangle(e)

        con_P2 = mesh.getNodeNumbersOfTriangle(e,2)
        con_P1 = mesh.getNodeNumbersOfTriangle(e,1)

        ## Stiffness matrix
        dphi = dphi_ref @ invJ
        elemA = np.einsum('i,ijk->ijk',wts,dphi@dphi.transpose(0,2,1)).sum(axis=0)
        elemB_x = np.einsum('i,ij,ik->ijk',wts,psi_ref,dphi[:,:,0]).sum(axis=0)
        elemB_y = np.einsum('i,ij,ik->ijk',wts,psi_ref,dphi[:,:,1]).sum(axis=0)

        stiffness[np.ix_(con_P2, con_P2)] += detJ*elemA
        stiffness[np.ix_(con_P2+N, con_P2+N)] += detJ*elemA
        stiffness[np.ix_(con_P1+2*N, con_P2)] -= detJ*elemB_x
        stiffness[np.ix_(con_P1+2*N, N+con_P2)] -= detJ*elemB_y
        stiffness[np.ix_(con_P2, con_P1+2*N)] -= detJ*elemB_x.T
        stiffness[np.ix_(N+con_P2, con_P1+2*N)] -= detJ*elemB_y.T

        ## Right hand side
        point = np.apply_along_axis(lambda x: mesh.calcMappedIntegrationPointOfTriangle(e,x), 1, pts)
        f = np.apply_along_axis(source, 1, point)
        f_x = np.einsum('i,i,ij->ij',wts,f[:,0],phi_ref).sum(axis=0)
        f_y = np.einsum('i,i,ij->ij',wts,f[:,1],phi_ref).sum(axis=0)
        rhs[con_P2] += detJ*f_x
        rhs[con_P2 + N] += detJ*f_y

    csr_mat = stiffness.tocsr()
    rows_removed = csr_mat[All_nodes, :]

    csc_mat = rows_removed.tocsc()
    result = csc_mat[:, All_nodes]

    stiffness = result.tolil()
    rhs = rhs[All_nodes]

    return stiffness, rhs

def apply_bc_poisson(mesh: MeshOps, A, f, param):
    # TODO
    coords = mesh.points
    N = mesh.getNumberNodes()
    numTagLines = mesh.getNumberOfTaggedLines()
    dirichlet = lambda coord: param["dirichlet"](coord[0],coord[1])

#     fun_ref_1 = np.array([1,1])
#     fun_ref_2 = np.array([1/3,1/3,4/3])

    for i in range(numTagLines):
        tag = mesh.getTagOfLine(i)
        if tag == 2:
            for node in mesh.getNodeNumbersOfLine(i,2):
                rhs = dirichlet(coords[node])
                A.rows[node] = [node]
                A.data[node] = [1.0]
                f[node] = -rhs[0]

                A.rows[node+N] = [node+N]
                A.data[node+N] = [1.0]
                f[node+N] = -rhs[1]
        elif tag == 4 or tag == 5:
            for node in mesh.getNodeNumbersOfLine(i,2):
                A.rows[node] = [node]
                A.data[node] = [1.0]
                f[node] = 0

                A.rows[node+N] = [node+N]
                A.data[node+N] = [1.0]
                f[node+N] = 0

    numRows, numCols = A.shape
    A.rows[numRows-1] = [numRows-1]
    A.data[numRows-1] = [1.0]
    f[numRows-1] = 0

    return A, f


def solve_poisson(meshfile, param):
    mesh = MeshOps(meshfile)
    N = mesh.getNumberNodes()

    print("\nAssembling stiffness matrix and etc...\n\n")
    A, f = assemble_stokes(mesh, param)

    print("\n\nApplying boundary conditions...\n\n")
    A, f = apply_bc_poisson(mesh, A, f, param)
    print(A.shape)

    print("\n\nSolving the linear system...\n\n")
    A = A.tocsr()
    sol = spsolve(A,f)
    u_x = sol[:N]
    u_y = sol[N:2*N]
    p = sol[2*N:]

    xs = mesh.points[:,0]
    ys = mesh.points[:,1]

    plt.tripcolor(xs,ys,u_x)
    plt.colorbar()

param_poisson = dict(
    source=lambda x, y: [0,0],
    dirichlet=lambda x, y: [(y-1)*(y+1), 0],
    neumann=0,
)

solve_poisson("unitSquareStokes.msh", param_poisson)

plt.show()