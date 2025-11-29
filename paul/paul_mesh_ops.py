###############################
# Edits: Just stuff to make Pylance happy
# - adding type hints
# - adding exceptions for passing order > 2
# - converting empty lists to empty np arrays
# - set empty tag lists to empty lists instead of None
################################

from typing import Literal
from matplotlib.pylab import int32
import numpy as np
import meshio
from numpy.typing import NDArray


class MeshOps:
    """
    Pythonic mesh operations class.
    Reads a Gmsh .msh file using meshio.
    Supports linear (line2, tri3) and quadratic (line3, tri6).
    """

    def __init__(self, filename):
        mesh: meshio.Mesh = meshio.read(filename)

        self.points: NDArray[np.floating] = mesh.points[:, :2]  # (N,2)
        self.nbNod: int = self.points.shape[0]

        self.triTags: NDArray[int32] = np.array([])
        self.lineTags: NDArray[int32] = np.array([])

        # --- extract elements by type ---
        self.triangles: NDArray[int32] = np.array([])
        self.triangles6: NDArray[int32] = np.array([])
        self.lines: NDArray[int32] = np.array([])
        self.lines3: NDArray[int32] = np.array([])

        for block in mesh.cells:
            if block.type == "triangle":
                self.triangles = block.data
            elif block.type == "triangle6":
                self.triangles6 = block.data
            elif block.type == "line":
                self.lines = block.data
            elif block.type == "line3":
                self.lines3 = block.data

        # Physical tags
        self._extract_physical_tags(mesh)

        # Extract low-order connectivity for mixed-order problem
        if len(self.triangles6) > 0:
            self.triangles = self.triangles6[:, :3]  # only corners
        if len(self.lines3) > 0:
            self.lines = self.lines3[:, :2]

        self.nbTriangles: int = len(self.triangles)
        self.nbLines: int = len(self.lines)

    # ----------------------------------------------------------
    #   Physical tag extraction from meshio cell_data
    # ----------------------------------------------------------
    def _extract_physical_tags(self, mesh: meshio.Mesh) -> None:

        if "gmsh:physical" in mesh.cell_data_dict:
            tags = mesh.cell_data_dict["gmsh:physical"]
        else:
            raise RuntimeError("Mesh file has no physical tags.")

        for block, tag in zip(mesh.cells, tags.values()):
            if block.type == "triangle":
                self.triTags = tag
            elif block.type == "triangle6":
                self.triTags = tag
            elif block.type == "line":
                self.lineTags = tag
            elif block.type == "line3":
                self.lineTags = tag

    # ----------------------------------------------------------
    #   Basic info
    # ----------------------------------------------------------
    def getNumberNodes(self) -> int:
        return self.nbNod

    def getNodeList(self) -> NDArray[np.floating]:
        return self.points

    def getNumberOfTriangles(self) -> int:
        return self.nbTriangles

    def getNumberOfTaggedLines(self) -> int:
        return self.nbLines

    def getVolumeElementListTriangles(self) -> NDArray[int32]:
        return self.triangles[:, :3]

    def getTagOfLine(self, i) -> NDArray[int32]:
        return self.lineTags[i]

    def getTagOfTriangle(self, i) -> NDArray[int32]:
        return self.triTags[i]

    # ----------------------------------------------------------
    #   Node connectivity
    # ----------------------------------------------------------
    def getNodeNumbersOfLine(self, i, order=1) -> NDArray[int32]:
        if order == 1:
            return self.lines[i, :2]
        elif order == 2:
            return self.lines3[i, :3]
        raise Exception("wrong order")

    def getNodeNumbersOfTriangle(self, i, order=1) -> NDArray[int32]:
        if order == 1:
            return self.triangles[i, :3]
        elif order == 2:
            return self.triangles6[i, :6]
        raise Exception("wrong order")

    # ----------------------------------------------------------
    #   Geometry: Jacobians
    # ----------------------------------------------------------
    def calcJacobianOfLine(self, i: int) -> NDArray[np.floating]:
        n1, n2 = self.lines[i, :2]
        p1, p2 = self.points[n1], self.points[n2]
        return 0.5 * (p2 - p1)  # 2x1 vector

    def calcJacobianDeterminantOfLine(self, i: int) -> np.floating:
        J = self.calcJacobianOfLine(i)
        return np.linalg.norm(J)

    def calcMappedIntegrationPointOfLine(self, i: int, xi: int) -> NDArray[np.floating]:
        # xi in [-1,1]
        n1, n2 = self.lines[i, :2]
        p1, p2 = self.points[n1], self.points[n2]
        return 0.5 * ((1 - xi) * p1 + (1 + xi) * p2)

    def getNormalVectorOfLine(self, i: int) -> NDArray[np.floating]:
        n1, n2 = self.lines[i, :2]
        p1, p2 = self.points[n1], self.points[n2]
        t = p2 - p1
        t = t / np.linalg.norm(t)
        return np.array([t[1], -t[0]])

    # ---------------- triangle geometry -----------------------
    def calcJacobianOfTriangle(self, i: int) -> NDArray[np.floating]:
        n1, n2, n3 = self.triangles[i, :3]
        p1, p2, p3 = self.points[n1], self.points[n2], self.points[n3]
        J = np.array([[p2[0] - p1[0], p3[0] - p1[0]], [p2[1] - p1[1], p3[1] - p1[1]]])
        return J

    def calcInverseJacobianOfTriangle(self, i: int) -> NDArray[np.floating]:
        return np.linalg.inv(self.calcJacobianOfTriangle(i))

    def calcJacobianDeterminantOfTriangle(self, i: int) -> np.floating:
        return abs(np.linalg.det(self.calcJacobianOfTriangle(i)))

    def calcMappedIntegrationPointOfTriangle(self, i, ip) -> NDArray[np.floating]:
        # ip = (xi,eta) in ref triangle
        n1 = self.triangles[i, 0]
        p1 = self.points[n1]
        J = self.calcJacobianOfTriangle(i)
        return p1 + J @ np.array(ip)

    # ----------------------------------------------------------
    #   Quadrature rules
    # ----------------------------------------------------------
    @staticmethod
    def IntegrationRuleOfLine() -> (
        tuple[NDArray[np.floating], NDArray[np.floating], Literal[3]]
    ):
        pts = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        wts = np.array([5 / 9, 8 / 9, 5 / 9])
        return wts, pts, 3

    @staticmethod
    def IntegrationRuleOfTriangle() -> (
        tuple[NDArray[np.floating], NDArray[np.floating], Literal[7]]
    ):
        pts = np.zeros((7, 2))
        pts[0] = [1 / 3, 1 / 3]
        pts[1] = [(6 + np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21]
        pts[2] = [(9 - 2 * np.sqrt(15)) / 21, (6 + np.sqrt(15)) / 21]
        pts[3] = [(6 + np.sqrt(15)) / 21, (9 - 2 * np.sqrt(15)) / 21]
        pts[4] = [(6 - np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21]
        pts[5] = [(9 + 2 * np.sqrt(15)) / 21, (6 - np.sqrt(15)) / 21]
        pts[6] = [(6 - np.sqrt(15)) / 21, (9 + 2 * np.sqrt(15)) / 21]

        w0 = 9 / 80
        w1 = (155 + np.sqrt(15)) / 2400
        w2 = (155 - np.sqrt(15)) / 2400
        wts = np.array([w0, w1, w1, w1, w2, w2, w2])

        return wts, pts, 7
