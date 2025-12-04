#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;
using namespace Eigen;

using SparseMatrixD = SparseMatrix<double>;
using TripletD = Triplet<double>;

struct MeshOps
{
  vector<std::array<double, 2>> points;
  vector<std::array<int, 3>> triangles;  // P1 connectivity
  vector<std::array<int, 6>> triangles6; // P2 connectivity
  vector<std::array<int, 3>> lines;      // boundary edges + midpoint
  vector<int> lineTags;

  MeshOps(const string &filename)
  {
    ifstream fin(filename);
    if (!fin)
    {
      cerr << "Could not open mesh file: " << filename << endl;
      exit(1);
    }

    string line;
    while (getline(fin, line))
    {
      istringstream iss(line);
      string token;
      iss >> token;
      if (token == "v")
      { // vertex
        double x, y;
        iss >> x >> y;
        points.push_back({x, y});
      }
      else if (token == "t")
      { // triangle
        int n1, n2, n3;
        iss >> n1 >> n2 >> n3;
        triangles.push_back({n1, n2, n3});
      }
      else if (token == "l")
      { // line
        int n1, n2, tag;
        iss >> n1 >> n2 >> tag;
        lines.push_back({n1, n2, -1});
        lineTags.push_back(tag);
      }
    }
    fin.close();

    if (triangles.empty())
    {
      cerr << "Error: No triangles loaded. Check your mesh file." << endl;
      exit(1);
    }

    add_tri6_line3();
  }

  int getNumberNodes() { return points.size(); }
  int getNumberOfTriangles() { return triangles.size(); }

  std::array<int, 6> getNodeNumbersOfTriangle(int e, int order = 2)
  {
    return triangles6[e];
  }
  std::array<int, 3> getNodeNumbersOfLine(int e)
  {
    return lines[e];
  }

private:
  std::array<double, 2> get_midpoint(const std::array<double, 2> &p1, const std::array<double, 2> &p2)
  {
    return {(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0};
  }

  void add_tri6_line3()
  {
    vector<std::array<double, 2>> extra_points;
    int points_idx = points.size() - 1;
    vector<std::array<int, 6>> new_triangles;

    map<std::array<int, 2>, int> edge_map;

    for (auto &tri : triangles)
    {
      std::array<int, 6> t6;
      for (int i = 0; i < 3; i++)
        t6[i] = tri[i];

      std::array<std::array<int, 2>, 3> edges = {{{tri[0], tri[1]}, {tri[1], tri[2]}, {tri[2], tri[0]}}};
      for (int i = 0; i < 3; i++)
      {
        std::array<int, 2> e = edges[i];
        if (e[0] > e[1])
          swap(e[0], e[1]);
        int mp_idx;
        if (edge_map.count(e))
        {
          mp_idx = edge_map[e];
        }
        else
        {
          points_idx++;
          mp_idx = points_idx;
          edge_map[e] = mp_idx;
          extra_points.push_back(get_midpoint(points[e[0]], points[e[1]]));
        }
        t6[3 + i] = mp_idx;
      }
      new_triangles.push_back(t6);
    }

    points.insert(points.end(), extra_points.begin(), extra_points.end());
    triangles6 = new_triangles;

    for (int i = 0; i < lines.size(); i++)
    {
      std::array<int, 2> e = {lines[i][0], lines[i][1]};
      if (e[0] > e[1])
        swap(e[0], e[1]);
      int mp_idx = edge_map[e];
      lines[i][2] = mp_idx;
    }
  }
};

std::array<double, 2> source_f(double x, double y)
{
  return {0.0, 0.0};
}

std::array<double, 2> omega2(double x, double y)
{
  return {y * (y - 1) * (y + 1), 0.0};
}

void assemble_stokes(MeshOps &mesh, SparseMatrixD &M, VectorXd &f)
{
  int N_u = mesh.getNumberNodes();       // number of velocity nodes (P2)
  int N_p = mesh.getNumberOfTriangles(); // number of pressure nodes (P1)

  vector<TripletD> triplets;
  f = VectorXd::Zero(2 * N_u + N_p);

  for (int i = 0; i < 2 * N_u; i++)
    triplets.push_back({i, i, 1.0});
  for (int i = 0; i < N_p; i++)
    triplets.push_back({2 * N_u + i, 2 * N_u + i, 0.0});

  M.resize(2 * N_u + N_p, 2 * N_u + N_p);
  M.setFromTriplets(triplets.begin(), triplets.end());
}

void apply_bc_stokes(MeshOps &mesh, SparseMatrixD &M, VectorXd &f)
{
  int N_u = mesh.getNumberNodes();

  for (int i = 0; i < mesh.lines.size(); i++)
  {
    int tag = mesh.lineTags[i];
    if (tag == 2)
    { // Dirichlet
      auto l = mesh.getNodeNumbersOfLine(i);
      for (int j = 0; j < 3; j++)
      {
        int idx = l[j];
        if (idx >= N_u)
          continue;
        f(idx) = omega2(mesh.points[idx][0], mesh.points[idx][1])[0];
        f(N_u + idx) = omega2(mesh.points[idx][0], mesh.points[idx][1])[1];

        for (SparseMatrixD::InnerIterator it(M, idx); it; ++it)
          it.valueRef() = 0.0;
        M.coeffRef(idx, idx) = 1.0;
        for (SparseMatrixD::InnerIterator it(M, N_u + idx); it; ++it)
          it.valueRef() = 0.0;
        M.coeffRef(N_u + idx, N_u + idx) = 1.0;
      }
    }
  }

  if (2 * N_u < M.rows())
  {
    f(2 * N_u) = 0.0;
    for (SparseMatrixD::InnerIterator it(M, 2 * N_u); it; ++it)
      it.valueRef() = 0.0;
    M.coeffRef(2 * N_u, 2 * N_u) = 1.0;
  }
  else
  {
    cerr << "Warning: Cannot fix pressure, index out of range!" << endl;
  }
}

int main()
{
  MeshOps mesh("../mesh/unitSquareStokes.msh");
  cout << "Mesh info: points=" << mesh.getNumberNodes()
       << ", triangles=" << mesh.getNumberOfTriangles()
       << ", lines=" << mesh.lines.size() << endl;

  SparseMatrixD M;
  VectorXd f;

  assemble_stokes(mesh, M, f);
  apply_bc_stokes(mesh, M, f);

  SparseLU<SparseMatrixD> solver;
  solver.compute(M);
  if (solver.info() != Success)
  {
    cerr << "Decomposition failed!" << endl;
    return -1;
  }
  VectorXd sol = solver.solve(f);
  if (solver.info() != Success)
  {
    cerr << "Solving failed!" << endl;
    return -1;
  }

  int N_u = mesh.getNumberNodes();
  cout << "Solution vector size: " << sol.size() << endl;
  for (int i = 0; i < min(10, (int)sol.size()); i++)
    cout << sol(i) << endl;

  return 0;
}
