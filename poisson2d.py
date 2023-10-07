import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = ue.diff(x, 2) + ue.diff(y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        x_array = np.linspace(0, self.L, N+1)
        y_array = np.linspace(0, self.L, N+1)
        self.xij, self.yij = np.meshgrid(x_array, y_array, indexing='ij')
        #tutaj refactor we should use smesh
        #smesh = np.meshgrid(x, y, indexing='ij', sparse=True)

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2x = (1./self.dx**2)*self.D2()
        D2y = (1./self.dy**2)*self.D2()
        return (sparse.kron(D2x, sparse.eye(self.N+1)) +
            sparse.kron(sparse.eye(self.N+1), D2y))

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        return np.where(B.ravel() == 1)[0]

    def assemble(self,N):
        """Return assembled matrix A and right hand side vector b"""
        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        boundary_indc = self.get_boundary_indices()

        A = self.laplace()
        A = A.tolil()
        for i in boundary_indc:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        x_indices = self.xij[:,0]
        y_indices = self.yij[0]
        b = F.ravel()
        #b = F.ravel().copy()
        for b_index in boundary_indc:
            row_index, col_index = np.unravel_index(b_index, (N+1, N+1))
            b[b_index] = self.ue.subs({x: x_indices[row_index], y: y_indices[col_index]}).n()
        return A, b

    def l2_error(self, u):
        u_exact = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        return np.sqrt(self.dx * self.dy * np.sum((u - u_exact) ** 2))

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        # I know its not consistent but i need it

        self.dx = self.L/N
        self.dy = self.L/N
        #in this case the mesh is square, so dx == dy == h
        self.h = self.dx

        self.create_mesh(N)
        A, b = self.assemble(N)
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes

        """

        E = []
        h = []
        N0 = 8 # start value of
        for m in range(m):
            u = self(N0)
            #error = self.l2_error(u)
            #E.append(error)
            E.append(self.l2_error(u))
            h.append(self.dx)
            N0 *= 2
        # orders
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x_arg, y_arg):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)


        """
        #tutaj change it: linear interpolation works in this case, but Lagrangian interpolation should work
        x_indices = np.linspace(0, self.L, self.N+1)
        y_indices = np.linspace(0, self.L, self.N+1)
        interpolator = RegularGridInterpolator((x_indices, y_indices), self.U, method='linear', bounds_error=False, fill_value=None)
        interpolated_value = interpolator((x_arg, y_arg))
        return interpolated_value

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h, y: 1-sol.h/2}).n()) < 1e-3

