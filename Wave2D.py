import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

# where to put L
class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # can cause overwriting of initialization settings...
        self.N = N
        # assume class is properly initialized, bascially L should be set in constructor
        self.dx = self.L/self.N
        # in this very case the mesh is square, so dx equals dy, but I still define it
        self.dy = self.L/self.N
        x = np.linspace(0, self.L, N+1)
        y = np.linspace(0, self.L, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij', sparse = sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self._w

    @w.setter
    def w(self, value):
        """Set the dispersion coefficient"""
        self._w = value

    @property
    def dt(self):
        """Return the time step"""
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set the dispersion coefficient"""
        self._dt = value

    def bla(self):
        print("returning bla")
        return "bla"

    def ue(self):
        """Return the exact standing wave"""
        return sp.sin(self.mx*sp.pi*x)*sp.sin(self.my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, L, N, mx, my):
        """Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        L : number
        The length of the domain in both x and y directions

        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.L = L
        self.N = N
        self.mx = mx
        self.my = my

    #u solution
    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_exact = sp.lambdify((x, y, t), self.ue())(self.xij, self.yij, (t0)*self.dt)
        error = np.sqrt(self.dx**2 * np.sum((u - u_exact) ** 2))
        return [error]



    # apply boundary conditions
    # arrays are passed by reference so I do not need to return it
    def apply_bcs(self, Un):
        Un[0] = 0
        Un[-1] = 0
        Un[:, -1] = 0
        Un[:, 0] = 0

    #n%-1 returns 0!!!
    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """

        kx = mx*sp.pi; ky = my*sp.pi
        self.w = c * sp.sqrt(kx**2 + ky**2)

        # L equa to 1 should be set at initialization
        self.initialize(1, N,  mx, my)
        self.create_mesh(N, sparse=False)

        D = self.D2(N)/self.dx**2

        self.dt = cfl*self.dx/c

        Unp1, Un, Unm1 = np.zeros((3, N+1, N+1))
        t0 = 0
        kx = mx*sp.pi; ky = my*sp.pi
        Unm1 = sp.lambdify((x, y, t), self.ue())(self.xij, self.yij, t0)
        Un = sp.lambdify((x, y, t), self.ue())(self.xij, self.yij, self.dt)
        plotdata = {0: Unm1.copy()}
        for n in range(1, Nt +1):
            Unp1[:] = 2*Un - Unm1 + (c*self.dt)**2*(D @ Un + Un @ D.T)
            # Set boundary conditions
            self.apply_bcs(Unp1)
            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1
            if store_data == -1:
                # in case of -1 we do not store inside the for loop, but we continue conmputing for all timestemps
                continue
            if n % store_data == 0:
                plotdata[n] = Unm1.copy() # Unm1 is now swapped to Un


        if store_data > 0:
            return plotdata
        else:
            if store_data == -1:
                # computations are done within <0, Nt) step
                #Un is Unp1 so 1 step ahead,
                return self.dx, self.l2_error(Un, Nt +1)
            else:
                raise ValueError(f'store_data has value of {store_data}, expected values are -1 or integers greater than 0')

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            print(f"dx: {dx}, error {err}")
            #print(type(err))
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

def test_convergence_wave2d():
    sol = Wave2D()
    sol.convergence_rates(mx=2, my=3)
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2


'''

    class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        raise NotImplementedError

    def ue(self, mx, my):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    mx = my = 2
    cfl = 1 / np.sqrt(2)
    assert abs(r[-1]) < 1e-15
    raise NotImplementedError
'''