import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from firedrake.__future__ import interpolate


class InterpolateAtArbitraryPoints:
    """Fast interpolation of a UFL expression or Function at specified points.

    Implement [Primary API: Interpolation onto a vertex-only mesh]
    (https://www.firedrakeproject.org/point-evaluation.html#primary-api-interpolation-onto-a-vertex-only-mesh)
    as a convenient to use class that is much faster than evaluating a function
    using the `.at()` method.

    Attributes
    ----------
        _vom (firedrake.mesh.MeshGeometry): VertexOnlyMesh.
        _P0DG (firedrake.functionspaceimpl.WithGeometry): Polynomial degree 0 Discontinuous Galerkin (P0DG) space.

    Methods
    -------
        __call__: Invoke interpolation for a given function at `_points`.

    Examples
    --------
        Example usage of InterpolateAtArbitraryPoints for both a UFL expression,
        f, and a finite element Function, u:

        # Imports
        import numpy as np
        import matplotlib.pyplot as plt
        from firedrake import *
        from firedrake.__future__ import interpolate

        # Create mesh and function space
        n_elements = 100
        mesh = UnitIntervalMesh(n_elements)
        V = FunctionSpace(mesh, "CG", 1)

        # Define a UFL function
        x = SpatialCoordinate(mesh)
        f = sin(2*pi*x[0])

        # Projection onto a FunctionSpace to create a Function
        u = project(f, V)

        # Create set of points at which to evaluate f or u
        xx = np.linspace(0, 1, 1001).reshape(-1, 1)

        # Create reusable interpolator instance
        point_interpolator = InterpolateAtArbitraryPoints(xx, mesh)

        # Evaluate UFL expression f at points xx
        f_at_points = point_interpolator(f)

        # Evaluate Function u at points xx
        u_at_points = point_interpolator(u)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(xx[:, 0], f_at_points.dat.data, label="f - UFL expression")
        ax.plot(xx[:, 0], u_at_points.dat.data, label="u - Function on FunctionSpace")
        ax.legend();

    """  # noqa: E501

    def __init__(self, points, mesh):
        """Creates a re-usable interpolator instance.

        Args:
        ----
            points (numpy.ndarray): Array with shape (num_points, ndim).
            mesh (firedrake.mesh.MeshGeometry): Parent mesh.

        """  # noqa: D401
        # points should be a numpy array with shape (num_points, ndim)
        # where ndim = 1 for a 1D mesh, 2 for a 2D mesh, 3 for a 3D mesh.
        # Each point should have a corresponding number of values (1 for 1D,
        # 2 for 2D, 3 for 3D).
        assert len(points.shape) == 2  # noqa: S101
        assert points.shape[1] in [1, 2, 3]  # noqa: S101

        self._points = points
        self._mesh = mesh

        self._vom = VertexOnlyMesh(self._mesh, self._points)
        self._P0DG = FunctionSpace(self._vom, "DG", 0)

    def __call__(self, f):
        """Creates a re-usable interpolator instance.

        Args:
        ----
            f (firedrake.function.Function or any valid UFL expression): Function/expression to evaluate at _points.

        Returns:
        -------
            (numpy.ndarray): Function/expression evaluated at _points.

        """  # noqa: D401
        return assemble(interpolate(f, self._P0DG))
