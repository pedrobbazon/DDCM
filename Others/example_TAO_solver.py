from firedrake import *

from firedrake.adjoint import *

import petsc4py.PETSc as PETSc

class TAOsolver():

    def __init__(self):

        # set up

        mesh = UnitSquareMesh(10,10)

        V = FunctionSpace(mesh, "DG", 0)

        x, y = SpatialCoordinate(mesh)

        u_ = Function(V).interpolate(x**2+y)  #target soln

        self.f = Function(V)  # control

        u = Function(V)  # state variable

        v = TestFunction(V)

        F = (u - self.f)*v*dx  # state constraint

        # annotate solve and define reduced functional
        continue_annotation()

        solve(F == 0, u)

        J = assemble((u - u_)**2*dx)

        self.Jhat = ReducedFunctional(J, Control(self.f))

        pause_annotation()

        # control and gradient vectors for TAO

        with self.f.dat.vec_ro as f:

            x = f.duplicate()  # control

            g = f.duplicate()  # gradient

        solver = PETSc.TAO().create()

        solver.setType("lmvm")  # limited-memory, variable-metric method

        solver.setFromOptions()

        solver.setObjectiveGradient(self.myObjGrad, g)

        solver.setTolerances(gatol=1.0e-14,grtol=1.0e-14)

        solver.solve(x)

        tape = get_working_tape()

        tape.clear_tape()

        with self.f.dat.vec_wo as f:

            f.copy(x)

        print("norm(f) = ", norm(self.f))

        print("J = ", norm(self.f - u_)**2)

    def myObjGrad(self, tao, x, g):

        with self.f.dat.vec_wo as f:

            x.copy(f)

        J = self.Jhat.__call__(self.f)

        dJ = self.Jhat.derivative()
        #dJ = self.Jhat.derivative(options={"riesz_representation": None})

        with dJ.dat.vec_ro as dJ:

            dJ.copy(g)

        return J

if __name__=="__main__":

    TAOsolver()

