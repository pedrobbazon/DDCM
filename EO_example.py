import functools



from firedrake import *

from firedrake.adjoint import *



continue_annotation()



class StiffnessOperator(AbstractExternalOperator):



  def __init__(self, *operands, function_space, **kwargs):

      AbstractExternalOperator.__init__(self, *operands, function_space=function_space, **kwargs)



  @assemble_method(0, (0,))

  def assemble_R(self, *args, **kwargs):

      Id4 = np.einsum("ij,kl", np.eye(2), np.eye(2))

      I4 = 0.5 * (np.einsum("ik,jl", np.eye(2), np.eye(2)) + np.einsum("il,jk", np.eye(2), np.eye(2)))



      stiffness = Function(self.function_space()).interpolate(as_tensor((2884.61 * Id4

                                              + 2 * 1923.07 * I4)))



      return stiffness



mesh = UnitSquareMesh(50, 50)



Vf = TensorFunctionSpace(mesh, "CG", 1, shape=(2, 2, 2, 2))

f = Function(Vf)

R = functools.partial(StiffnessOperator, function_space=Vf)





Vu = VectorFunctionSpace(mesh, "CG", 1)

u_trial = TrialFunction(Vu)

u_test = TestFunction(Vu)

u = Function(Vu)



i,j,k,l = indices(4)

a = inner(as_tensor(R(f)[i,j,k,l] * (sym(grad(u_trial)))[k, l], (i, j)), (sym(grad(u_test)))) * dx

L = dot(Constant((0, 0)), u_test) * dx



# this works
solve(a == L, u)



# this doesn't
#prob = LinearVariationalProblem(a, L, u)
#solver = LinearVariationalSolver(prob)
#solver.solve()
