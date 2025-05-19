import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 20)
from firedrake import *
from firedrake.adjoint import *

# Solve the linear systems at time step with NonlinearVariationalSolver (when True) 
# or as linear problems with solve (when False) 
nonlinear_solver = True

############
#    SETUP
############

# Define mesh and function spaces
n = 32
mesh = PeriodicUnitIntervalMesh(n)
V = FunctionSpace(mesh, "Lagrange", 1)
R = FunctionSpace(mesh, "R", 0)
v = TestFunction(V)
vTr = TrialFunction(V)

# Create Functions
u = Function(V, name="Velocity")
u_ = Function(V)
lamda = Function(V, name = "Adjoint") 
lamda_ = Function(V)

# Assign initial condition to u
x = SpatialCoordinate(mesh)[0]
u_ = Function(V).interpolate(sin(2 * pi*x))
u.assign(u_)

# Set constants
w = Constant(1)
nu = Function(R)
nu.assign(0.01)

dt = 1.0/n

# Define solvers for forward and adjoint in two different ways
def forward_bilinear_form(trial, test):
    return (inner(trial, test) + dt * inner(w * trial.dx(0), test) + dt * nu*inner(grad(trial), grad(test)))*dx
forward_linear_form = inner(u_, v) *dx

def adjoint_bilinear_form(trial, test):
    return (inner(trial, test) - dt * inner(w * trial.dx(0), test) + dt * nu * inner(grad(trial), grad(test)))*dx
adjoint_linear_form = inner(lamda_, v) *dx

if nonlinear_solver:

    # forward solver
    F_fw = forward_bilinear_form(u, v) - forward_linear_form
    problem = NonlinearVariationalProblem(F_fw, u)
    solver = NonlinearVariationalSolver(problem)
    def solve_forward():
        solver.solve()

    # adjoint solver
    F_adj = adjoint_bilinear_form(lamda, v) - adjoint_linear_form
    problem_adj = NonlinearVariationalProblem(F_adj, lamda)
    solver_adj = NonlinearVariationalSolver(problem_adj)
    def solve_adj():
        solver_adj.solve() 

else:      
    def solve_forward():
        solve(forward_bilinear_form(vTr, v) == forward_linear_form, u)
    def solve_adj():
        solve(adjoint_bilinear_form(vTr, v) == adjoint_linear_form, lamda)      
    
# Timestepping details
end_time = 0.5
num_timesteps = int(end_time/dt)
print("Number of time steps :: %d" % num_timesteps)

############
#    FORWARD PROBLEM
############
print("Solving forward problem...")

forward_solutions = [u.copy(deepcopy=True)]
i = 0
t = 0.0

continue_annotation()
while (t < end_time): # start time stepping forward problem
    solve_forward()
    u_.assign(u)
    t += dt
    i += 1
    forward_solutions.append(u.copy(deepcopy=True))

# Compute gradient of J automatically
J_form = inner(u, u)*dx
J = assemble(J_form)
g = compute_gradient(J, Control(nu))
dJ_dnu_auto = g.dat.data_ro[0]
print("(automatic) Gradient of J w.r.t. diffusivity = %.4f" % dJ_dnu_auto)


############
#    ADJOINT PROBLEM (automatic)
############

stop_annotating()
from firedrake.adjoint import get_solve_blocks
solve_blocks = get_solve_blocks()  # We have to do this before manual adj solution - if not, it appends the solve_blocks from adjoint computations too!
assert len(solve_blocks) == num_timesteps
for block in solve_blocks:
    assert block.adj_sol is not None

# 'Initial condition' for both adjoint
dJdu = assemble(derivative(J_form, u))

# time step for adjoint
adjoint_solutions_alt = []
for i in range(num_timesteps+1):
    if i == num_timesteps:
        adjoint_solution = dJdu.riesz_representation(riesz_map="L2")
    else:
        adjoint_solution = solve_blocks[i].adj_sol
    adjoint_solutions_alt.append(adjoint_solution.copy(deepcopy = True))


############
#    ADJOINT PROBLEM (manual)
############

lamda_.interpolate(2 * forward_solutions[-1])
adjoint_solutions = [lamda_.copy(deepcopy = True)]
t = 0.0
print("Solving adjoint problem...")
while (t < end_time):
    solve_adj()
    lamda_.assign(lamda)
    t += dt
    i += 1
    adjoint_solutions.append(lamda.copy(deepcopy=True))
adjoint_solutions.reverse()

# Compute dJ/dnu manually
dJ_dnu = 0.
t = 0.0
i = 0
while (t < end_time):
    dJ_dnu += -dt * assemble(forward_solutions[i + 1].dx(0) * adjoint_solutions[i].dx(0) * dx) 
    t += dt
    i += 1

print("(Manual comp.) Gradient of J w.r.t. diffusivity = {:.4f}".format(dJ_dnu))

# Check difference between adjoint solutions
diff = 0
for i in range(num_timesteps+1):
    diff += assemble((adjoint_solutions[i] - adjoint_solutions_alt[i])**2 * dx)
print("L2 difference between manual and automatic adjoint computations :: %.4e" % diff)


############
#       PLOT
############
timesteps_per_plot = 4
num_plots = num_timesteps//timesteps_per_plot
fig, axs = plt.subplots(num_plots+1, sharex='col', sharey = True)
axs[0].set_xlim([0,1])
for i in range(num_plots + 1):
    plot(forward_solutions[i * timesteps_per_plot], axes=axs[i], label = "forward")
    plot(adjoint_solutions[i * timesteps_per_plot], axes=axs[i], label = "adjoint (manual)")
    plot(adjoint_solutions_alt[i * timesteps_per_plot], axes=axs[i], label = "adjoint (AD)")
    axs[i].set_title('t={:.2f}'.format(i*timesteps_per_plot*dt))
axs[0].legend(fontsize = 7)
plt.show()












