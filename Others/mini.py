from firedrake import *
from pyop2.datatypes import IntType
import numpy
import petsc4py
from mpi4py import MPI
import os
petsc4py.PETSc.Sys.popErrorHandler()

from datetime import datetime

import alfi
from alfi.stabilisation import *
from alfi.transfer import *

from tabulate import tabulate
import ufl.algorithms

# Define two-dimensional versions of cross and curl operators
def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]


def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])


def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])


def acurl(x):
    return as_vector([
                     x[2].dx(1),
                     -x[2].dx(0),
                     x[1].dx(0) - x[0].dx(1)
                  ])

# Definition of Burman Stabilisation term to avoid oscillations in velocity u
def BurmanStab(B, C, wind, stab_weight, mesh):
    n = FacetNormal(mesh)
    h = FacetArea(mesh)
    beta = avg(facet_avg(sqrt(inner(wind, wind)+1e-10)))
    gamma1 = stab_weight  # as chosen in doi:10.1016/j.apnum.2007.11.001
    stabilisation_form = 0.5 * gamma1 * avg(h)**2 * beta * dot(jump(grad(B), n), jump(grad(C), n))*dS
    return stabilisation_form


# monitor
def div_B(B):
    return norm(div(B), "L2")

def div_u(u):
    return norm(div(u), "L2")

def build_nonlinear_solver(F, z, bcs, Jp=None, solver_parameters=None, options_prefix=None):
    problem = NonlinearVariationalProblem(F, z, bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(problem,
                                        solver_parameters=solver_parameters,
                                        options_prefix=options_prefix)
    return solver

class PressureFixBC(DirichletBC):
    def __init__(self, V, val, subdomain, method="topological"):
        super().__init__(V, val, subdomain, method)
        sec = V.dm.getDefaultSection()
        dm = V.mesh().topology_dm

        coordsSection = dm.getCoordinateSection()
        dim = dm.getCoordinateDim()
        coordsVec = dm.getCoordinatesLocal()

        (vStart, vEnd) = dm.getDepthStratum(0)
        indices = []
        for pt in range(vStart, vEnd):
            x = dm.getVecClosure(coordsSection, coordsVec, pt).reshape(-1, dim).mean(axis=0)
            if x.dot(x) == 0.0:  # fix [0, 0] in original mesh coordinates (bottom left corner)
                if dm.getLabelValue("pyop2_ghost", pt) == -1:
                    indices = [pt]
                break

        nodes = []
        for i in indices:
            if sec.getDof(i) > 0:
                nodes.append(sec.getOffset(i))

        if V.mesh().comm.rank == 0:
            nodes = [0]
        else:
            nodes = []
        self.nodes = numpy.asarray(nodes, dtype=IntType)

        if len(self.nodes) > 0:
            print("Fixing nodes %s" % self.nodes)
        import sys
        sys.stdout.flush()


# remember to use outerSchur0
class SchurPC(AssembledPC):
    def form(self, pc, test, trial):
        a_old, bcs = AssembledPC.form(self, pc, test, trial)
        [B, j, sigma] = split(trial)
        [Bt, jt, sigmat] = split(test)

        state = self.get_appctx(pc)['state']
        [u_n, p_n, B_n, j_n, sigma_n] = split(state)
        A = (
              a_old
            + alpha * inner(j, jt) * dx
            + alpha * inner(B, vcurl(jt)) * dx
            + alpha * inner(vcurl(j), Bt) * dx
            - alpha * inner(vcurl(sigma), Bt) * dx
            - S * inner(scross(u_n, B), sigmat) * dx
            + S * inner(sigma, sigmat) * dx
            + gamma2 * inner(div(B), div(Bt)) * dx
        )

        return (A, bcs)


# solver parameters
# Increase buffer size for MUMPS solver
ICNTL_14 = 5000
tele_reduc_fac = int(MPI.COMM_WORLD.size/4)
if tele_reduc_fac < 1:
    tele_reduc_fac = 1

#main solver for fieldsplit
fs2by3={
    "mat_type": "nest", 
    "snes_type": "newtonls",
    "snes_monitor": None,
    "ksp_type": "fgmres", 
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit", 
    "pc_fieldsplit_type": "schur", 
    "pc_fieldsplit_0_fields": "0, 1",
    "pc_fieldsplit_1_fields": "2, 3, 4",
    "pc_fieldsplit_schur_precondition": "a11", 
    "pc_fieldsplit_schur_fact_type": "full", 
}


# solver for (u, p) block: schur complement inverse with massinv + F-cycle on augmented block
nsfstar = {
        "ksp_type": "fgmres",
        "ksp_atol": 1.0e-7,
        "ksp_rtol": 1.0e-7,
        "ksp_max_it": 2,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur", 
        "pc_fieldsplit_schur_precondition": "a11",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0": {
             #"ksp_type": "preonly",
            #"pc_type": "lu",
            #"pc_factor_mat_solver_type": "mumps",
            "ksp_type": "gmres",
            "ksp_max_it": 1,
            "ksp_norm_type": "unpreconditioned",
            "pc_type": "mg",
            "pc_mg_cycle_type": "v",
            "pc_mg_type": "full",
            "mg_levels_ksp_type": "fgmres",
            "mg_levels_ksp_convergence_test": "skip",
            "mg_levels_ksp_max_it": 6,
            "mg_levels_ksp_norm_type": "unpreconditioned",
            "mg_levels_pc_type": "python",
            "mg_levels_pc_python_type": "firedrake.ASMStarPC",
            "mg_levels_pc_star_backend": "tinyasm",
            #"mg_levels_pc_star_construct_dim": 0,
            #"mg_levels_pc_star_sub_sub_ksp_type": "preonly",
            #"mg_levels_pc_star_sub_sub_pc_type": "lu",
            #"mg_levels_pc_star_sub_sub_pc_factor_mat_solver_type": "umfpack",
            "mg_coarse_ksp_type": "richardson",
            "mg_coarse_ksp_max_it": 1,
            "mg_coarse_ksp_norm_type": "unpreconditioned",
            "mg_coarse_pc_type": "python",
            "mg_coarse_pc_python_type": "firedrake.AssembledPC",
            "mg_coarse_assembled": {
            "mat_type": "aij",
            "pc_type": "telescope",
            "pc_telescope_reduction_factor": tele_reduc_fac,
            "pc_telescope_subcomm_type": "contiguous",
            "telescope_pc_type": "lu",
            "telescope_pc_factor_mat_solver_type": "mumps",
            "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
         }
    },

        "fieldsplit_1":{
                "ksp_type": "preonly",
                "pc_type": "python",
                #"pc_python_type": __name__ + ".MassInv",
                "pc_python_type": "alfi.solver.DGMassInv",
                "assembled_pc_type": "lu",
                "assembled_pc_factor_mat_solver_type": "mumps",
        },

}

# solver for Outer Schur complement block (B, j, sigma): monolithic F-cycle multigrid
outerSchur0 = {
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "pc_type": "python",
        "pc_python_type": __name__ + ".SchurPC", 
        "assembled_mg_transfer_manager": __name__ + ".transfermanager",
        #"assembled_pc_type": "lu", 
        #"assembled_pc_factor_mat_solver_type": "mumps", 
        "ksp_max_it": 2, 
        "assembled":{
            "ksp_type": "gmres",
            "ksp_max_it": 1, 
            "ksp_norm_type": "unpreconditioned",
            "pc_type": "mg", 
            "pc_mg_cycle_type": "v",
            "pc_mg_type": "full",
            "mg_coarse": {
               "ksp_type": "richardson",
               "ksp_mat_it": 1, 
               "ksp_norm_type": "unpreconditioned",
               "pc_type": "python",
               "pc_python_type": "firedrake.AssembledPC",
               "assembled": {
                     "mat_type": "aij",
                     "pc_type": "telescope",
                     "pc_telescope_reduction_factor": tele_reduc_fac,
                     "pc_telescope_subcomm_type": "contiguous",
                     "telescope_pc_type": "lu",
                     "telescope_pc_factor_mat_solver_type": "mumps",
                     "telescope_pc_factor_mat_mumps_icntl_14": ICNTL_14,
               }
            },
            "mg_levels": {
                "ksp_type": "fgmres",
                "ksp_convergence_test": "skip",
                "ksp_mat_it": 6, 
                "ksp_norm_type": "unpreconditioned",
                "pc_type": "python", 
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_backend": "tinyasm",
                #"pc_python_type": "firedrake.PatchPC",
                #"patch_pc_patch_save_operators": True, 
#"patch_pc_patch_precompute_element_tensors": True, 
#                "patch_pc_patch_construct_type": "star",
#                "patch_pc_patch_construct_dim": 0, 
#                "patch_pc_patch_sub_mat_type": "seqdense",
#                "patch_sub_ksp_type": "preonly",
#                "patch_sub_pc_type": "lu",
            },
        },
       
}


params = fs2by3
params["fieldsplit_0"] = nsfstar
params["fieldsplit_1"] = outerSchur0

# input list for continuation
res = [1]
rems = [1]
Ss = [1]

####  parameters
# mesh parameters
baseN = 4
nref = 1
k = 1
hierarchy = "uniform" 
discr = "rt"
dp = {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
# physical parameters
Re = Constant(res[0])
Rem = Constant(rems[0])
S = Constant(Ss[0])
alpha = S/Rem
gamma = Constant(10000)
gamma2 = S/Rem
advect = Constant(1)
# problem setup
linearisation = "newton"
testproblem = "ldc"
stab = True
# Stabilisation weight for BurmanStabilisation
stab_weight = Constant(3e-3)
# boundary condition
bcs_ids_apply = 4
bcs_ids_dont_apply = (1, 2, 3)

##### end of parameters
base = UnitSquareMesh(baseN, baseN, distribution_parameters = dp)
mh = MeshHierarchy(base, nref)
mesh = mh[-1]
(x, y) = SpatialCoordinate(mesh)

Vd = FunctionSpace(mesh, "RT", k)
Vc = FunctionSpace(mesh, "N1curl", k)
Vg = VectorFunctionSpace(mesh, "CG", k)
Vn = FunctionSpace(mesh, "DG", k-1)
Vg_ = FunctionSpace(mesh, "CG", k)

# used for BurmanStabilisation
z_last_u = Function(Vd)

# (u, p, B, j, sigma)
Z = MixedFunctionSpace([Vd, Vn, Vd, Vg_, Vg_])
z = Function(Z)
z_test = TestFunction(Z)

(u, p, B, j, sigma) = split(z)
(ut, pt, Bt, jt, sigmat) = split(z_test)

u_ex = Constant((1, 0))
B_init = Constant((0, 1))
B_ex = Function(Vd).interpolate(B_init)

eps = lambda x: sym(grad(x))
F = (

      2/Re * inner(eps(u), eps(ut)) *dx 
    - inner(p, div(ut)) * dx
    + S * inner(vcross(B, j), ut) * dx
    + gamma * inner(div(u), div(ut)) * dx # AL term

    - inner(div(u), pt) * dx
    + alpha * inner(vcurl(j), Bt) * dx
    - alpha * inner(vcurl(sigma), Bt) * dx
    + gamma2 * inner(div(B), div(Bt)) * dx # AL term

    + S * inner(sigma, sigmat) * dx
    - S * inner(scross(u, B), sigmat) * dx
    - S * inner(j, jt) * dx
    + alpha * inner(B, vcurl(jt)) * dx
    )

n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])

w = TrialFunction(Z)
[w_u, w_p, w_B, w_j, w_sigma] = split(w) #where the operator acts on
J_newton = ufl.algorithms.expand_derivatives(derivative(F, z, w))

if linearisation == "newton":
    J = J_newton
elif linearisation == "picard":
    J_picard = (
        J_newton
    + S*inner(scross(u, w_B), sigmat) * dx
    - S*inner(vcross(w_B, j), ut) * dx
    - inner(dot(grad(w_u), u), ut) * dx
    )
    J = J_picard


if discr in ["rt"]:
    h = CellVolume(mesh)/FacetArea(mesh)
    sigma = Constant(10) * Z.sub(0).ufl_element().degree()**2  # penalty term
    theta = Constant(1)  # theta=1 means upwind discretization of nonlinear advection term
    uflux_int = 0.5*(dot(u, n) + theta*abs(dot(u, n))) * u
    uflux_ext_1 = 0.5*(inner(u, n) + theta*abs(inner(u, n))) * u
    uflux_ext_2 = 0.5*(inner(u, n) - theta*abs(inner(u, n))) * u_ex

    F_DG = (
         - 1/Re * inner(avg(2*sym(grad(u))), 2*avg(outer(ut, n))) * dS
         - 1/Re * inner(avg(2*sym(grad(ut))), 2*avg(outer(u, n))) * dS
         + 1/Re * sigma/avg(h) * inner(2*avg(outer(u, n)), 2*avg(outer(ut, n))) * dS
         - inner(outer(ut, n), 2/Re*sym(grad(u))) * ds
         - inner(outer(u-u_ex, n), 2/Re*sym(grad(ut))) * ds(bcs_ids_apply)
         + 1/Re*(sigma/h)*inner(ut, u-u_ex) * ds(bcs_ids_apply)
         - advect * dot(u, div(outer(ut, u))) * dx
         + advect * dot(ut('+')-ut('-'), uflux_int('+')-uflux_int('-')) * dS
         + advect * dot(ut, uflux_ext_1) * ds
         + advect * dot(ut, uflux_ext_2) * ds(bcs_ids_apply)
    )

    if bcs_ids_dont_apply is not None:
        F_DG += (
            - inner(outer(u, n), 2/Re*sym(grad(ut))) * ds(bcs_ids_dont_apply)
            + 1/Re*(sigma/h)*inner(ut, u) * ds(bcs_ids_dont_apply)
           )

    F += F_DG

# Add Burman Stabilisation
if stab:
    initial = Function(Vd).interpolate(as_vector([sin(y), x]))
    z_last_u.assign(initial)
    stabilisation = BurmanStabilisation(Z.sub(0), state=z_last_u, h=FacetArea(mesh), weight=stab_weight)
    stabilisation_form_u = stabilisation.form(u, ut)
    F += (advect * stabilisation_form_u)


bcs = [
       DirichletBC(Z.sub(0), u_ex, (4,)),
       DirichletBC(Z.sub(0), 0, (1, 2, 3, )),
       DirichletBC(Z.sub(2), B_ex, "on_boundary"),
       DirichletBC(Z.sub(3), 0, "on_boundary"),
       DirichletBC(Z.sub(4), 0, "on_boundary"), #important
       PressureFixBC(Z.sub(1), 0, 1),
       ]

# if I pass J=J, it is difficult to converge
# if I remove J=J, it converges
problem = NonlinearVariationalProblem(F, z, bcs, J=J)
appctx = {"Re": Re, "gamma": gamma, "nu": 1 / Re, "Rem": Rem, "gamma2": gamma2}
solver = NonlinearVariationalSolver(problem, solver_parameters=params, options_prefix="", appctx=appctx)
qtransfer = NullTransfer()
Etransfer = NullTransfer()
vtransfer = SVSchoeberlTransfer((1 / Re, gamma), 2, hierarchy)
dgtransfer = DGInjection()
transfers = {
                Vn.ufl_element(): (prolong, restrict, qtransfer.inject),
                Vg_.ufl_element(): (prolong, restrict, Etransfer.inject),
                VectorElement("DG", mesh.ufl_cell(), k): (dgtransfer.prolong, restrict, dgtransfer.inject),
            }
transfermanager = TransferManager(native_transfers=transfers)
solver.set_transfer_manager(transfermanager)
solver.solve()

