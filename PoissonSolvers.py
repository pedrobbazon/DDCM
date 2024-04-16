import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
from firedrake.__future__ import interpolate

petsc_opt = {
    'snes_type': 'newtonls',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_linesearch_type': 'basic',
    'snes_max_it': 100,
    'snes_divergence_tolerance': 1e20,
    'snes_rtol': 1e-9,
    'snes_atol': 1e-11,
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

#
# User defined utility functions to set your standard model 
# (constitutive and numerical parameters, flux laws, fem formulations, etc)
#
# Program your Flux Law: (Mixed or Primal): please, follow format for the names: FLX_mymodel())
def FLM_DarcyLinear(self, potential, flux):
    kappa = self.params[0]
    return flux + kappa * grad(potential)

def FLM_ForcheimerReg(self, potential, flux):
    eta = self.params[0]
    muokap = self.params[1]
    betarho = self.params[2]
    modreg = eta*(sqrt(1.0 + inner(flux,flux)/eta**2) - 1.0)
    return grad(potential) + muokap*flux + betarho * modreg * flux

def FLM_Klinkenberg(self, potential, flux):
    kapomu = self.params[0]
    DK = self.params[1]
    return flux + kapomu * grad(potential) + (DK/potential)*grad(potential)
    
def FLM_DarcyNonLinear(self, potential, flux):
    def Kofu(u, params):
        kappa0 = params[0]
        viol = params[1]
        return kappa0*(1 + 10*0.5 * (1.0 + tanh(viol*(u-2.0))))
    return flux + Kofu(potential,self.params) * grad(potential)
    
def FLP_DarcyLinear(self, potential):
    kappa = self.params[0]
    return -kappa * grad(potential)

#-------------------------------------------------------------------------------------
# Program your FEM formulation (Mixed or Primal): please, follow format for the names: 
def PoissonDualMixedHdiv(domain, flag_quads, flag_setting, sourceval, bcvals,
                         FluxLaw, params, petsc_opt, orderF=2, orderU=1):
    print( "|> Solving with Dual-Mixed formulation\n")

    mesh = domain.msh

    if(flag_quads):
        HD = fd.FunctionSpace(mesh, "RTCF", orderF)
        CG = fd.FunctionSpace(mesh, "CG", orderU)
    else:
        HD = fd.FunctionSpace(mesh, "DRT", orderF)
        CG = fd.FunctionSpace(mesh, "CG", orderU)

    W = HD * CG

    w = fd.Function(W)
    sigma, u = fd.split(w)
    tau, v = fd.TestFunctions(W)

    x, y = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

    if(flag_setting == 'Full-Setting'):
        f = fd.Constant(sourceval)
        bcall = [fd.DirichletBC(W.sub(1), fd.Constant(bcsvals), 'on_boundary')]
    elif(flag_setting == 'Uniaxial'):
        f = fd.Constant(0.0)
        pbottom, ptop = fd.Constant(1.0), fd.Constant(3.0)
        flag_bottom, flag_top = 3, 4
        bcall = [fd.DirichletBC(W.sub(1), pbottom, flag_bottom), fd.DirichletBC(W.sub(1), ptop, flag_top)]
    else:
        raise ValuerError('Unrecognized problem setting option')
    
    Res  = (inner(FluxLaw(sigma,u, params),tau))*dx
    Res += inner(sigma,grad(v))*dx
    Res += f*v*fd.dx
    fd.solve(Res == 0, w, bcs=bcall, solver_parameters=petsc_opt)
    return w.sub(0), w.sub(1), fd.project(grad(u), domain.VDG0)
#-------------------------------------------------------------------------------------