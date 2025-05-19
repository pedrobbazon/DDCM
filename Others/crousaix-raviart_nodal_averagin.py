import firedrake as fd

import finat
import numpy as np

def vertex_quadrature(ref_el):

    x = ref_el.vertices
    w = np.ones(len(x)) # Just sum all the values at the vertices

    return finat.quadrature.QuadratureRule(point_set=finat.point_set.PointSet(x), weights=w)

mesh = ... #Some mesh

CR = fd.FunctionSpace(mesh, "CR", 1)
CG = fd.FunctionSpace(mesh, "CG", 1)

uh = ...               # A given CR Function
uavg = fd.Function(CG) # The nodal interpolant

quad_rule_vertex = vertex_quadrature(CR.finat_element.cell)
dxvertex = fd.dx(scheme=quad_rule_vertex)

# The determinant of the Jacobian of the affine maping (in 2D) is 2|K|, so we should cancel that contribution
detJ = 2 * fd.CellVolume(mesh)

u = fd.TrialFunction(CG)
v = fd.TestFunction(CG)

fd.solve((1./detJ) * u * v * dxvertex == (1./detJ) * uh * v * dxvertex, uavg)
