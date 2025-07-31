import firedrake as fd

import finat

import numpy as np



N = 20

mesh = fd.UnitSquareMesh(N, N)

deg_quad = 5

V = fd.FunctionSpace(mesh, "CG", 2)

dx_quad = fd.dx(degree = deg_quad)



# Define a function in V

X, Y = fd.SpatialCoordinate(mesh)

u = fd.Function(V).interpolate(X * (1-X)**0.33 * Y * fd.cos(2 * np.pi * Y) * fd.sin(2 * np.pi * X)) # Random function



# Integrate firedrake expression

Du_exp = fd.sqrt(fd.inner(fd.grad(u), fd.grad(u)))

int_Du_exp = fd.assemble(Du_exp * dx_quad)

print("int_Du :: %.10e" % int_Du_exp)



# Interpolate function onto a quadrature FE space

quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree()) # Gauss-Legendre quad rule for degree p polynomial

quad_FE = fd.FiniteElement("Quadrature", degree=deg_quad, quad_scheme='default')

QD = fd.FunctionSpace(mesh, quad_FE, deg_quad) # The 1 here indicates polynomial degree of the space. It does not change the output below though...

Du_qd = fd.Function(QD).interpolate(Du_exp)



# Integrate again - I want to get same value as int_Du here!

int_Du_qd = fd.assemble(Du_qd * dx_quad)

print("int_Du_qd :: %.10e" % int_Du_qd)


